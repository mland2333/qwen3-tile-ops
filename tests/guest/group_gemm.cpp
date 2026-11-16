// per-group GEMM 原语测试：
//   32×256 i8  @  128×256 i8，K 按 128 分两组
//   对照组：两次 per-token gemm_nt（各 K=128）+ f32 相加
// 两组 scale 都是 1.0，所以两组路径应逐位相同，用来验证：
//   - int32 partial 每组清零、4 个 32 列块独立累加
//   - adequant 的 per-group a_scale / per-channel b_scale
//   - 12 个 acc 的预算（4 int + 4 f32 partial + 4 f32 accum）
#include <cstdint>

#include "../kernels/gemm.h"
#include "../kernels/group_gemm.h"
#include "../kernels/mlp.h"
#include "../kernels/quant.h"

namespace {

constexpr int ROWS = 32;
constexpr int K = 256;
constexpr int N = 128;
constexpr int GROUPS = 2;

constexpr std::uintptr_t A_OFF = 0x0000;
constexpr std::uintptr_t B_OFF = 0x2000;
constexpr std::uintptr_t AS_OFF = 0x10000;
constexpr std::uintptr_t BS_OFF = 0x10200;
constexpr std::uintptr_t OUTG_OFF = 0x11000;
constexpr std::uintptr_t OUTR_OFF = 0x15000;
constexpr std::uintptr_t OUTR1_OFF = 0x19000;
constexpr std::uintptr_t RESULT_OFF = 0x1D000;
constexpr std::uintptr_t X_OFF = 0x20000;
constexpr std::uintptr_t QI8_OFF = 0x28000;
constexpr std::uintptr_t REFQI8_OFF = 0x2A000;
constexpr std::uintptr_t QS_OFF = 0x2C000;
constexpr std::uintptr_t REFS0_OFF = 0x2C100;
constexpr std::uintptr_t REFS1_OFF = 0x2C200;
constexpr std::uintptr_t RESULT2_OFF = 0x2C300;

} // namespace

int main() {
  std::uintptr_t base;
  asm volatile("mv %0, x31" : "=r"(base));
  auto *blob = reinterpret_cast<std::byte *>(base);
  auto *a = reinterpret_cast<std::int8_t *>(blob + A_OFF);
  auto *b = reinterpret_cast<std::int8_t *>(blob + B_OFF);
  auto *as = reinterpret_cast<std::uint32_t *>(blob + AS_OFF);
  auto *bs = reinterpret_cast<std::uint32_t *>(blob + BS_OFF);

  // 确定性数据，值域很小，int32 点积不会超过 f32 的精确整数范围。
  for (int i = 0; i < ROWS; ++i)
    for (int k = 0; k < K; ++k)
      a[i * K + k] = static_cast<std::int8_t>((i * 7 + k * 3 + 1) % 7 - 3);
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k)
      b[j * K + k] = static_cast<std::int8_t>((j * 5 + k * 2 + 2) % 5 - 2);
  for (int i = 0; i < ROWS; ++i)
    for (int g = 0; g < GROUPS; ++g)
      as[i * GROUPS + g] = 0x3F800000u;   // 1.0f
  for (int j = 0; j < N; ++j) bs[j] = 0x3F800000u;

  const qwen3_ops::MatrixView av{reinterpret_cast<std::byte *>(a), K, 1, ROWS, K,
                                 qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView bv{reinterpret_cast<std::byte *>(b), K, 1, N, K,
                                 qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView asv{reinterpret_cast<std::byte *>(as),
                                  GROUPS * 4, 4, ROWS, GROUPS,
                                  qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView bsv{reinterpret_cast<std::byte *>(bs),
                                  4, 4, N, 1, qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView outg{blob + OUTG_OFF, N * 4, 4, ROWS, N,
                                   qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView outr{blob + OUTR_OFF, N * 4, 4, ROWS, N,
                                   qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView outr1{blob + OUTR1_OFF, N * 4, 4, ROWS, N,
                                    qwen3_ops::kDtF32};

  qwen3_ops::gemm_nt_group_store(av, asv, bv, bsv, outg);

  // 对照组：group 0 和 group 1 各做一次 K=128 的 per-token GEMM，再 f32 相加。
  const qwen3_ops::MatrixView a0 =
      qwen3_ops::sub_view(av, 0, 0, ROWS, 128);
  const qwen3_ops::MatrixView b0 =
      qwen3_ops::sub_view(bv, 0, 0, N, 128);
  const qwen3_ops::MatrixView a1 =
      qwen3_ops::sub_view(av, 0, 128, ROWS, 128);
  const qwen3_ops::MatrixView b1 =
      qwen3_ops::sub_view(bv, 0, 128, N, 128);
  const qwen3_ops::MatrixView as0 =
      qwen3_ops::sub_view(asv, 0, 0, ROWS, 1);
  const qwen3_ops::MatrixView as1 =
      qwen3_ops::sub_view(asv, 0, 1, ROWS, 1);
  qwen3_ops::gemm_nt(a0, b0, bsv, as0, outr);
  qwen3_ops::gemm_nt(a1, b1, bsv, as1, outr1);
  qwen3_ops::add_rows(outr, outr1);

  const auto *g = reinterpret_cast<const std::uint32_t *>(blob + OUTG_OFF);
  const auto *r = reinterpret_cast<const std::uint32_t *>(blob + OUTR_OFF);
  std::uint32_t fails = 0, first = 0xFFFFFFFFu, got = 0, ref = 0;
  for (int i = 0; i < ROWS * N; ++i) {
    if (g[i] != r[i]) {
      if (fails == 0) {
        first = static_cast<std::uint32_t>(i);
        got = g[i];
        ref = r[i];
      }
      ++fails;
    }
  }
  auto *res = reinterpret_cast<std::uint32_t *>(blob + RESULT_OFF);
  res[0] = fails;
  res[1] = first;
  res[2] = got;
  res[3] = ref;

  // ---- per-head quant：32×256 f32，按 128 分 2 组 ----
  auto *xv = reinterpret_cast<std::uint32_t *>(blob + X_OFF);
  auto f32_bits = [](int n) -> std::uint32_t {
    switch (n) {
    case 0: return 0x3F800000u;  // 1.0
    case 1: return 0x40000000u;  // 2.0
    case 2: return 0x40400000u;  // 3.0
    case 3: return 0x40800000u;  // 4.0
    case 4: return 0x40A00000u;  // 5.0
    case 5: return 0x40C00000u;  // 6.0
    default: return 0x40E00000u; // 7.0
    }
  };
  for (int i = 0; i < ROWS * K; ++i) {
    xv[i] = f32_bits(i % 7) | ((i & 1) ? 0x80000000u : 0u);
  }

  const qwen3_ops::MatrixView xv_m{blob + X_OFF, K * 4, 4, ROWS, K,
                                   qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView qi8_m{blob + QI8_OFF, K, 1, ROWS, K,
                                    qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView refqi8_m{blob + REFQI8_OFF, K, 1, ROWS, K,
                                       qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView qs_m{blob + QS_OFF, GROUPS * 4, 4, ROWS, GROUPS,
                                   qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView rs0_m{blob + REFS0_OFF, 4, 4, ROWS, 1,
                                    qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView rs1_m{blob + REFS1_OFF, 4, 4, ROWS, 1,
                                    qwen3_ops::kDtF32};

  qwen3_ops::quant_rows_heads(xv_m, qi8_m, qs_m, 128);
  qwen3_ops::quant_rows(qwen3_ops::sub_view(xv_m, 0, 0, ROWS, 128),
                        qwen3_ops::sub_view(refqi8_m, 0, 0, ROWS, 128), rs0_m);
  qwen3_ops::quant_rows(qwen3_ops::sub_view(xv_m, 0, 128, ROWS, 128),
                        qwen3_ops::sub_view(refqi8_m, 0, 128, ROWS, 128), rs1_m);

  const auto *qg = reinterpret_cast<const std::uint8_t *>(blob + QI8_OFF);
  const auto *qr = reinterpret_cast<const std::uint8_t *>(blob + REFQI8_OFF);
  std::uint32_t qfails = 0, qfirst = 0xFFFFFFFFu, qgot = 0, qref = 0;
  for (int i = 0; i < ROWS * K; ++i) {
    if (qg[i] != qr[i]) {
      if (qfails == 0) {
        qfirst = static_cast<std::uint32_t>(i);
        qgot = qg[i];
        qref = qr[i];
      }
      ++qfails;
    }
  }
  const auto *sg = reinterpret_cast<const std::uint32_t *>(blob + QS_OFF);
  const auto *s0 = reinterpret_cast<const std::uint32_t *>(blob + REFS0_OFF);
  const auto *s1 = reinterpret_cast<const std::uint32_t *>(blob + REFS1_OFF);
  for (int i = 0; i < ROWS; ++i) {
    for (int g = 0; g < GROUPS; ++g) {
      const std::uint32_t ref_s = (g == 0) ? s0[i] : s1[i];
      if (sg[i * GROUPS + g] != ref_s) {
        if (qfails == 0) {
          qfirst = static_cast<std::uint32_t>(ROWS * K + i * GROUPS + g);
          qgot = sg[i * GROUPS + g];
          qref = ref_s;
        }
        ++qfails;
      }
    }
  }
  auto *res2 = reinterpret_cast<std::uint32_t *>(blob + RESULT2_OFF);
  res2[0] = qfails;
  res2[1] = qfirst;
  res2[2] = qgot;
  res2[3] = qref;
  return 0;
}
