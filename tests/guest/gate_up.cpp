// gate/up + SwiGLU + quant fusion isolated test.
// Compares the old two-GEMM + swiglu_quant_heads path against the fused
// one-block gate/up path for 32x512 A, 256x512 gate/up, 256-col groups.
#include <cstdint>

#include "../kernels/gemm.h"
#include "../kernels/group_gemm.h"
#include "../kernels/swiglu_quant.h"
#include "../kernels/mlp.h"

namespace {

constexpr int ROWS = 32;
constexpr int K = 512;
constexpr int N = 256;
constexpr int GROUPS = 2;

constexpr std::uintptr_t A_OFF = 0x0000;
constexpr std::uintptr_t WG_OFF = 0x4000;
constexpr std::uintptr_t WU_OFF = 0x24000;
constexpr std::uintptr_t AS_OFF = 0x44000;
constexpr std::uintptr_t WGS_OFF = 0x44100;
constexpr std::uintptr_t WUS_OFF = 0x44200;
constexpr std::uintptr_t GTO_OFF = 0x45000;
constexpr std::uintptr_t UTO_OFF = 0x4D000;
constexpr std::uintptr_t GTN_OFF = 0x55000;
constexpr std::uintptr_t UTN_OFF = 0x5D000;
constexpr std::uintptr_t S8O_OFF = 0x65000;
constexpr std::uintptr_t SSO_OFF = 0x67000;
constexpr std::uintptr_t S8N_OFF = 0x68000;
constexpr std::uintptr_t SSN_OFF = 0x6A000;
constexpr std::uintptr_t RESULT_OFF = 0x6B000;

constexpr std::uint8_t ZB = 32;

} // namespace

int main() {
  std::uintptr_t base;
  asm volatile("mv %0, x31" : "=r"(base));
  auto *blob = reinterpret_cast<std::byte *>(base);
  auto *a = reinterpret_cast<std::int8_t *>(blob + A_OFF);
  auto *wg = reinterpret_cast<std::int8_t *>(blob + WG_OFF);
  auto *wu = reinterpret_cast<std::int8_t *>(blob + WU_OFF);
  auto *as = reinterpret_cast<std::uint32_t *>(blob + AS_OFF);
  auto *wgs = reinterpret_cast<std::uint32_t *>(blob + WGS_OFF);
  auto *wus = reinterpret_cast<std::uint32_t *>(blob + WUS_OFF);
  const std::uint32_t sb[5] = {0x3F800000u, 0x3F000000u, 0x3E800000u,
                               0x40000000u, 0x3F400000u}; // 1,.5,.25,2,.75

  for (int i = 0; i < ROWS; ++i)
    for (int k = 0; k < K; ++k)
      a[i * K + k] = static_cast<std::int8_t>((i * 7 + k * 3 + 1) % 7 - 3);
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k) {
      wg[j * K + k] = static_cast<std::int8_t>((j * 5 + k * 2 + 2) % 5 - 2);
      wu[j * K + k] = static_cast<std::int8_t>((j * 3 + k * 7 + 4) % 7 - 3);
    }
  for (int i = 0; i < ROWS; ++i)
    for (int g = 0; g < GROUPS; ++g)
      as[i * GROUPS + g] = sb[(i + g) % 5];
  for (int j = 0; j < N; ++j) {
    wgs[j] = sb[j % 5];
    wus[j] = sb[(j * 3) % 5];
  }

  const qwen3_ops::MatrixView av{reinterpret_cast<std::byte *>(a), K, 1, ROWS, K,
                                 qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView wgv{reinterpret_cast<std::byte *>(wg), K, 1, N, K,
                                  qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView wuv{reinterpret_cast<std::byte *>(wu), K, 1, N, K,
                                  qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView asv{reinterpret_cast<std::byte *>(as),
                                  GROUPS * 4, 4, ROWS, GROUPS,
                                  qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView wgsv{reinterpret_cast<std::byte *>(wgs), 4, 4, N, 1,
                                   qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView wusv{reinterpret_cast<std::byte *>(wus), 4, 4, N, 1,
                                   qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView gto{blob + GTO_OFF, N * 4, 4, ROWS, N,
                                  qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView uto{blob + UTO_OFF, N * 4, 4, ROWS, N,
                                  qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView s8o{blob + S8O_OFF, N, 1, ROWS, N,
                                  qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView s8n{blob + S8N_OFF, N, 1, ROWS, N,
                                  qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView sso{blob + SSO_OFF, 4, 4, ROWS, 1,
                                  qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView ssn{blob + SSN_OFF, 4, 4, ROWS, 1,
                                  qwen3_ops::kDtF32};

  qwen3_ops::gemm_nt_group_store(av, asv, wgv, wgsv, gto, 256);
  qwen3_ops::gemm_nt_group_store(av, asv, wuv, wusv, uto, 256);
  qwen3_ops::swiglu_quant_heads(gto, uto, s8o, sso);

  qwen3_ops::gemm_gate_up_swiglu_quant(av, asv, wgv, wgsv, wuv, wusv,
                                       s8n, ssn);

  auto fail_count = [](const std::uint32_t *x, const std::uint32_t *y, int n) {
    std::uint32_t f = 0;
    for (int i = 0; i < n; ++i) f += (x[i] != y[i]);
    return f;
  };
  auto *res = reinterpret_cast<std::uint32_t *>(blob + RESULT_OFF);
  const auto *gto32 = reinterpret_cast<const std::uint32_t *>(blob + GTO_OFF);
  const auto *uto32 = reinterpret_cast<const std::uint32_t *>(blob + UTO_OFF);
  const auto *gtn32 = reinterpret_cast<const std::uint32_t *>(blob + GTN_OFF);
  const auto *utn32 = reinterpret_cast<const std::uint32_t *>(blob + UTN_OFF);
  const auto *s8o8 = reinterpret_cast<const std::uint8_t *>(blob + S8O_OFF);
  const auto *s8n8 = reinterpret_cast<const std::uint8_t *>(blob + S8N_OFF);
  const auto *sso32 = reinterpret_cast<const std::uint32_t *>(blob + SSO_OFF);
  const auto *ssn32 = reinterpret_cast<const std::uint32_t *>(blob + SSN_OFF);
  res[0] = 0;
  res[1] = 0;
  std::uint32_t sf = 0, sfirst = 0xFFFFFFFFu;
  for (int i = 0; i < ROWS * N; ++i)
    if (s8o8[i] != s8n8[i]) { if (sf == 0) sfirst = static_cast<std::uint32_t>(i); ++sf; }
  res[2] = sf; res[3] = sfirst;
  res[4] = fail_count(sso32, ssn32, ROWS);
  res[5] = 0;
  return 0;
}
