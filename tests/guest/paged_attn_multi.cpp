// Two requests sharing one paged KV pool, each with its own page table.
// Each request is compared against its own contiguous attention output.
#include <cstdint>
#include "../kernels/attention_gqa.h"
#include "../kernels/kv_pages.h"

namespace {
constexpr int HEADS = 4, KV_HEADS = 2, HD = 128;
constexpr int Q_DIM = HEADS * HD, KV_DIM = KV_HEADS * HD;
constexpr int Q0 = 4, K0 = 80;
constexpr int Q1 = 8, K1 = 48;
constexpr int PT = 32;
constexpr int PAGES = 5;
constexpr int PAGE_BYTES = PT * KV_DIM;
constexpr int SROW = KV_HEADS * 4, SPAGE = PT * SROW;

constexpr std::uintptr_t KP = 0x00000, VP = 0x0A000;
constexpr std::uintptr_t KS = 0x14000, VS = 0x15000;
constexpr std::uintptr_t K0D = 0x16000, V0D = 0x1B000;
constexpr std::uintptr_t K1D = 0x20000, V1D = 0x23000;
constexpr std::uintptr_t Q0D = 0x26000, Q1D = 0x27000;
constexpr std::uintptr_t QS0 = 0x28000, QS1 = 0x28100;
constexpr std::uintptr_t KS0 = 0x28200, VS0 = 0x28400;
constexpr std::uintptr_t KS1 = 0x28600, VS1 = 0x28800;
constexpr std::uintptr_t O8C0 = 0x29000, OSC0 = 0x2A000;
constexpr std::uintptr_t O8P0 = 0x2A800, OSP0 = 0x2B000;
constexpr std::uintptr_t O8C1 = 0x2B800, OSC1 = 0x2C800;
constexpr std::uintptr_t O8P1 = 0x2D000, OSP1 = 0x2E000;
constexpr std::uintptr_t PTAB = 0x2F000;
constexpr std::uintptr_t RES = 0x2F800;
} // namespace

int main() {
  std::uintptr_t base;
  asm volatile("mv %0, x31" : "=r"(base));
  auto *blob = reinterpret_cast<std::byte *>(base);
  auto *k0 = reinterpret_cast<std::int8_t *>(blob + K0D);
  auto *v0 = reinterpret_cast<std::int8_t *>(blob + V0D);
  auto *k1 = reinterpret_cast<std::int8_t *>(blob + K1D);
  auto *v1 = reinterpret_cast<std::int8_t *>(blob + V1D);
  auto *q0 = reinterpret_cast<std::int8_t *>(blob + Q0D);
  auto *q1 = reinterpret_cast<std::int8_t *>(blob + Q1D);
  auto *qs0 = reinterpret_cast<std::uint32_t *>(blob + QS0);
  auto *qs1 = reinterpret_cast<std::uint32_t *>(blob + QS1);
  auto *ks0 = reinterpret_cast<std::uint32_t *>(blob + KS0);
  auto *vs0 = reinterpret_cast<std::uint32_t *>(blob + VS0);
  auto *ks1 = reinterpret_cast<std::uint32_t *>(blob + KS1);
  auto *vs1 = reinterpret_cast<std::uint32_t *>(blob + VS1);

  for (int i = 0; i < Q0; ++i)
    for (int d = 0; d < Q_DIM; ++d)
      q0[i * Q_DIM + d] = static_cast<std::int8_t>((i * 7 + d * 3 + 1) % 7 - 3);
  for (int i = 0; i < Q1; ++i)
    for (int d = 0; d < Q_DIM; ++d)
      q1[i * Q_DIM + d] = static_cast<std::int8_t>((i * 5 + d * 2 + 2) % 5 - 2);
  for (int t = 0; t < K0; ++t)
    for (int d = 0; d < KV_DIM; ++d) {
      k0[t * KV_DIM + d] = static_cast<std::int8_t>((t * 5 + d * 2 + 2) % 5 - 2);
      v0[t * KV_DIM + d] = static_cast<std::int8_t>((t * 3 + d * 7 + 4) % 7 - 3);
    }
  for (int t = 0; t < K1; ++t)
    for (int d = 0; d < KV_DIM; ++d) {
      k1[t * KV_DIM + d] = static_cast<std::int8_t>((t * 3 + d * 5 + 1) % 7 - 3);
      v1[t * KV_DIM + d] = static_cast<std::int8_t>((t * 7 + d * 3 + 2) % 5 - 2);
    }
  const std::uint32_t sb[4] = {0x3F800000u, 0x3F000000u, 0x3E800000u, 0x3F400000u};
  for (int i = 0; i < Q0; ++i) for (int h = 0; h < HEADS; ++h) qs0[i * HEADS + h] = sb[(i + h) % 4];
  for (int i = 0; i < Q1; ++i) for (int h = 0; h < HEADS; ++h) qs1[i * HEADS + h] = sb[(i * 2 + h) % 4];
  for (int t = 0; t < K0; ++t) for (int h = 0; h < KV_HEADS; ++h) { ks0[t * KV_HEADS + h] = sb[(t + h) % 4]; vs0[t * KV_HEADS + h] = sb[(t * 3 + h) % 4]; }
  for (int t = 0; t < K1; ++t) for (int h = 0; h < KV_HEADS; ++h) { ks1[t * KV_HEADS + h] = sb[(t * 2 + h) % 4]; vs1[t * KV_HEADS + h] = sb[(t * 5 + h) % 4]; }

  const qwen3_ops::MatrixView q0v{reinterpret_cast<std::byte *>(q0), Q_DIM, 1, Q0, Q_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView q1v{reinterpret_cast<std::byte *>(q1), Q_DIM, 1, Q1, Q_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView k0v{reinterpret_cast<std::byte *>(k0), KV_DIM, 1, K0, KV_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView v0v{reinterpret_cast<std::byte *>(v0), KV_DIM, 1, K0, KV_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView k1v{reinterpret_cast<std::byte *>(k1), KV_DIM, 1, K1, KV_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView v1v{reinterpret_cast<std::byte *>(v1), KV_DIM, 1, K1, KV_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView qs0v{reinterpret_cast<std::byte *>(qs0), HEADS * 4, 4, Q0, HEADS, qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView qs1v{reinterpret_cast<std::byte *>(qs1), HEADS * 4, 4, Q1, HEADS, qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView ks0v{reinterpret_cast<std::byte *>(ks0), KV_HEADS * 4, 4, K0, KV_HEADS, qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView vs0v{reinterpret_cast<std::byte *>(vs0), KV_HEADS * 4, 4, K0, KV_HEADS, qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView ks1v{reinterpret_cast<std::byte *>(ks1), KV_HEADS * 4, 4, K1, KV_HEADS, qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView vs1v{reinterpret_cast<std::byte *>(vs1), KV_HEADS * 4, 4, K1, KV_HEADS, qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView o8c0{blob + O8C0, Q_DIM, 1, Q0, Q_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView osc0{blob + OSC0, HEADS * 4, 4, Q0, HEADS, qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView o8p0{blob + O8P0, Q_DIM, 1, Q0, Q_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView osp0{blob + OSP0, HEADS * 4, 4, Q0, HEADS, qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView o8c1{blob + O8C1, Q_DIM, 1, Q1, Q_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView osc1{blob + OSC1, HEADS * 4, 4, Q1, HEADS, qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView o8p1{blob + O8P1, Q_DIM, 1, Q1, Q_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView osp1{blob + OSP1, HEADS * 4, 4, Q1, HEADS, qwen3_ops::kDtF32};

  unsigned short *pt = reinterpret_cast<unsigned short *>(blob + PTAB);
  pt[0] = 2; pt[1] = 0; pt[2] = 1;   // request 0
  pt[3] = 3; pt[4] = 4;              // request 1
  const qwen3_ops::PagedKV paged{blob + KP, blob + VP, blob + KS, blob + VS,
                                 pt, PT, PAGE_BYTES, KV_DIM, SPAGE, SROW};
  qwen3_ops::paged_kv_scatter(k0v, v0v, ks0v, vs0v, paged, 0, K0, KV_DIM, KV_HEADS);
  // request 1's page table starts at pt+3; use a second view over the same pool.
  const qwen3_ops::PagedKV paged1{blob + KP, blob + VP, blob + KS, blob + VS,
                                  pt + 3, PT, PAGE_BYTES, KV_DIM, SPAGE, SROW};
  qwen3_ops::paged_kv_scatter(k1v, v1v, ks1v, vs1v, paged1, 0, K1, KV_DIM, KV_HEADS);

  const float qk = 0.08838834764831845f;
  qwen3_ops::attention_heads_gqa_tokscale(q0v, k0v, v0v, qs0v, ks0v, vs0v,
                                          o8c0, osc0, HEADS, KV_HEADS, HD, true, qk);
  qwen3_ops::attention_heads_gqa_tokscale(q1v, k1v, v1v, qs1v, ks1v, vs1v,
                                          o8c1, osc1, HEADS, KV_HEADS, HD, true, qk);
  qwen3_ops::attention_heads_gqa_tokscale_paged(q0v, qs0v, paged, K0, o8p0, osp0,
                                                HEADS, KV_HEADS, HD, true, qk);
  qwen3_ops::attention_heads_gqa_tokscale_paged(q1v, qs1v, paged1, K1, o8p1, osp1,
                                                HEADS, KV_HEADS, HD, true, qk);

  auto cmp8 = [](const std::uint8_t *a, const std::uint8_t *b, int n) {
    std::uint32_t c = 0;
    for (int i = 0; i < n; ++i) c += (a[i] != b[i]);
    return c;
  };
  auto cmp32 = [](const std::uint32_t *a, const std::uint32_t *b, int n) {
    std::uint32_t c = 0;
    for (int i = 0; i < n; ++i) c += (a[i] != b[i]);
    return c;
  };
  auto *res = reinterpret_cast<std::uint32_t *>(blob + RES);
  const auto *c0 = reinterpret_cast<const std::uint8_t *>(blob + O8C0);
  const auto *p0 = reinterpret_cast<const std::uint8_t *>(blob + O8P0);
  const auto *c1 = reinterpret_cast<const std::uint8_t *>(blob + O8C1);
  const auto *p1 = reinterpret_cast<const std::uint8_t *>(blob + O8P1);
  const auto *s0c = reinterpret_cast<const std::uint32_t *>(blob + OSC0);
  const auto *s0p = reinterpret_cast<const std::uint32_t *>(blob + OSP0);
  const auto *s1c = reinterpret_cast<const std::uint32_t *>(blob + OSC1);
  const auto *s1p = reinterpret_cast<const std::uint32_t *>(blob + OSP1);
  res[0] = cmp8(c0, p0, Q0 * Q_DIM);
  res[1] = cmp32(s0c, s0p, Q0 * HEADS);
  res[2] = cmp8(c1, p1, Q1 * Q_DIM);
  res[3] = cmp32(s1c, s1p, Q1 * HEADS);
  return 0;
}
