// Paged attention vs contiguous attention, bit-exact, with a shuffled page table.
#include <cstdint>

#include "../kernels/attention_gqa.h"
#include "../kernels/kv_pages.h"

namespace {
constexpr int Q_ROWS = 4;
constexpr int KV_LEN = 80;
constexpr int HEADS = 4;
constexpr int KV_HEADS = 2;
constexpr int HD = 128;
constexpr int Q_DIM = HEADS * HD;
constexpr int KV_DIM = KV_HEADS * HD;
constexpr int PAGE_TOKENS = 32;
constexpr int PAGES = 3;
constexpr int PAGE_BYTES = PAGE_TOKENS * KV_DIM;
constexpr int SCALE_ROW = KV_HEADS * 4;
constexpr int SCALE_PAGE = PAGE_TOKENS * SCALE_ROW;

constexpr std::uintptr_t Q_OFF = 0x0000;
constexpr std::uintptr_t K_OFF = 0x1000;
constexpr std::uintptr_t V_OFF = 0x7000;
constexpr std::uintptr_t QS_OFF = 0xC000;
constexpr std::uintptr_t KS_OFF = 0xC100;
constexpr std::uintptr_t VS_OFF = 0xC400;
constexpr std::uintptr_t O8C_OFF = 0xD000;
constexpr std::uintptr_t OSC_OFF = 0xD800;
constexpr std::uintptr_t O8P_OFF = 0xE000;
constexpr std::uintptr_t OSP_OFF = 0xE800;
constexpr std::uintptr_t KP_OFF = 0x10000;
constexpr std::uintptr_t VP_OFF = 0x16000;
constexpr std::uintptr_t KSP_OFF = 0x1C000;
constexpr std::uintptr_t VSP_OFF = 0x1C400;
constexpr std::uintptr_t PT_OFF = 0x1D000;
constexpr std::uintptr_t RESULT_OFF = 0x1E000;
} // namespace

int main() {
  std::uintptr_t base;
  asm volatile("mv %0, x31" : "=r"(base));
  auto *blob = reinterpret_cast<std::byte *>(base);
  auto *q = reinterpret_cast<std::int8_t *>(blob + Q_OFF);
  auto *k = reinterpret_cast<std::int8_t *>(blob + K_OFF);
  auto *v = reinterpret_cast<std::int8_t *>(blob + V_OFF);
  auto *qs = reinterpret_cast<std::uint32_t *>(blob + QS_OFF);
  auto *ks = reinterpret_cast<std::uint32_t *>(blob + KS_OFF);
  auto *vs = reinterpret_cast<std::uint32_t *>(blob + VS_OFF);

  for (int i = 0; i < Q_ROWS; ++i)
    for (int d = 0; d < Q_DIM; ++d)
      q[i * Q_DIM + d] = static_cast<std::int8_t>((i * 7 + d * 3 + 1) % 7 - 3);
  for (int t = 0; t < KV_LEN; ++t)
    for (int d = 0; d < KV_DIM; ++d) {
      k[t * KV_DIM + d] = static_cast<std::int8_t>((t * 5 + d * 2 + 2) % 5 - 2);
      v[t * KV_DIM + d] = static_cast<std::int8_t>((t * 3 + d * 7 + 4) % 7 - 3);
    }
  const std::uint32_t sb[4] = {0x3F800000u, 0x3F000000u, 0x3E800000u,
                               0x3F400000u}; // 1, .5, .25, .75
  for (int i = 0; i < Q_ROWS; ++i)
    for (int h = 0; h < HEADS; ++h)
      qs[i * HEADS + h] = sb[(i + h) % 4];
  for (int t = 0; t < KV_LEN; ++t)
    for (int h = 0; h < KV_HEADS; ++h) {
      ks[t * KV_HEADS + h] = sb[(t + h) % 4];
      vs[t * KV_HEADS + h] = sb[(t * 3 + h) % 4];
    }

  const qwen3_ops::MatrixView qv{reinterpret_cast<std::byte *>(q), Q_DIM, 1, Q_ROWS, Q_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView kv{reinterpret_cast<std::byte *>(k), KV_DIM, 1, KV_LEN, KV_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView vv{reinterpret_cast<std::byte *>(v), KV_DIM, 1, KV_LEN, KV_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView qsv{reinterpret_cast<std::byte *>(qs),
                                  HEADS * 4, 4, Q_ROWS, HEADS,
                                  qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView ksv{reinterpret_cast<std::byte *>(ks),
                                  KV_HEADS * 4, 4, KV_LEN, KV_HEADS,
                                  qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView vsv{reinterpret_cast<std::byte *>(vs),
                                  KV_HEADS * 4, 4, KV_LEN, KV_HEADS,
                                  qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView o8c{blob + O8C_OFF, Q_DIM, 1, Q_ROWS, Q_DIM,
                                  qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView osc{blob + OSC_OFF, HEADS * 4, 4, Q_ROWS, HEADS,
                                  qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView o8p{blob + O8P_OFF, Q_DIM, 1, Q_ROWS, Q_DIM,
                                  qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView osp{blob + OSP_OFF, HEADS * 4, 4, Q_ROWS, HEADS,
                                  qwen3_ops::kDtF32};

  auto *pt = reinterpret_cast<unsigned short *>(blob + PT_OFF);
  pt[0] = 2; pt[1] = 0; pt[2] = 1;   // shuffled logical -> physical
  qwen3_ops::PagedKV paged{
      blob + KP_OFF, blob + VP_OFF, blob + KSP_OFF, blob + VSP_OFF,
      pt, PAGE_TOKENS, PAGE_BYTES, KV_DIM, SCALE_PAGE, SCALE_ROW};

  qwen3_ops::paged_kv_scatter(kv, vv, ksv, vsv, paged, 0, KV_LEN, KV_DIM,
                              KV_HEADS);

  const float qk_scale = 0.08838834764831845f; // 1/sqrt(128)
  qwen3_ops::attention_heads_gqa_tokscale(
      qv, kv, vv, qsv, ksv, vsv, o8c, osc, HEADS, KV_HEADS, HD,
      /*causal=*/true, qk_scale);
  qwen3_ops::attention_heads_gqa_tokscale_paged(
      qv, qsv, paged, KV_LEN, o8p, osp, HEADS, KV_HEADS, HD,
      /*causal=*/true, qk_scale);

  const auto *c8 = reinterpret_cast<const std::uint8_t *>(blob + O8C_OFF);
  const auto *p8 = reinterpret_cast<const std::uint8_t *>(blob + O8P_OFF);
  const auto *cs = reinterpret_cast<const std::uint32_t *>(blob + OSC_OFF);
  const auto *ps = reinterpret_cast<const std::uint32_t *>(blob + OSP_OFF);
  std::uint32_t f8 = 0, first = 0xFFFFFFFFu;
  for (int i = 0; i < Q_ROWS * Q_DIM; ++i)
    if (c8[i] != p8[i]) { if (f8 == 0) first = static_cast<std::uint32_t>(i); ++f8; }
  std::uint32_t fs = 0;
  for (int i = 0; i < Q_ROWS * HEADS; ++i) fs += (cs[i] != ps[i]);
  auto *res = reinterpret_cast<std::uint32_t *>(blob + RESULT_OFF);
  res[0] = f8;
  res[1] = first;
  res[2] = fs;
  res[3] = 0;
  return 0;
}
