// Batch K/V scatter: two decode rows, two requests, distinct page tables.
#include <cstdint>
#include "../kernels/kv_pages.h"

namespace {
constexpr int KV_DIM = 256;
constexpr int KV_HEADS = 2;
constexpr int PT = 32;
constexpr int PAGES = 2;
constexpr int PAGE_BYTES = PT * KV_DIM;
constexpr int SROW = KV_HEADS * 4, SPAGE = PT * SROW;
constexpr std::uintptr_t KS_OFF = 0x0000, VS_OFF = 0x0800;
constexpr std::uintptr_t V_OFF = 0x1000, KS2_OFF = 0x1800, VS2_OFF = 0x1900;
constexpr std::uintptr_t KP_OFF = 0x2000, VP_OFF = 0x6000;
constexpr std::uintptr_t KSP_OFF = 0xA000, VSP_OFF = 0xA400;
constexpr std::uintptr_t PT_OFF = 0xA800, TOK_OFF = 0xA900;
constexpr std::uintptr_t RES_OFF = 0xAA00;
} // namespace

int main() {
  std::uintptr_t base;
  asm volatile("mv %0, x31" : "=r"(base));
  auto *blob = reinterpret_cast<std::byte *>(base);
  auto *k = reinterpret_cast<std::int8_t *>(blob + KS_OFF);
  auto *v = reinterpret_cast<std::int8_t *>(blob + V_OFF);
  auto *ks = reinterpret_cast<std::uint32_t *>(blob + KS2_OFF);
  auto *vs = reinterpret_cast<std::uint32_t *>(blob + VS2_OFF);
  for (int t = 0; t < 2; ++t)
    for (int d = 0; d < KV_DIM; ++d) {
      k[t * KV_DIM + d] = static_cast<std::int8_t>((t * 7 + d * 3 + 1) % 7 - 3);
      v[t * KV_DIM + d] = static_cast<std::int8_t>((t * 5 + d * 2 + 2) % 5 - 2);
    }
  const std::uint32_t sb[4] = {0x3F800000u, 0x3F000000u, 0x3E800000u, 0x3F400000u};
  for (int t = 0; t < 2; ++t)
    for (int h = 0; h < KV_HEADS; ++h) {
      ks[t * KV_HEADS + h] = sb[(t + h) % 4];
      vs[t * KV_HEADS + h] = sb[(t * 3 + h) % 4];
    }
  const qwen3_ops::MatrixView kv{reinterpret_cast<std::byte *>(k), KV_DIM, 1, 2,
                                 KV_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView vv{reinterpret_cast<std::byte *>(v), KV_DIM, 1, 2,
                                 KV_DIM, qwen3_ops::kDtI8};
  const qwen3_ops::MatrixView ksv{reinterpret_cast<std::byte *>(ks), KV_HEADS * 4,
                                  4, 2, KV_HEADS, qwen3_ops::kDtF32};
  const qwen3_ops::MatrixView vsv{reinterpret_cast<std::byte *>(vs), KV_HEADS * 4,
                                  4, 2, KV_HEADS, qwen3_ops::kDtF32};
  auto *pt = reinterpret_cast<unsigned short *>(blob + PT_OFF);
  pt[0] = 1; pt[1] = 0;          // request 0: page 1
  pt[2] = 0;                     // request 1: page 0
  auto *tok = reinterpret_cast<std::uint32_t *>(blob + TOK_OFF);
  tok[0] = 0; tok[1] = 0;
  const qwen3_ops::PagedKV p0{blob + KP_OFF, blob + VP_OFF, blob + KSP_OFF,
                              blob + VSP_OFF, pt, PT, PAGE_BYTES, KV_DIM,
                              SPAGE, SROW};
  const qwen3_ops::PagedKV p1{blob + KP_OFF, blob + VP_OFF, blob + KSP_OFF,
                              blob + VSP_OFF, pt + 2, PT, PAGE_BYTES, KV_DIM,
                              SPAGE, SROW};
  const qwen3_ops::PagedKV *tables[2] = {&p0, &p1};
  qwen3_ops::paged_kv_scatter_batch(kv, vv, ksv, vsv, tables, tok, 2, KV_DIM,
                                    KV_HEADS);
  std::uint32_t fails = 0;
  for (int t = 0; t < 2; ++t) {
    const int page = (t == 0) ? 1 : 0;
    const auto *pk = reinterpret_cast<const std::uint8_t *>(
        blob + KP_OFF + page * PAGE_BYTES);
    const auto *pv = reinterpret_cast<const std::uint8_t *>(
        blob + VP_OFF + page * PAGE_BYTES);
    const auto *r = reinterpret_cast<const std::uint8_t *>(blob + KS_OFF + t * KV_DIM);
    const auto *rv = reinterpret_cast<const std::uint8_t *>(blob + V_OFF + t * KV_DIM);
    for (int d = 0; d < KV_DIM; ++d) {
      fails += (pk[d] != r[d]);
      fails += (pv[d] != rv[d]);
    }
    const auto *ps = reinterpret_cast<const std::uint32_t *>(
        blob + KSP_OFF + page * SPAGE);
    const auto *pvs = reinterpret_cast<const std::uint32_t *>(
        blob + VSP_OFF + page * SPAGE);
    const auto *ss = reinterpret_cast<const std::uint32_t *>(
        blob + KS2_OFF + t * KV_HEADS * 4);
    const auto *svs = reinterpret_cast<const std::uint32_t *>(
        blob + VS2_OFF + t * KV_HEADS * 4);
    for (int h = 0; h < KV_HEADS; ++h) {
      fails += (ps[h] != ss[h]);
      fails += (pvs[h] != svs[h]);
    }
  }
  auto *res = reinterpret_cast<std::uint32_t *>(blob + RES_OFF);
  res[0] = fails;
  return 0;
}
