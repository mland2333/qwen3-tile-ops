// Paged KV cache: 32-token pages, per-request page table.
// The attention kernel only needs page base pointers; the host/engine owns
// allocation and page-table contents.
#ifndef QWEN3_OPS_KV_PAGES_H
#define QWEN3_OPS_KV_PAGES_H

#include "tile.h"
#include "io.h"
#include "matrix.h"

namespace qwen3_ops {

struct PagedKV {
  std::byte *k;              // [page][page_tokens][row_bytes] i8
  std::byte *v;              // same
  std::byte *ks;             // [page][page_tokens][kv_heads] f32
  std::byte *vs;             // same
  const unsigned short *table; // logical page -> physical page
  std::uint32_t page_tokens;  // 32
  std::uint32_t page_bytes;   // page_tokens * row_bytes
  std::uint32_t row_bytes;    // kv_heads * head_dim
  std::uint32_t scale_page_bytes;
  std::uint32_t scale_row_bytes; // kv_heads * 4
};

YAI_ALWAYS_INLINE std::byte *paged_k_base(const PagedKV &kv,
                                          std::uint32_t token,
                                          std::uint32_t off = 0) {
  const std::uint32_t page = kv.table[token / kv.page_tokens];
  const std::uint32_t slot = token % kv.page_tokens;
  return kv.k + page * kv.page_bytes + slot * kv.row_bytes + off;
}

YAI_ALWAYS_INLINE std::byte *paged_v_base(const PagedKV &kv,
                                          std::uint32_t token,
                                          std::uint32_t off = 0) {
  const std::uint32_t page = kv.table[token / kv.page_tokens];
  const std::uint32_t slot = token % kv.page_tokens;
  return kv.v + page * kv.page_bytes + slot * kv.row_bytes + off;
}

YAI_ALWAYS_INLINE std::byte *paged_ks_base(const PagedKV &kv,
                                           std::uint32_t token,
                                           std::uint32_t off = 0) {
  const std::uint32_t page = kv.table[token / kv.page_tokens];
  const std::uint32_t slot = token % kv.page_tokens;
  return kv.ks + page * kv.scale_page_bytes + slot * kv.scale_row_bytes + off;
}

YAI_ALWAYS_INLINE std::byte *paged_vs_base(const PagedKV &kv,
                                           std::uint32_t token,
                                           std::uint32_t off = 0) {
  const std::uint32_t page = kv.table[token / kv.page_tokens];
  const std::uint32_t slot = token % kv.page_tokens;
  return kv.vs + page * kv.scale_page_bytes + slot * kv.scale_row_bytes + off;
}

// Scatter rows of contiguous K/V (+ per-token scales) into the pages.
// k/v: [rows][kv_dim] i8; ks/vs: [rows][kv_heads] f32.
YAI_OP_INLINE void paged_kv_scatter(const MatrixView &k,
                                    const MatrixView &v,
                                    const MatrixView &ks,
                                    const MatrixView &vs,
                                    const PagedKV &kv,
                                    std::uint32_t start_token,
                                    std::uint32_t rows,
                                    std::uint32_t kv_dim,
                                    std::uint32_t kv_heads) {
  for (std::uint32_t r = 0; r < rows; ++r) {
    const std::uint32_t tok = start_token + r;
    std::byte *dk = paged_k_base(kv, tok);
    std::byte *dv = paged_v_base(kv, tok);
    std::byte *dks = paged_ks_base(kv, tok);
    std::byte *dvs = paged_vs_base(kv, tok);
    const std::byte *sk = k.data + r * static_cast<std::uint32_t>(k.row_bytes);
    const std::byte *sv = v.data + r * static_cast<std::uint32_t>(v.row_bytes);
    for (std::uint32_t c = 0; c < kv_dim; ++c) {
      dk[c] = sk[c];
      dv[c] = sv[c];
    }
    const std::byte *sks = ks.data + r * static_cast<std::uint32_t>(ks.row_bytes);
    const std::byte *svs = vs.data + r * static_cast<std::uint32_t>(vs.row_bytes);
    for (std::uint32_t h = 0; h < kv_heads; ++h) {
      dks[h * 4 + 0] = sks[h * 4 + 0];
      dks[h * 4 + 1] = sks[h * 4 + 1];
      dks[h * 4 + 2] = sks[h * 4 + 2];
      dks[h * 4 + 3] = sks[h * 4 + 3];
      dvs[h * 4 + 0] = svs[h * 4 + 0];
      dvs[h * 4 + 1] = svs[h * 4 + 1];
      dvs[h * 4 + 2] = svs[h * 4 + 2];
      dvs[h * 4 + 3] = svs[h * 4 + 3];
    }
  }
}

} // namespace qwen3_ops

#endif
