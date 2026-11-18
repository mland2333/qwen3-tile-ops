// Serve guest, route B: one guest call performs a batch forward over all
// scheduled rows, then per-request paged attention and lm_head.
//
// The host ABI is identical to route A (ServeCmd with up to 8 items).  Item
// row ranges are packed in order, so the guest derives the row map by summing
// rows.  K/V is stored to contiguous staging and scattered into each request's
// pages; Q/K rope uses a gathered per-row cos/sin table.
#include <cstddef>
#include <cstdint>

#include "../kernels/attention_gqa.h"
#include "../kernels/config.h"
#include "../kernels/embedding.h"
#include "../kernels/group_gemm.h"
#include "../kernels/kv_pages.h"
#include "../kernels/norm_rope_quant.h"
#include "../kernels/qwen3.h"
#include "../kernels/swiglu_quant.h"
#include "serve_abi.h"
#include "serve_views.h"

namespace {

using namespace qwen3_ops;

// Build the per-layer PagedKV descriptions from each item's blob page table.
YAI_ALWAYS_INLINE void make_paged(const LayerViews &lv, std::byte *blob,
                                  const ServeItem &it, PagedKV &paged) {
  const std::uint32_t kv_row = static_cast<std::uint32_t>(lv.kcache.row_bytes);
  const std::uint32_t ks_row = static_cast<std::uint32_t>(lv.kcscale.row_bytes);
  constexpr std::uint32_t kPageTokens = 32;
  paged = PagedKV{
      lv.kcache.data, lv.vcache.data, lv.kcscale.data, lv.vcscale.data,
      reinterpret_cast<const unsigned short *>(blob + it.ptab), kPageTokens,
      kPageTokens * kv_row, kv_row, kPageTokens * ks_row, ks_row};
}

YAI_ALWAYS_INLINE void batch_forward(ModelViews &w, std::byte *blob,
                                     const ServeCmd &cmd, std::uint32_t n,
                                     std::uint32_t total) {
  constexpr std::uint32_t H = mb::kHidden;
  constexpr std::uint32_t HD = mb::kHeadDim;
  constexpr std::uint32_t QW = mb::kHeads * HD;
  constexpr std::uint32_t KW = mb::kKvHeads * HD;
  constexpr std::uint32_t HALF = mb::kHalf;
  constexpr std::uint32_t N = mb::kSeq;

  std::uint32_t row_start[8];
  std::uint32_t ids_all[N];
  std::uint32_t row = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    const ServeItem &it = cmd.items[i];
    row_start[i] = row;
    const auto *ids = reinterpret_cast<const std::uint32_t *>(blob + it.ids);
    for (std::uint32_t r = 0; r < it.rows; ++r) ids_all[row + r] = ids[r];
    row += it.rows;
  }

  // Gather the rope tables for the absolute position of every activation row.
  const MatrixView cos_all = f32_view(blob + serve_abi::kCos, total, HALF);
  const MatrixView sin_all = f32_view(blob + serve_abi::kSin, total, HALF);
  for (std::uint32_t i = 0; i < n; ++i) {
    const ServeItem &it = cmd.items[i];
    for (std::uint32_t r = 0; r < it.rows; ++r) {
      const std::uint32_t pos = it.pos + r;
      auto *cd = reinterpret_cast<std::uint32_t *>(
          cos_all.data + (row_start[i] + r) * cos_all.row_bytes);
      auto *sd = reinterpret_cast<std::uint32_t *>(
          sin_all.data + (row_start[i] + r) * sin_all.row_bytes);
      const auto *cs = reinterpret_cast<const std::uint32_t *>(
          w.cos_t.data + pos * static_cast<std::uint32_t>(w.cos_t.row_bytes));
      const auto *ss = reinterpret_cast<const std::uint32_t *>(
          w.sin_t.data + pos * static_cast<std::uint32_t>(w.sin_t.row_bytes));
      for (std::uint32_t j = 0; j < HALF; ++j) {
        cd[j] = cs[j];
        sd[j] = ss[j];
      }
    }
  }

  const MatrixView x_all = rows_of(w.x, 0, total);
  const MatrixView h_all = rows_of(w.h, 0, total);
  const MatrixView h8_all = rows_of(w.h8, 0, total);
  const MatrixView hs_all = rows_of(w.hs, 0, total);
  const MatrixView q_all = rows_of(w.q, 0, total);
  const MatrixView k_all = rows_of(w.k, 0, total);
  const MatrixView v_all = rows_of(w.v, 0, total);
  const MatrixView q8_all = rows_of(w.q8, 0, total);
  const MatrixView qs_all = rows_of(w.qs, 0, total);
  const MatrixView o_all = rows_of(w.o, 0, total);
  const MatrixView o8_all = rows_of(w.o8, 0, total);
  const MatrixView os_all = rows_of(w.os, 0, total);
  const MatrixView k8_all = rows_of(w.k8, 0, total);
  const MatrixView v8_all = rows_of(w.v8, 0, total);
  const MatrixView ks_all = rows_of(w.ks, 0, total);
  const MatrixView vs_all = rows_of(w.vs, 0, total);
  const MatrixView s8_all = rows_of(w.s8, 0, total);
  const MatrixView ss_all = rows_of(w.ss, 0, total);

  const MatrixView ids_view =
      i32_view(reinterpret_cast<std::byte *>(ids_all), total, 1);
  const MatrixView n1_0 = f32_view(w.blob + mb::kN1, H, 1);
  embedding_rmsnorm_rows(w.emb, w.emb_s, ids_view, x_all, n1_0, h_all,
                         1.0f / H);

  for (int L = 0; L < mb::kLayers; ++L) {
    LayerViews lv;
    make_layer_views(lv, w.blob, L);
    if (L != 0) rmsnorm_rows(x_all, lv.n1, h_all, 1.0f / H);
    quant_rows_heads(h_all, h8_all, hs_all, 256);

    // V: GEMM + per-head quant into contiguous staging.
    gemm_nt_group_store(h8_all, hs_all, lv.wv, lv.wvs, v_all, 256, nullptr,
                        &v8_all, &vs_all);
    // K: GEMM + norm/rope/quant into contiguous staging.
    gemm_nt_group_store(h8_all, hs_all, lv.wk, lv.wks, k_all, 256, nullptr,
                        &k8_all, &ks_all, &lv.nk, &cos_all, &sin_all,
                        1.0f / HD);

    // Batch scatter: row r -> item table/token.
    PagedKV paged[8];
    const PagedKV *tables[N];
    std::uint32_t tokens[N];
    for (std::uint32_t i = 0; i < n; ++i) {
      const ServeItem &it = cmd.items[i];
      make_paged(lv, blob, it, paged[i]);
      for (std::uint32_t r = 0; r < it.rows; ++r) {
        tables[row_start[i] + r] = &paged[i];
        tokens[row_start[i] + r] = it.pos + r;
      }
    }
    paged_kv_scatter_batch(k8_all, v8_all, ks_all, vs_all, tables, tokens,
                           total, KW, mb::kKvHeads);

    // Q: one batch GEMM, per-row rope.
    gemm_nt_group_store(h8_all, hs_all, lv.wq, lv.wqs, q_all, 256, nullptr,
                        &q8_all, &qs_all, &lv.nq, &cos_all, &sin_all,
                        1.0f / HD);

    // Attention is per request; rows of one request are contiguous.
    for (std::uint32_t i = 0; i < n; ++i) {
      const ServeItem &it = cmd.items[i];
      const std::uint32_t rs = row_start[i];
      attention_heads_gqa_tokscale_paged(
          sub_view(w.q8, rs, 0, it.rows, QW),
          sub_view(w.qs, rs, 0, it.rows, mb::kHeads), paged[i],
          it.pos + it.rows, sub_view(w.o8, rs, 0, it.rows, QW),
          sub_view(w.os, rs, 0, it.rows, mb::kHeads), mb::kHeads,
          mb::kKvHeads, HD, /*causal=*/true, kInvSqrt128);
    }

    // O + residual, then the batch MLP.
    gemm_nt_group_store(o8_all, os_all, lv.wo, lv.wos, o_all, 128, &x_all);
    rmsnorm_rows(x_all, lv.n2, h_all, 1.0f / H);
    quant_rows_heads(h_all, h8_all, hs_all, 256);
    gemm_gate_up_swiglu_quant(h8_all, hs_all, lv.wg, lv.wgs, lv.wu, lv.wus,
                              s8_all, ss_all);
    gemm_nt_group_store(s8_all, ss_all, lv.wd, lv.wds, o_all, 256, &x_all);
  }

  // Per-request final norm + lm_head into that request's logits slot.
  for (std::uint32_t i = 0; i < n; ++i) {
    const ServeItem &it = cmd.items[i];
    if (!it.needs) continue;
    const std::uint32_t last = row_start[i] + it.rows - 1;
    const MatrixView xl = rows_of(w.x, last, 1);
    const MatrixView hl = rows_of(w.o, 0, 1);
    const MatrixView h8l = sub_view(w.q8, 0, 0, 1, H);
    const MatrixView hsl{w.qs.data, w.qs.row_bytes, 4, 1, H / 256, kDtF32};
    rmsnorm_rows(xl, w.nf, hl, 1.0f / H);
    quant_rows_heads(hl, h8l, hsl, 256);
    const MatrixView logits_i = f32_view(blob + it.logits, 1, mb::kVocab);
    gemm_nt_group_store(h8l, hsl, w.lm, w.lm_s, logits_i, 256);
  }
}

} // namespace

int main() {
  std::uintptr_t base;
  asm volatile("mv %0, x31" : "=r"(base));
  auto *blob = reinterpret_cast<std::byte *>(base);
  ModelViews w;
  make_model_views(w, blob);
  const auto *cmd = reinterpret_cast<const ServeCmd *>(blob + serve_abi::kCmd);
  *reinterpret_cast<std::uint32_t *>(blob + serve_abi::kAbi) =
      static_cast<std::uint32_t>(serve_abi::kEnd);

  const std::uint32_t n = cmd->n < 8 ? cmd->n : 8;
  std::uint32_t total = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    const ServeItem &it = cmd->items[i];
    if (it.rows == 0 || total + it.rows > mb::kSeq) return 1;
    total += it.rows;
  }
  if (n == 0) return 0;
  batch_forward(w, blob, *cmd, n, total);
  return 0;
}
