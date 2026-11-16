// ============================================================================
// Q/K 的 norm + rope + per-head quant。
// 两个入口：
//   norm_rope_quant_accs：直接吃 GEMM 的 4 个 f32 acc（acc-level，零 f32 落地）
//   norm_rope_quant_heads：从 f32 q/k 读 4 个 acc（给需要 f32 的旧路径）
// ============================================================================
#ifndef QWEN3_OPS_NORM_ROPE_QUANT_H
#define QWEN3_OPS_NORM_ROPE_QUANT_H

#include "tile.h"
#include "config.h"
#include "io.h"
#include "matrix.h"

namespace qwen3_ops {

YAI_ALWAYS_INLINE void norm_rope_quant_accs(
    Acc &a0, Acc &a1, Acc &a2, Acc &a3, const MatrixView &w,
    const MatrixView &cos_t, const MatrixView &sin_t, const MatrixView &q8,
    const MatrixView &scale, std::uint32_t col_base, std::uint32_t scale_off,
    std::uint8_t nr, float inv_hd, bool do_rope) {
  constexpr std::uint8_t ZB = 32;
  Vec32 wv{};
  wv.set_dtype(Vec32DType::F32);
  wv.set_stride(w.row_bytes);

  Vec32 ss{};
  ss.set_dtype(Vec32DType::F32);
  set_len(ss, nr);
  ss.fill(0.0f);
  a0.row_sum_sq(ss); a1.row_sum_sq(ss);
  a2.row_sum_sq(ss); a3.row_sum_sq(ss);
  ss.mul_(inv_hd);
  ss.add_(kEps);
  ss.rsqrt_();

  load_vec_n(wv, w, 0, 32);  a0.mul_(ss, true); a0.mul_(wv, false);
  load_vec_n(wv, w, 32, 32); a1.mul_(ss, true); a1.mul_(wv, false);
  load_vec_n(wv, w, 64, 32); a2.mul_(ss, true); a2.mul_(wv, false);
  load_vec_n(wv, w, 96, 32); a3.mul_(ss, true); a3.mul_(wv, false);

  if (do_rope) {
    Acc co{}, si{};
    config_acc(co, cos_t); config_acc(si, sin_t);
    set_zone(co, nr, ZB); load_acc_at(co, cos_t.data + 0 * cos_t.col_bytes);
    set_zone(si, nr, ZB); load_acc_at(si, sin_t.data + 0 * sin_t.col_bytes);
    Acc l = a0.mul(co);
    Acc tt = a2.mul(si);
    l = l.sub(tt);
    Acc r = a0.mul(si);
    Acc t2 = a2.mul(co);
    r = r.add(t2);
    a0 = static_cast<Acc &&>(l);
    a2 = static_cast<Acc &&>(r);
    set_zone(co, nr, ZB); load_acc_at(co, cos_t.data + 32 * cos_t.col_bytes);
    set_zone(si, nr, ZB); load_acc_at(si, sin_t.data + 32 * sin_t.col_bytes);
    Acc l2 = a1.mul(co);
    Acc t3 = a3.mul(si);
    l2 = l2.sub(t3);
    Acc r2 = a1.mul(si);
    Acc t4 = a3.mul(co);
    r2 = r2.add(t4);
    a1 = static_cast<Acc &&>(l2);
    a3 = static_cast<Acc &&>(r2);
  }

  Vec32 m0 = a0.row_absmax();
  Vec32 m1 = a1.row_absmax();
  Vec32 m2 = a2.row_absmax();
  Vec32 m3 = a3.row_absmax();
  Vec32 m = m0.max(m1).max(m2.max(m3));
  m.mul_(kInv127);
  m.set_stride(scale.row_bytes);
  Tile t{};
  config_tile(t, q8);
  set_zone(t, nr, ZB);
  a0.quant_ext(m, t);
  store_tile_at(t, q8.data + (col_base + 0 * ZB) * q8.col_bytes);
  a1.quant_ext(m, t);
  store_tile_at(t, q8.data + (col_base + 1 * ZB) * q8.col_bytes);
  a2.quant_ext(m, t);
  store_tile_at(t, q8.data + (col_base + 2 * ZB) * q8.col_bytes);
  a3.quant_ext(m, t);
  store_tile_at(t, q8.data + (col_base + 3 * ZB) * q8.col_bytes);
  store_vec_at(m, scale.data + scale_off);
}

YAI_OP_INLINE void norm_rope_quant_heads(const MatrixView &x,
                                         const MatrixView &w,
                                         const MatrixView &cos_t,
                                         const MatrixView &sin_t,
                                         const MatrixView &q8,
                                         const MatrixView &scale,
                                         int heads, float inv_hd,
                                         bool do_rope) {
  constexpr std::uint8_t ZB = 32;
  const std::uint8_t nr = block_zone(x.rows, 0);
  const std::uint32_t HD = x.cols / static_cast<std::uint32_t>(heads);
  Acc a0{}, a1{}, a2{}, a3{};
  config_acc(a0, x); config_acc(a1, x); config_acc(a2, x); config_acc(a3, x);
  for (int h = 0; h < heads; ++h) {
    const std::uint32_t hc = static_cast<std::uint32_t>(h) * HD;
    set_zone(a0, nr, ZB); load_acc_at(a0, x.data + hc * x.col_bytes);
    set_zone(a1, nr, ZB); load_acc_at(a1, x.data + (hc + 32) * x.col_bytes);
    set_zone(a2, nr, ZB); load_acc_at(a2, x.data + (hc + 64) * x.col_bytes);
    set_zone(a3, nr, ZB); load_acc_at(a3, x.data + (hc + 96) * x.col_bytes);
    norm_rope_quant_accs(a0, a1, a2, a3, w, cos_t, sin_t, q8, scale, hc,
                         static_cast<std::uint32_t>(h) * scale.col_bytes, nr,
                         inv_hd, do_rope);
  }
}

} // namespace qwen3_ops

#endif
