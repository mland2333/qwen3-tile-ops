// ============================================================================
// SwiGLU + per-group quant 融合：读 f32 gate/up，按 8×32=256 列一组
// 在 acc 上算 silu(g)*u，直接写 s8 + ss；不再写 f32 sw。
// ============================================================================
#ifndef QWEN3_OPS_SWIGLU_QUANT_H
#define QWEN3_OPS_SWIGLU_QUANT_H

#include "tile.h"
#include "config.h"
#include "io.h"
#include "matrix.h"
#include "group_gemm.h"

namespace qwen3_ops {

YAI_OP_INLINE void swiglu_quant_heads(const MatrixView &gate,
                                      const MatrixView &up,
                                      const MatrixView &s8,
                                      const MatrixView &ss) {
  constexpr std::uint8_t ZB = 32;
  const std::uint8_t nr = block_zone(gate.rows, 0);
  Acc sw0{}, sw1{}, sw2{}, sw3{}, sw4{}, sw5{}, sw6{}, sw7{};
  Acc e{}, u{};
  config_acc(sw0, gate); config_acc(sw1, gate); config_acc(sw2, gate); config_acc(sw3, gate);
  config_acc(sw4, gate); config_acc(sw5, gate); config_acc(sw6, gate); config_acc(sw7, gate);
  config_acc(e, gate); config_acc(u, up);
  set_zone(sw0, nr, ZB); set_zone(sw1, nr, ZB); set_zone(sw2, nr, ZB); set_zone(sw3, nr, ZB);
  set_zone(sw4, nr, ZB); set_zone(sw5, nr, ZB); set_zone(sw6, nr, ZB); set_zone(sw7, nr, ZB);
  set_zone(e, nr, ZB); set_zone(u, nr, ZB);
  Tile t{};
  config_tile(t, s8);
  Vec32 m{};
  m.set_dtype(Vec32DType::F32);
  m.set_stride(ss.row_bytes);

  for (std::uint32_t g0 = 0; g0 < gate.cols; g0 += 256) {
    // 8 个 32 列块，sw0..sw7 常驻
    load_acc_at(sw0, gate.data + (g0 + 0 * ZB) * gate.col_bytes);
    load_acc_at(sw1, gate.data + (g0 + 1 * ZB) * gate.col_bytes);
    load_acc_at(sw2, gate.data + (g0 + 2 * ZB) * gate.col_bytes);
    load_acc_at(sw3, gate.data + (g0 + 3 * ZB) * gate.col_bytes);
    load_acc_at(sw4, gate.data + (g0 + 4 * ZB) * gate.col_bytes);
    load_acc_at(sw5, gate.data + (g0 + 5 * ZB) * gate.col_bytes);
    load_acc_at(sw6, gate.data + (g0 + 6 * ZB) * gate.col_bytes);
    load_acc_at(sw7, gate.data + (g0 + 7 * ZB) * gate.col_bytes);
    load_acc_at(e, gate.data + (g0 + 0 * ZB) * gate.col_bytes);
    load_acc_at(u, up.data + (g0 + 0 * ZB) * up.col_bytes);
    e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw0.mul_(e); sw0.mul_(u);
    load_acc_at(e, gate.data + (g0 + 1 * ZB) * gate.col_bytes);
    load_acc_at(u, up.data + (g0 + 1 * ZB) * up.col_bytes);
    e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw1.mul_(e); sw1.mul_(u);
    load_acc_at(e, gate.data + (g0 + 2 * ZB) * gate.col_bytes);
    load_acc_at(u, up.data + (g0 + 2 * ZB) * up.col_bytes);
    e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw2.mul_(e); sw2.mul_(u);
    load_acc_at(e, gate.data + (g0 + 3 * ZB) * gate.col_bytes);
    load_acc_at(u, up.data + (g0 + 3 * ZB) * up.col_bytes);
    e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw3.mul_(e); sw3.mul_(u);
    load_acc_at(e, gate.data + (g0 + 4 * ZB) * gate.col_bytes);
    load_acc_at(u, up.data + (g0 + 4 * ZB) * up.col_bytes);
    e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw4.mul_(e); sw4.mul_(u);
    load_acc_at(e, gate.data + (g0 + 5 * ZB) * gate.col_bytes);
    load_acc_at(u, up.data + (g0 + 5 * ZB) * up.col_bytes);
    e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw5.mul_(e); sw5.mul_(u);
    load_acc_at(e, gate.data + (g0 + 6 * ZB) * gate.col_bytes);
    load_acc_at(u, up.data + (g0 + 6 * ZB) * up.col_bytes);
    e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw6.mul_(e); sw6.mul_(u);
    load_acc_at(e, gate.data + (g0 + 7 * ZB) * gate.col_bytes);
    load_acc_at(u, up.data + (g0 + 7 * ZB) * up.col_bytes);
    e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw7.mul_(e); sw7.mul_(u);

    Vec32 m0 = sw0.row_absmax();
    Vec32 m1 = sw1.row_absmax();
    Vec32 m2 = sw2.row_absmax();
    Vec32 m3 = sw3.row_absmax();
    Vec32 m4 = sw4.row_absmax();
    Vec32 m5 = sw5.row_absmax();
    Vec32 m6 = sw6.row_absmax();
    Vec32 m7 = sw7.row_absmax();
    m = m0.max(m1).max(m2.max(m3)).max(m4.max(m5).max(m6.max(m7)));
    m.mul_(kInv127);
    set_zone(t, nr, ZB);
    sw0.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 0 * ZB) * s8.col_bytes);
    sw1.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 1 * ZB) * s8.col_bytes);
    sw2.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 2 * ZB) * s8.col_bytes);
    sw3.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 3 * ZB) * s8.col_bytes);
    sw4.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 4 * ZB) * s8.col_bytes);
    sw5.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 5 * ZB) * s8.col_bytes);
    sw6.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 6 * ZB) * s8.col_bytes);
    sw7.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 7 * ZB) * s8.col_bytes);
    store_vec_at(m, ss.data + (g0 / 256) * ss.col_bytes);
  }
}

// gate/up acc-level：每个 32 列块先算 gate/up（i32 K 累加 + adequant），
// 立即在 Acc 上做 silu(g)*u；8 个 32 列块（256 列）一组做 absmax + quant。
// 不再写 f32 gt/ut。
YAI_ALWAYS_INLINE void gemm_one_block_gate_up(
    const MatrixView &a, const MatrixView &a_scale,
    const MatrixView &wg, const MatrixView &wgs,
    const MatrixView &wu, const MatrixView &wus,
    std::uint32_t n, std::uint32_t group_k, std::uint8_t rows,
    Acc &gate, Acc &up, Tile &ta, Tile &tb) {
  constexpr std::uint8_t ZB = 32;
  const std::uint32_t K = a.cols;
  Acc part_g{}, part_u{};
  part_g.set_dtype(AccDType::I32); set_zone(part_g, rows, ZB);
  part_u.set_dtype(AccDType::I32); set_zone(part_u, rows, ZB);
  Acc pf_g{}, pf_u{};
  pf_g.set_dtype(AccDType::F32); set_zone(pf_g, rows, ZB);
  pf_u.set_dtype(AccDType::F32); set_zone(pf_u, rows, ZB);
  Vec32 asv{};
  asv.set_dtype(Vec32DType::F32);
  asv.set_stride(a_scale.row_bytes);
  Vec32 wgv{}, wuv{};
  wgv.set_dtype(Vec32DType::F32); wgv.set_stride(4);
  wuv.set_dtype(Vec32DType::F32); wuv.set_stride(4);
  set_len(wgv, ZB); set_len(wuv, ZB);
  load_vec_at(wgv, wgs.data + n * 4);
  load_vec_at(wuv, wus.data + n * 4);

  gate.fill(0.0f);
  up.fill(0.0f);
  for (std::uint32_t k0 = 0; k0 < K; k0 += group_k) {
    const std::uint8_t nc0 = block_zone(K, k0);
    // A tile is loaded once per K block and reused by gate and up.
    set_zone(ta, rows, nc0);
    load_tile_at(ta, a.data + k0 * static_cast<std::uint32_t>(a.col_bytes));
    set_zone(tb, ZB, nc0);
    load_tile_at(tb, wg.data + n * static_cast<std::uint32_t>(wg.row_bytes) +
                     k0 * static_cast<std::uint32_t>(wg.col_bytes));
    ta.matmul(tb, part_g, true, false, true);
    load_tile_at(tb, wu.data + n * static_cast<std::uint32_t>(wu.row_bytes) +
                     k0 * static_cast<std::uint32_t>(wu.col_bytes));
    ta.matmul(tb, part_u, true, false, true);
    for (std::uint32_t kk = ZB; kk < group_k; kk += ZB) {
      const std::uint8_t nc = block_zone(K, k0 + kk);
      set_zone(ta, rows, nc);
      load_tile_at(ta, a.data + (k0 + kk) * static_cast<std::uint32_t>(a.col_bytes));
      set_zone(tb, ZB, nc);
      load_tile_at(tb, wg.data + n * static_cast<std::uint32_t>(wg.row_bytes) +
                       (k0 + kk) * static_cast<std::uint32_t>(wg.col_bytes));
      ta.matmul(tb, part_g, false, false, true);
      load_tile_at(tb, wu.data + n * static_cast<std::uint32_t>(wu.row_bytes) +
                       (k0 + kk) * static_cast<std::uint32_t>(wu.col_bytes));
      ta.matmul(tb, part_u, false, false, true);
    }
    set_len(asv, rows);
    load_vec_at(asv, a_scale.data + (k0 / group_k) * 4);
    adequant(pf_g, part_g, wgv, asv);
    gate.add_(pf_g);
    adequant(pf_u, part_u, wuv, asv);
    up.add_(pf_u);
  }
}

YAI_OP_INLINE void gemm_gate_up_swiglu_quant(
    const MatrixView &a, const MatrixView &a_scale,
    const MatrixView &wg, const MatrixView &wgs,
    const MatrixView &wu, const MatrixView &wus,
    const MatrixView &s8, const MatrixView &ss,
    std::uint32_t group_k = 256) {
  constexpr std::uint8_t ZB = 32;
  const std::uint8_t nr = block_zone(a.rows, 0);
  const std::uint32_t rows = a.rows;

  Tile ta{};
  config_tile(ta, a);
  Tile tb{};
  config_tile(tb, wg);
  Tile t{};
  config_tile(t, s8);
  Vec32 m{};
  m.set_dtype(Vec32DType::F32);
  m.set_stride(ss.row_bytes);

  Acc sw0{}, sw1{}, sw2{}, sw3{}, sw4{}, sw5{}, sw6{}, sw7{};
  Acc e{}, u{};
  init_f32_acc(sw0, nr, ZB); init_f32_acc(sw1, nr, ZB);
  init_f32_acc(sw2, nr, ZB); init_f32_acc(sw3, nr, ZB);
  init_f32_acc(sw4, nr, ZB); init_f32_acc(sw5, nr, ZB);
  init_f32_acc(sw6, nr, ZB); init_f32_acc(sw7, nr, ZB);
  init_f32_acc(e, nr, ZB); init_f32_acc(u, nr, ZB);

  for (std::uint32_t g0 = 0; g0 < s8.cols; g0 += 256) {
    gemm_one_block_gate_up(a, a_scale, wg, wgs, wu, wus,
                           g0 + 0 * ZB, group_k, rows, sw0, u, ta, tb);
    e = sw0.copy(); e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw0.mul_(e); sw0.mul_(u);

    gemm_one_block_gate_up(a, a_scale, wg, wgs, wu, wus,
                           g0 + 1 * ZB, group_k, rows, sw1, u, ta, tb);
    e = sw1.copy(); e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw1.mul_(e); sw1.mul_(u);

    gemm_one_block_gate_up(a, a_scale, wg, wgs, wu, wus,
                           g0 + 2 * ZB, group_k, rows, sw2, u, ta, tb);
    e = sw2.copy(); e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw2.mul_(e); sw2.mul_(u);

    gemm_one_block_gate_up(a, a_scale, wg, wgs, wu, wus,
                           g0 + 3 * ZB, group_k, rows, sw3, u, ta, tb);
    e = sw3.copy(); e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw3.mul_(e); sw3.mul_(u);

    gemm_one_block_gate_up(a, a_scale, wg, wgs, wu, wus,
                           g0 + 4 * ZB, group_k, rows, sw4, u, ta, tb);
    e = sw4.copy(); e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw4.mul_(e); sw4.mul_(u);

    gemm_one_block_gate_up(a, a_scale, wg, wgs, wu, wus,
                           g0 + 5 * ZB, group_k, rows, sw5, u, ta, tb);
    e = sw5.copy(); e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw5.mul_(e); sw5.mul_(u);

    gemm_one_block_gate_up(a, a_scale, wg, wgs, wu, wus,
                           g0 + 6 * ZB, group_k, rows, sw6, u, ta, tb);
    e = sw6.copy(); e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw6.mul_(e); sw6.mul_(u);

    gemm_one_block_gate_up(a, a_scale, wg, wgs, wu, wus,
                           g0 + 7 * ZB, group_k, rows, sw7, u, ta, tb);
    e = sw7.copy(); e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_();
    sw7.mul_(e); sw7.mul_(u);

    Vec32 m0 = sw0.row_absmax();
    Vec32 m1 = sw1.row_absmax();
    Vec32 m2 = sw2.row_absmax();
    Vec32 m3 = sw3.row_absmax();
    Vec32 m4 = sw4.row_absmax();
    Vec32 m5 = sw5.row_absmax();
    Vec32 m6 = sw6.row_absmax();
    Vec32 m7 = sw7.row_absmax();
    m = m0.max(m1).max(m2.max(m3)).max(m4.max(m5).max(m6.max(m7)));
    m.mul_(kInv127);
    m.set_stride(ss.row_bytes);
    set_zone(t, nr, ZB);
    sw0.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 0 * ZB) * s8.col_bytes);
    sw1.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 1 * ZB) * s8.col_bytes);
    sw2.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 2 * ZB) * s8.col_bytes);
    sw3.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 3 * ZB) * s8.col_bytes);
    sw4.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 4 * ZB) * s8.col_bytes);
    sw5.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 5 * ZB) * s8.col_bytes);
    sw6.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 6 * ZB) * s8.col_bytes);
    sw7.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 7 * ZB) * s8.col_bytes);
    store_vec_at(m, ss.data + (g0 / 256) * ss.col_bytes);
  }
}


} // namespace qwen3_ops

#endif
