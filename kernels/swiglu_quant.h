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

// gate/up acc-level：同一个函数里内联两遍 K-loop，直接累加进局部 gate/up
// acc，最后在 acc 上 SwiGLU + per-128 quant；不做跨调用的 acc move。
#define QWEN3_GATEUP_KLOOP(O0, O1, O2, O3, B, BS) do { \
O0.set_dtype(AccDType::F32); set_zone(O0, nr, ZB); O0.fill(0.0f); \
O1.set_dtype(AccDType::F32); set_zone(O1, nr, ZB); O1.fill(0.0f); \
O2.set_dtype(AccDType::F32); set_zone(O2, nr, ZB); O2.fill(0.0f); \
O3.set_dtype(AccDType::F32); set_zone(O3, nr, ZB); O3.fill(0.0f); \
Tile at{}, bt0{}, bt1{}, bt2{}, bt3{}; \
config_tile(at, a); set_zone(at, nr, ZB); \
config_tile(bt0, B); set_zone(bt0, ZB, ZB); \
config_tile(bt1, B); set_zone(bt1, ZB, ZB); \
config_tile(bt2, B); set_zone(bt2, ZB, ZB); \
config_tile(bt3, B); set_zone(bt3, ZB, ZB); \
Acc p0{}, p1{}, p2{}, p3{}, pf{}; \
p0.set_dtype(AccDType::I32); set_zone(p0, nr, ZB); \
p1.set_dtype(AccDType::I32); set_zone(p1, nr, ZB); \
p2.set_dtype(AccDType::I32); set_zone(p2, nr, ZB); \
p3.set_dtype(AccDType::I32); set_zone(p3, nr, ZB); \
pf.set_dtype(AccDType::F32); set_zone(pf, nr, ZB); \
Vec32 asv{}, bs0{}, bs1{}, bs2{}, bs3{}; \
asv.set_dtype(Vec32DType::F32); asv.set_stride(a_scale.row_bytes); \
bs0.set_dtype(Vec32DType::F32); bs0.set_stride(4); \
bs1.set_dtype(Vec32DType::F32); bs1.set_stride(4); \
bs2.set_dtype(Vec32DType::F32); bs2.set_stride(4); \
bs3.set_dtype(Vec32DType::F32); bs3.set_stride(4); \
set_len(bs0, ZB); load_vec_at(bs0, BS.data + 0 * ZB * 4); \
set_len(bs1, ZB); load_vec_at(bs1, BS.data + 1 * ZB * 4); \
set_len(bs2, ZB); load_vec_at(bs2, BS.data + 2 * ZB * 4); \
set_len(bs3, ZB); load_vec_at(bs3, BS.data + 3 * ZB * 4); \
for (std::uint32_t k0 = 0; k0 < K; k0 += group_k) { \
const std::uint8_t nc0 = block_zone(K, k0); \
set_zone(at, nr, nc0); \
load_tile_at(at, a.data + k0 * a.col_bytes); \
set_zone(bt0, ZB, nc0); load_tile_at(bt0, B.data + 0 * ZB * B.row_bytes + k0 * B.col_bytes); \
set_zone(bt1, ZB, nc0); load_tile_at(bt1, B.data + 1 * ZB * B.row_bytes + k0 * B.col_bytes); \
set_zone(bt2, ZB, nc0); load_tile_at(bt2, B.data + 2 * ZB * B.row_bytes + k0 * B.col_bytes); \
set_zone(bt3, ZB, nc0); load_tile_at(bt3, B.data + 3 * ZB * B.row_bytes + k0 * B.col_bytes); \
at.matmul(bt0, p0, true, false, true); \
at.matmul(bt1, p1, true, false, true); \
at.matmul(bt2, p2, true, false, true); \
at.matmul(bt3, p3, true, false, true); \
for (std::uint32_t kk = ZB; kk < group_k; kk += ZB) { \
const std::uint8_t nc = block_zone(K, k0 + kk); \
set_zone(at, nr, nc); \
load_tile_at(at, a.data + (k0 + kk) * a.col_bytes); \
set_zone(bt0, ZB, nc); load_tile_at(bt0, B.data + 0 * ZB * B.row_bytes + (k0 + kk) * B.col_bytes); \
set_zone(bt1, ZB, nc); load_tile_at(bt1, B.data + 1 * ZB * B.row_bytes + (k0 + kk) * B.col_bytes); \
set_zone(bt2, ZB, nc); load_tile_at(bt2, B.data + 2 * ZB * B.row_bytes + (k0 + kk) * B.col_bytes); \
set_zone(bt3, ZB, nc); load_tile_at(bt3, B.data + 3 * ZB * B.row_bytes + (k0 + kk) * B.col_bytes); \
at.matmul(bt0, p0, false, false, true); \
at.matmul(bt1, p1, false, false, true); \
at.matmul(bt2, p2, false, false, true); \
at.matmul(bt3, p3, false, false, true); \
} \
set_len(asv, nr); load_vec_at(asv, a_scale.data + (k0 / group_k) * 4); \
adequant(pf, p0, bs0, asv); O0.add_(pf); \
adequant(pf, p1, bs1, asv); O1.add_(pf); \
adequant(pf, p2, bs2, asv); O2.add_(pf); \
adequant(pf, p3, bs3, asv); O3.add_(pf); \
} \
} while (0)

YAI_OP_INLINE void gemm_gate_up_swiglu_quant(
    const MatrixView &a, const MatrixView &a_scale,
    const MatrixView &wg, const MatrixView &wgs,
    const MatrixView &wu, const MatrixView &wus,
    const MatrixView &s8, const MatrixView &ss,
    std::uint32_t group_k = 256) {
  constexpr std::uint8_t ZB = 32;
  const std::uint8_t nr = block_zone(a.rows, 0);
  const std::uint32_t K = a.cols;
  const std::uint32_t rows = a.rows;


  Tile t{}; config_tile(t, s8);
  Vec32 m{}; m.set_dtype(Vec32DType::F32); m.set_stride(ss.row_bytes);
  for (std::uint32_t g0 = 0; g0 < s8.cols; g0 += 128) {
    MatrixView bg = sub_view(wg, g0, 0, 128, K);
    MatrixView bgs = sub_view(wgs, g0, 0, 128, 1);
    MatrixView bu = sub_view(wu, g0, 0, 128, K);
    MatrixView bus = sub_view(wus, g0, 0, 128, 1);
    Acc g0a{}, g1a{}, g2a{}, g3a{}, u0a{}, u1a{}, u2a{}, u3a{};
    QWEN3_GATEUP_KLOOP(g0a, g1a, g2a, g3a, bg, bgs);
    QWEN3_GATEUP_KLOOP(u0a, u1a, u2a, u3a, bu, bus);

    Acc e{};
    e = g0a.copy(); e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_(); g0a.mul_(e); g0a.mul_(u0a);
    e = g1a.copy(); e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_(); g1a.mul_(e); g1a.mul_(u1a);
    e = g2a.copy(); e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_(); g2a.mul_(e); g2a.mul_(u2a);
    e = g3a.copy(); e.mul_(-1.0f); e.exp_(); e.add_(1.0f); e.rcp_(); g3a.mul_(e); g3a.mul_(u3a);

    Vec32 m0 = g0a.row_absmax();
    Vec32 m1 = g1a.row_absmax();
    Vec32 m2 = g2a.row_absmax();
    Vec32 m3 = g3a.row_absmax();
    m = m0.max(m1).max(m2.max(m3));
    m.mul_(kInv127);
    set_zone(t, nr, ZB);
    g0a.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 0 * ZB) * s8.col_bytes);
    g1a.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 1 * ZB) * s8.col_bytes);
    g2a.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 2 * ZB) * s8.col_bytes);
    g3a.quant_ext(m, t); store_tile_at(t, s8.data + (g0 + 3 * ZB) * s8.col_bytes);
    store_vec_at(m, ss.data + (g0 / 128) * ss.col_bytes);
  }
}

#undef QWEN3_GATEUP_KLOOP

} // namespace qwen3_ops

#endif
