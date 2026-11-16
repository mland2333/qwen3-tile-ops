// ============================================================================
// per-128-group 的 GEMM 原语：D[m,n] = Σ_g ( A_g · B_g ) * a_scale[m,g] * b_scale[n]
//
// 动机：激活按 128 个元素一组量化后，一个组在 acc 里放得下，但 K 维会跨多个
// 组，不能像 per-token 那样 K 全累加完再乘一次 scale。这里按组：
//   int32 partial = A_g · B_g
//   f32 partial_f = partial * a_scale[g] * b_scale
//   f32 accum   += partial_f
// N 固定 128 列（4 个 32 列块），acc 预算：
//   4 int32 partial + 4 f32 partial + 4 f32 accum = 12 / 16。
// tile 预算：4 N 块 × 4 K 块 = 16 B tile + 4 A tile = 20 / 32。
//
// 这是原语版本，假设 rows<=32、N 是 128 的倍数、K 是 128 的倍数；先用于
// 数值/acc 预算验证，再往模型里接。
// ============================================================================
#ifndef QWEN3_OPS_GROUP_GEMM_H
#define QWEN3_OPS_GROUP_GEMM_H

#include "tile.h"
#include "config.h"
#include "io.h"
#include "matrix.h"
#include "norm_rope_quant.h"

namespace qwen3_ops {

YAI_OP_INLINE void gemm_nt_group_store(const MatrixView &a,
                                       const MatrixView &a_scale,
                                       const MatrixView &b,
                                       const MatrixView &b_scale,
                                       const MatrixView &out,
                                       std::uint32_t group_k = 128,
                                       const MatrixView *add = nullptr,
                                       const MatrixView *q8 = nullptr,
                                       const MatrixView *qs = nullptr,
                                       const MatrixView *nw = nullptr,
                                       const MatrixView *cos = nullptr,
                                       const MatrixView *sin = nullptr,
                                       float inv_hd = 0.0f,
                                       Acc *ao0 = nullptr, Acc *ao1 = nullptr,
                                       Acc *ao2 = nullptr, Acc *ao3 = nullptr) {
  constexpr std::uint32_t NG = 128;
  constexpr std::uint8_t ZB = 32;
  const std::uint8_t rows = block_zone(a.rows, 0);
  const std::uint32_t K = a.cols;
  const std::uint32_t N = b.rows;
  if (K % group_k != 0 || N % NG != 0) {
    return;  // 原语只处理整组；模型集成时再补 tail
  }

  // 4 个输出 32 列块。tile 层不允许把 Acc 放数组，所以显式写 4 份。
  Acc acc0{}, acc1{}, acc2{}, acc3{};
  Acc part0{}, part1{}, part2{}, part3{};
  Acc pf0{}, pf1{}, pf2{}, pf3{};
  acc0.set_dtype(AccDType::F32); set_zone(acc0, rows, ZB);
  acc1.set_dtype(AccDType::F32); set_zone(acc1, rows, ZB);
  acc2.set_dtype(AccDType::F32); set_zone(acc2, rows, ZB);
  acc3.set_dtype(AccDType::F32); set_zone(acc3, rows, ZB);
  acc0.set_strides(out.row_bytes, out.col_bytes);
  acc1.set_strides(out.row_bytes, out.col_bytes);
  acc2.set_strides(out.row_bytes, out.col_bytes);
  acc3.set_strides(out.row_bytes, out.col_bytes);

  part0.set_dtype(AccDType::I32); set_zone(part0, rows, ZB);
  part1.set_dtype(AccDType::I32); set_zone(part1, rows, ZB);
  part2.set_dtype(AccDType::I32); set_zone(part2, rows, ZB);
  part3.set_dtype(AccDType::I32); set_zone(part3, rows, ZB);
  pf0.set_dtype(AccDType::F32); set_zone(pf0, rows, ZB);
  pf1.set_dtype(AccDType::F32); set_zone(pf1, rows, ZB);
  pf2.set_dtype(AccDType::F32); set_zone(pf2, rows, ZB);
  pf3.set_dtype(AccDType::F32); set_zone(pf3, rows, ZB);

  Tile at{};
  config_tile(at, a);
  set_zone(at, rows, ZB);
  Tile bt0{}, bt1{}, bt2{}, bt3{};
  config_tile(bt0, b); set_zone(bt0, ZB, ZB);
  config_tile(bt1, b); set_zone(bt1, ZB, ZB);
  config_tile(bt2, b); set_zone(bt2, ZB, ZB);
  config_tile(bt3, b); set_zone(bt3, ZB, ZB);

  Vec32 asv{};
  asv.set_dtype(Vec32DType::F32);
  asv.set_stride(a_scale.row_bytes);
  Vec32 bs0{}, bs1{}, bs2{}, bs3{};
  bs0.set_dtype(Vec32DType::F32); bs0.set_stride(4);
  bs1.set_dtype(Vec32DType::F32); bs1.set_stride(4);
  bs2.set_dtype(Vec32DType::F32); bs2.set_stride(4);
  bs3.set_dtype(Vec32DType::F32); bs3.set_stride(4);

  for (std::uint32_t n0 = 0; n0 < N; n0 += NG) {
    acc0.fill(0.0f);
    acc1.fill(0.0f);
    acc2.fill(0.0f);
    acc3.fill(0.0f);

    // b_scale 只依赖 N 块，和 K 组无关：提到 K 循环外，避免每组重载。
    set_len(bs0, ZB); load_vec_at(bs0, b_scale.data + (n0 + 0 * ZB) * 4);
    set_len(bs1, ZB); load_vec_at(bs1, b_scale.data + (n0 + 1 * ZB) * 4);
    set_len(bs2, ZB); load_vec_at(bs2, b_scale.data + (n0 + 2 * ZB) * 4);
    set_len(bs3, ZB); load_vec_at(bs3, b_scale.data + (n0 + 3 * ZB) * 4);

    for (std::uint32_t k0 = 0; k0 < K; k0 += group_k) {
      const std::uint32_t kc = group_k;
      // 第一个 K 块必须 zero=true；把它放在循环外，保证 matmul 的 zero 是
      // 编译期常量（后端要求 field selector 是常量，不能 if-convert 成 select）。
      {
        const std::uint8_t nc = block_zone(K, k0);
        set_zone(at, rows, nc);
        load_tile_at(at, a.data + k0 * static_cast<std::uint32_t>(a.col_bytes));
        set_zone(bt0, ZB, nc);
        load_tile_at(bt0, b.data + (n0 + 0 * ZB) * static_cast<std::uint32_t>(b.row_bytes) +
                          k0 * static_cast<std::uint32_t>(b.col_bytes));
        set_zone(bt1, ZB, nc);
        load_tile_at(bt1, b.data + (n0 + 1 * ZB) * static_cast<std::uint32_t>(b.row_bytes) +
                          k0 * static_cast<std::uint32_t>(b.col_bytes));
        set_zone(bt2, ZB, nc);
        load_tile_at(bt2, b.data + (n0 + 2 * ZB) * static_cast<std::uint32_t>(b.row_bytes) +
                          k0 * static_cast<std::uint32_t>(b.col_bytes));
        set_zone(bt3, ZB, nc);
        load_tile_at(bt3, b.data + (n0 + 3 * ZB) * static_cast<std::uint32_t>(b.row_bytes) +
                          k0 * static_cast<std::uint32_t>(b.col_bytes));
        at.matmul(bt0, part0, true, false, true);
        at.matmul(bt1, part1, true, false, true);
        at.matmul(bt2, part2, true, false, true);
        at.matmul(bt3, part3, true, false, true);
      }
      for (std::uint32_t kk = ZB; kk + ZB < kc; kk += ZB) {
        const std::uint8_t nc = block_zone(K, k0 + kk);
        set_zone(at, rows, nc);
        load_tile_at(at, a.data + (k0 + kk) * static_cast<std::uint32_t>(a.col_bytes));
        set_zone(bt0, ZB, nc);
        load_tile_at(bt0, b.data + (n0 + 0 * ZB) * static_cast<std::uint32_t>(b.row_bytes) +
                          (k0 + kk) * static_cast<std::uint32_t>(b.col_bytes));
        set_zone(bt1, ZB, nc);
        load_tile_at(bt1, b.data + (n0 + 1 * ZB) * static_cast<std::uint32_t>(b.row_bytes) +
                          (k0 + kk) * static_cast<std::uint32_t>(b.col_bytes));
        set_zone(bt2, ZB, nc);
        load_tile_at(bt2, b.data + (n0 + 2 * ZB) * static_cast<std::uint32_t>(b.row_bytes) +
                          (k0 + kk) * static_cast<std::uint32_t>(b.col_bytes));
        set_zone(bt3, ZB, nc);
        load_tile_at(bt3, b.data + (n0 + 3 * ZB) * static_cast<std::uint32_t>(b.row_bytes) +
                          (k0 + kk) * static_cast<std::uint32_t>(b.col_bytes));
        at.matmul(bt0, part0, false, false, true);
        at.matmul(bt1, part1, false, false, true);
        at.matmul(bt2, part2, false, false, true);
        at.matmul(bt3, part3, false, false, true);
      }

      // 最后一个 K 块：mma + drain 作为相邻 pseudo；前面的块已经累加到 part。
      {
        const std::uint32_t kk = kc - ZB;
        const std::uint8_t nc = block_zone(K, k0 + kk);
        set_zone(at, rows, nc);
        load_tile_at(at, a.data + (k0 + kk) * static_cast<std::uint32_t>(a.col_bytes));
        set_zone(bt0, ZB, nc);
        load_tile_at(bt0, b.data + (n0 + 0 * ZB) * static_cast<std::uint32_t>(b.row_bytes) +
                          (k0 + kk) * static_cast<std::uint32_t>(b.col_bytes));
        set_zone(bt1, ZB, nc);
        load_tile_at(bt1, b.data + (n0 + 1 * ZB) * static_cast<std::uint32_t>(b.row_bytes) +
                          (k0 + kk) * static_cast<std::uint32_t>(b.col_bytes));
        set_zone(bt2, ZB, nc);
        load_tile_at(bt2, b.data + (n0 + 2 * ZB) * static_cast<std::uint32_t>(b.row_bytes) +
                          (k0 + kk) * static_cast<std::uint32_t>(b.col_bytes));
        set_zone(bt3, ZB, nc);
        load_tile_at(bt3, b.data + (n0 + 3 * ZB) * static_cast<std::uint32_t>(b.row_bytes) +
                          (k0 + kk) * static_cast<std::uint32_t>(b.col_bytes));
        set_len(asv, rows);
        load_vec_at(asv, a_scale.data + (k0 / group_k) * 4);
        at.matmul_dequant(bt0, part0, pf0, bs0, asv, false, false, true);
        at.matmul_dequant(bt1, part1, pf1, bs1, asv, false, false, true);
        at.matmul_dequant(bt2, part2, pf2, bs2, asv, false, false, true);
        at.matmul_dequant(bt3, part3, pf3, bs3, asv, false, false, true);
        acc0.add_(pf0);
        acc1.add_(pf1);
        acc2.add_(pf2);
        acc3.add_(pf3);
      }
    }

    if (ao0) {
      *ao0 = static_cast<Acc &&>(acc0);
      *ao1 = static_cast<Acc &&>(acc1);
      *ao2 = static_cast<Acc &&>(acc2);
      *ao3 = static_cast<Acc &&>(acc3);
    } else if (nw) {
      // acc-level Q/K：GEMM 的 4 个 f32 acc 直接做 norm/rope/quant。
      norm_rope_quant_accs(acc0, acc1, acc2, acc3, *nw, *cos, *sin, *q8, *qs,
                           n0, (n0 / 128) * static_cast<std::uint32_t>(qs->col_bytes),
                           rows, inv_hd, true);
    } else if (q8) {
      // V 的 GEMM + per-head quant 融合：4 个 32 列块就是一个 head。
      Vec32 m0 = acc0.row_absmax();
      Vec32 m1 = acc1.row_absmax();
      Vec32 m2 = acc2.row_absmax();
      Vec32 m3 = acc3.row_absmax();
      Vec32 m = m0.max(m1).max(m2.max(m3));
      m.mul_(kInv127);
      m.set_stride(qs->row_bytes);
      Tile qt{};
      config_tile(qt, *q8);
      set_zone(qt, rows, ZB);
      acc0.quant_ext(m, qt);
      store_tile_at(qt, q8->data + (n0 + 0 * ZB) * static_cast<std::uint32_t>(q8->col_bytes));
      acc1.quant_ext(m, qt);
      store_tile_at(qt, q8->data + (n0 + 1 * ZB) * static_cast<std::uint32_t>(q8->col_bytes));
      acc2.quant_ext(m, qt);
      store_tile_at(qt, q8->data + (n0 + 2 * ZB) * static_cast<std::uint32_t>(q8->col_bytes));
      acc3.quant_ext(m, qt);
      store_tile_at(qt, q8->data + (n0 + 3 * ZB) * static_cast<std::uint32_t>(q8->col_bytes));
      store_vec_at(m, qs->data + (n0 / 128) * static_cast<std::uint32_t>(qs->col_bytes));
    } else if (add) {
      // 残差融合：out 就是 x，GEMM 结果直接加到 x 上；省掉 o 的 f32
      // write+read。pf0 在 K 循环后已经空闲，借来当 add 的临时 acc。
      pf0.set_strides(add->row_bytes, add->col_bytes);
      acc0.set_strides(add->row_bytes, add->col_bytes);
      acc1.set_strides(add->row_bytes, add->col_bytes);
      acc2.set_strides(add->row_bytes, add->col_bytes);
      acc3.set_strides(add->row_bytes, add->col_bytes);
      set_zone(pf0, rows, ZB);
      load_acc_at(pf0, add->data + (n0 + 0 * ZB) * static_cast<std::uint32_t>(add->col_bytes));
      acc0.add_(pf0);
      store_acc_at(acc0, add->data + (n0 + 0 * ZB) * static_cast<std::uint32_t>(add->col_bytes));
      load_acc_at(pf0, add->data + (n0 + 1 * ZB) * static_cast<std::uint32_t>(add->col_bytes));
      acc1.add_(pf0);
      store_acc_at(acc1, add->data + (n0 + 1 * ZB) * static_cast<std::uint32_t>(add->col_bytes));
      load_acc_at(pf0, add->data + (n0 + 2 * ZB) * static_cast<std::uint32_t>(add->col_bytes));
      acc2.add_(pf0);
      store_acc_at(acc2, add->data + (n0 + 2 * ZB) * static_cast<std::uint32_t>(add->col_bytes));
      load_acc_at(pf0, add->data + (n0 + 3 * ZB) * static_cast<std::uint32_t>(add->col_bytes));
      acc3.add_(pf0);
      store_acc_at(acc3, add->data + (n0 + 3 * ZB) * static_cast<std::uint32_t>(add->col_bytes));
    } else {
      store_acc_at(acc0, out.data + (n0 + 0 * ZB) * static_cast<std::uint32_t>(out.col_bytes));
      store_acc_at(acc1, out.data + (n0 + 1 * ZB) * static_cast<std::uint32_t>(out.col_bytes));
      store_acc_at(acc2, out.data + (n0 + 2 * ZB) * static_cast<std::uint32_t>(out.col_bytes));
      store_acc_at(acc3, out.data + (n0 + 3 * ZB) * static_cast<std::uint32_t>(out.col_bytes));
    }
  }
}

} // namespace qwen3_ops

#endif
