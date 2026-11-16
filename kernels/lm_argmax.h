// ============================================================================
// lm_head 与 argmax 融合：一边算 logits，一边维护 (best, idx)，logits 不落内存。
//
// 和 gemm_nt 的 epilogue 相比，每个 32 列的输出块算完（i32 → adequant）后不
// store，而是：
//   tmax  = of.row_max()        // 本块每行的最大值（f32）
//   tidx  = of.row_argmax()     // 本块的局部下标（0..31）
//   gidx  = tidx + n            // 全局下标 = 块基址 + 局部
//   mask  = (best < tmax)       // vcmp.lt
//   best  = select(mask, tmax, best)
//   bidx  = select(mask, gidx, bidx)
// (best, bidx) 是循环携带的 Vec32 值，整轮 n 循环都不落内存；每个 m 行块结束
// 时只 store 一次。这依赖 vcmp/vselect 的值形式：它们返回新 SSA 值，调用方
// 的句柄会跟着更新；旧的 void 原地内建不会。
//
// 只在"K 是 32 的倍数、N 是 32 的倍数"时走快路径（lm_head 满足）；否则退回
// gemm_nt + argmax_logits。
// ============================================================================
#ifndef QWEN3_OPS_LM_ARGMAX_H
#define QWEN3_OPS_LM_ARGMAX_H

#include "gemm.h"

namespace qwen3_ops {

// 把 n（整数）当成 f32 位模式传下去：向量单元按 u32 解释这个标量，
// 于是 gidx.add_(bits_of(n)) 就是"u32 加 n"，不需要标量浮点转换。
YAI_ALWAYS_INLINE float bits_of(std::uint32_t v) {
  return *reinterpret_cast<float *>(&v);
}

// epilogue：o(i32) → f32 → 并进 (best, bidx)。
// best/bidx 是调用方的 Vec32 循环携带值（dtype/len 已配好），本函数只更新
// 它们，不读写内存。
YAI_ALWAYS_INLINE void argmax_epilogue(const Vec32 &as, Vec32 &ws,
                                       Acc &o, Vec32 &best, Vec32 &bidx,
                                       std::uint8_t rows, std::uint8_t cols,
                                       std::uint32_t n) {
  Acc of{};
  init_f32_acc(of, rows, cols);
  ws.load();                      // 列 scale：本 n 块（flags=1 时指向下一块）
  adequant(of, o, ws, as);        // 与 gemm_nt 同一乘序：(f32(o)·ws[j])·as[i]
  Vec32 tmax = of.row_max();
  set_len(tmax, rows);
  Vec32 gidx = of.row_argmax();
  set_len(gidx, rows);
  gidx.set_dtype(Vec32DType::I32);
  gidx.add_(bits_of(n));          // u32 + 标量（按位解释）
  Vec32 mask = cmp_lt(best, tmax);            // mask = best < tmax（u32）
  best = select(tmax, best, mask);            // best = mask ? tmax : best
  bidx = select(gidx, bidx, mask);            // bidx = mask ? gidx : bidx
}

// f32 logits 块上的 argmax epilogue（per-group GEMM 已经完成了 dequant）。
YAI_ALWAYS_INLINE void argmax_epilogue_f32(Acc &of, Vec32 &best, Vec32 &bidx,
                                           std::uint8_t rows, std::uint8_t cols,
                                           std::uint32_t n) {
  Vec32 tmax = of.row_max();
  set_len(tmax, rows);
  Vec32 gidx = of.row_argmax();
  set_len(gidx, rows);
  gidx.set_dtype(Vec32DType::I32);
  gidx.add_(bits_of(n));
  Vec32 mask = cmp_lt(best, tmax);
  best = select(tmax, best, mask);
  bidx = select(gidx, bidx, mask);
}

// per-128-group 的 lm_head + argmax：K 每组先 int32 partial，再乘 a_scale[g]
// 和 b_scale，累加 f32，最后对每个 32 列块做 argmax epilogue。
// N 每轮 4 个 32 列块（4 int + 4 f32 partial + 4 f32 accum = 12 acc）。
YAI_OP_INLINE void gemm_nt_group_argmax(const MatrixView &a,
                                        const MatrixView &a_scale,
                                        const MatrixView &b,
                                        const MatrixView &b_scale,
                                        Vec32 &best, Vec32 &bidx,
                                        std::uint32_t group_k = 128) {
  constexpr std::uint8_t ZB = 32;
  const std::uint8_t rows = block_zone(a.rows, 0);
  const std::uint32_t K = a.cols;
  const std::uint32_t N = b.rows;
  if (K % group_k != 0 || N % 128 != 0) return;

  Acc acc0{}, acc1{}, acc2{}, acc3{};
  Acc part0{}, part1{}, part2{}, part3{};
  Acc pf0{}, pf1{}, pf2{}, pf3{};
  acc0.set_dtype(AccDType::F32); set_zone(acc0, rows, ZB);
  acc1.set_dtype(AccDType::F32); set_zone(acc1, rows, ZB);
  acc2.set_dtype(AccDType::F32); set_zone(acc2, rows, ZB);
  acc3.set_dtype(AccDType::F32); set_zone(acc3, rows, ZB);
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

  for (std::uint32_t n0 = 0; n0 < N; n0 += 128) {
    acc0.fill(0.0f); acc1.fill(0.0f); acc2.fill(0.0f); acc3.fill(0.0f);
    // b_scale 只依赖 N 块，和 K 组无关：提到 K 循环外，避免每组重载。
    set_len(bs0, ZB); load_vec_at(bs0, b_scale.data + (n0 + 0 * ZB) * 4);
    set_len(bs1, ZB); load_vec_at(bs1, b_scale.data + (n0 + 1 * ZB) * 4);
    set_len(bs2, ZB); load_vec_at(bs2, b_scale.data + (n0 + 2 * ZB) * 4);
    set_len(bs3, ZB); load_vec_at(bs3, b_scale.data + (n0 + 3 * ZB) * 4);

    for (std::uint32_t k0 = 0; k0 < K; k0 += group_k) {
      // first K block (zero=true)
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
      for (std::uint32_t kk = ZB; kk < group_k; kk += ZB) {
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
      set_len(asv, rows);
      load_vec_at(asv, a_scale.data + (k0 / group_k) * 4);
      adequant(pf0, part0, bs0, asv);
      adequant(pf1, part1, bs1, asv);
      adequant(pf2, part2, bs2, asv);
      adequant(pf3, part3, bs3, asv);
      acc0.add_(pf0); acc1.add_(pf1); acc2.add_(pf2); acc3.add_(pf3);
    }
    argmax_epilogue_f32(acc0, best, bidx, rows, ZB, n0 + 0 * ZB);
    argmax_epilogue_f32(acc1, best, bidx, rows, ZB, n0 + 1 * ZB);
    argmax_epilogue_f32(acc2, best, bidx, rows, ZB, n0 + 2 * ZB);
    argmax_epilogue_f32(acc3, best, bidx, rows, ZB, n0 + 3 * ZB);
  }
}

// a: [rows][K] i8，b: [N][K] i8 行主序，b_scale: [N]，a_scale: [rows]。
// 输出 best（f32，rows lane）到 best_out、bidx（i32，rows lane）到 bidx_out；
// 两者都只在每个 m 行块结束时写一次，循环内部不碰内存。
YAI_OP_INLINE void gemm_nt_argmax(const MatrixView &a, const MatrixView &b,
                                  const MatrixView &b_scale,
                                  const MatrixView &a_scale,
                                  const MatrixView &best_out,
                                  const MatrixView &bidx_out) {
  constexpr std::uint32_t G = kGemmNGroup;
  constexpr std::uint8_t ZB = static_cast<std::uint8_t>(kBlock);

  Tile ta{};
  config_tile(ta, a);
  ta.set_flags(2);
  Vec32 ws{};
  config_vec(ws, b_scale);
  set_len(ws, ZB);
  ws.set_flags(1);

  Tile t0{};
  config_tile(t0, b);
  set_zone(t0, ZB, ZB);
  t0.set_flags(2);
  Tile t1{};
  config_tile(t1, b);
  set_zone(t1, ZB, ZB);
  t1.set_flags(2);
  Tile t2{};
  config_tile(t2, b);
  set_zone(t2, ZB, ZB);
  t2.set_flags(2);
  Tile t3{};
  config_tile(t3, b);
  set_zone(t3, ZB, ZB);
  t3.set_flags(2);
  Tile t4{};
  config_tile(t4, b);
  set_zone(t4, ZB, ZB);
  t4.set_flags(2);
  Tile t5{};
  config_tile(t5, b);
  set_zone(t5, ZB, ZB);
  t5.set_flags(2);
  Tile t6{};
  config_tile(t6, b);
  set_zone(t6, ZB, ZB);
  t6.set_flags(2);
  Tile t7{};
  config_tile(t7, b);
  set_zone(t7, ZB, ZB);
  t7.set_flags(2);

  const std::uint32_t ab = static_cast<std::uint32_t>(a.row_bytes);
  const std::uint32_t bb = static_cast<std::uint32_t>(b.row_bytes);

  for (std::uint32_t m = 0; m < a.rows; m += kBlock) {
    const std::uint8_t rows = block_zone(a.rows, m);
    Vec32 as{};
    load_vec_n(as, a_scale, m, rows);
    // 循环携带的 argmax 状态：只在本 m 块开始时初始化，n 循环里不落内存。
    Vec32 best{};
    best.set_dtype(Vec32DType::F32);
    best.set_stride(4);
    set_len(best, rows);
    best.fill(-3.0e38f);
    Vec32 bidx{};
    bidx.set_dtype(Vec32DType::I32);
    bidx.set_stride(4);
    set_len(bidx, rows);
    bidx.fill(0.0f);
    set_zone(ta, rows, ZB);
    ws.set_data(b_scale.data);
    const std::uint32_t am = m * ab;
    std::uint32_t n = 0;
    for (; n + G * kBlock <= b.rows; n += G * kBlock) {
      Acc o0{};
      init_i32_acc(o0, rows, ZB);
      Acc o1{};
      init_i32_acc(o1, rows, ZB);
      Acc o2{};
      init_i32_acc(o2, rows, ZB);
      Acc o3{};
      init_i32_acc(o3, rows, ZB);
      Acc o4{};
      init_i32_acc(o4, rows, ZB);
      Acc o5{};
      init_i32_acc(o5, rows, ZB);
      Acc o6{};
      init_i32_acc(o6, rows, ZB);
      Acc o7{};
      init_i32_acc(o7, rows, ZB);
      const std::uint32_t bn = n * bb;
      ta.set_data(a.data + am);
      ta.load();
      t0.set_data(b.data + bn);
      t0.load();
      t1.set_data(b.data + bn + 1 * kBlock * bb);
      t1.load();
      t2.set_data(b.data + bn + 2 * kBlock * bb);
      t2.load();
      t3.set_data(b.data + bn + 3 * kBlock * bb);
      t3.load();
      t4.set_data(b.data + bn + 4 * kBlock * bb);
      t4.load();
      t5.set_data(b.data + bn + 5 * kBlock * bb);
      t5.load();
      t6.set_data(b.data + bn + 6 * kBlock * bb);
      t6.load();
      t7.set_data(b.data + bn + 7 * kBlock * bb);
      t7.load();
      ta.matmul(t0, o0, /*zero=*/true, false, true);
      ta.matmul(t1, o1, /*zero=*/true, false, true);
      ta.matmul(t2, o2, /*zero=*/true, false, true);
      ta.matmul(t3, o3, /*zero=*/true, false, true);
      ta.matmul(t4, o4, /*zero=*/true, false, true);
      ta.matmul(t5, o5, /*zero=*/true, false, true);
      ta.matmul(t6, o6, /*zero=*/true, false, true);
      ta.matmul(t7, o7, /*zero=*/true, false, true);
      for (std::uint32_t k = kBlock; k < a.cols; k += kBlock) {
        ta.load();
        t0.load();
        t1.load();
        t2.load();
        t3.load();
        t4.load();
        t5.load();
        t6.load();
        t7.load();
        ta.matmul(t0, o0, /*zero=*/false, false, true);
        ta.matmul(t1, o1, /*zero=*/false, false, true);
        ta.matmul(t2, o2, /*zero=*/false, false, true);
        ta.matmul(t3, o3, /*zero=*/false, false, true);
        ta.matmul(t4, o4, /*zero=*/false, false, true);
        ta.matmul(t5, o5, /*zero=*/false, false, true);
        ta.matmul(t6, o6, /*zero=*/false, false, true);
        ta.matmul(t7, o7, /*zero=*/false, false, true);
      }
      argmax_epilogue(as, ws, o0, best, bidx, rows, ZB, n + 0 * kBlock);
      argmax_epilogue(as, ws, o1, best, bidx, rows, ZB, n + 1 * kBlock);
      argmax_epilogue(as, ws, o2, best, bidx, rows, ZB, n + 2 * kBlock);
      argmax_epilogue(as, ws, o3, best, bidx, rows, ZB, n + 3 * kBlock);
      argmax_epilogue(as, ws, o4, best, bidx, rows, ZB, n + 4 * kBlock);
      argmax_epilogue(as, ws, o5, best, bidx, rows, ZB, n + 5 * kBlock);
      argmax_epilogue(as, ws, o6, best, bidx, rows, ZB, n + 6 * kBlock);
      argmax_epilogue(as, ws, o7, best, bidx, rows, ZB, n + 7 * kBlock);
    }
    // 尾块（N 不是 32 的倍数）：lm_head 的 N 恒为 32 的倍数，这里只处理
    // 剩余的整 32 块。
    for (; n + kBlock <= b.rows; n += kBlock) {
      Acc o{};
      init_i32_acc(o, rows, ZB);
      ta.set_data(a.data + am);
      ta.load();
      t0.set_data(b.data + n * bb);
      t0.load();
      ta.matmul(t0, o, /*zero=*/true, false, true);
      for (std::uint32_t k = kBlock; k < a.cols; k += kBlock) {
        ta.load();
        t0.load();
        ta.matmul(t0, o, /*zero=*/false, false, true);
      }
      argmax_epilogue(as, ws, o, best, bidx, rows, ZB, n);
    }
    // 整个 n 循环结束，只在这里读回/写出一次。
    store_vec(best, best_out, m);
    store_vec(bidx, bidx_out, m);
  }
}

} // namespace qwen3_ops

#endif
