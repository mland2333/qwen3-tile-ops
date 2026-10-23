// ============================================================================
// gemm_nt：int8 GEMM（右操作数逻辑转置）+ 双向 scale 的 epilogue
//
//   out[m][n] = ( Σ_k A[m][k] * B[n][k] ) * a_scale[m] * b_scale[n]
//
// 结构：m 块（A 的行）→ n 组（每组 G 个输出列块）→ k 块（INT32 累加）
//
// 性能要点（均为实测）:
//   1) 描述符外提：stride/dtype/zone 与块位置无关，只在生命周期起点写一次；
//      内层每次载入只写 base_addr（见 io.h 文件头）。
//   2) n 方向寄存器分块：不同输出列块只共享 A，一次 A 载入喂 G 个 mma。
//      G = 8：1 A tile + 8 B tile = 9 个 XTT（上限 16），8 个 acc（上限 16）。
//      注意 G 不是越大越好：更大的 G 让组内临时值更多、RA 更难排，实测有最优值。
//   3) 组内的 tile/acc 都在**组内创建**（活区间短），这样 RA 能把寄存器复用起来。
//
// 为什么 K 要累加完才转换：mma 是 i8×i8→i32，K 方向全程在 i32 里累加，只在最后做一次
// acvt + 两次 scale —— 与 host 参考逐位一致（中间转 f32 会多一次舍入）。
// ============================================================================
#ifndef YAI_QWEN3_OPS_GEMM_H
#define YAI_QWEN3_OPS_GEMM_H

#include "tile.h"
#include "config.h"
#include "io.h"
#include "matrix.h"

namespace qwen3_ops {

// 一组里放几个输出列块（XTT/XTA 各 16 个的硬限制：G + 1 <= 16）。
inline constexpr std::uint32_t kGemmNGroup = 8;

// 单个输出列块的完整 K 累加 + epilogue：尾块（不足一组）与 K 不是 32 倍数时
// 走这条兜底路径，所以它自己负责全部 zone/stride 配置。
YAI_ALWAYS_INLINE void gemm_one_n_block(const MatrixView &a, const MatrixView &b,
                                        const MatrixView &b_scale,
                                        const Vec32 &as, const MatrixView &out,
                                        Tile &ta, Tile &tb, std::uint32_t m,
                                        std::uint32_t n) {
  const std::uint8_t rows = block_zone(a.rows, m);
  const std::uint8_t cols = block_zone(b.rows, n);
  Acc o{};
  init_i32_acc(o, rows, cols);
  set_zone(ta, rows, static_cast<std::uint8_t>(kBlock));
  set_zone(tb, cols, static_cast<std::uint8_t>(kBlock));
  load_tile_at(ta, a.data + m * static_cast<std::uint32_t>(a.row_bytes));
  load_tile_at(tb, b.data + n * static_cast<std::uint32_t>(b.row_bytes));
  ta.matmul(tb, o, /*zero=*/true, /*lt=*/false, /*rt=*/true);
  for (std::uint32_t k = kBlock; k < a.cols; k += kBlock) {
    load_tile_at(ta, a.data + m * static_cast<std::uint32_t>(a.row_bytes) +
                         k * static_cast<std::uint32_t>(a.col_bytes));
    load_tile_at(tb, b.data + n * static_cast<std::uint32_t>(b.row_bytes) +
                         k * static_cast<std::uint32_t>(b.col_bytes));
    ta.matmul(tb, o, /*zero=*/false, /*lt=*/false, /*rt=*/true);
  }
  o.to_fp32();
  o.mul(as, true);
  Vec32 ws{};
  load_vec_n(ws, b_scale, n, block_zone(b.rows, n));
  o.mul(ws, false);
  store_acc(o, out, m, n);
}

// 快路径的输出块 epilogue。
YAI_ALWAYS_INLINE void gemm_epilogue_block(const Vec32 &as, Vec32 &ws,
                                           Acc &o, const MatrixView &b_scale,
                                           const MatrixView &out,
                                           std::uint32_t m, std::uint32_t n) {
  o.to_fp32();
  o.mul(as, true);
  ws.load();                            // 描述符自增（len*stride）指向下一个 n 块
  o.mul(ws, false);
  // acc 的 strides 已按 out 配好（与位置无关），这里只写 base_addr
  store_acc_at(o, out.data + m * static_cast<std::uint32_t>(out.row_bytes) +
                      n * static_cast<std::uint32_t>(out.col_bytes));
}

YAI_OP_INLINE void gemm_nt(const MatrixView &a, const MatrixView &b,
                               const MatrixView &b_scale,
                               const MatrixView &a_scale,
                               const MatrixView &out) {
  constexpr std::uint32_t G = kGemmNGroup;
  constexpr std::uint8_t ZB = static_cast<std::uint8_t>(kBlock);

  Tile ta{};
  config_tile(ta, a);
  // 描述符自增：按列自增的步长是 cols*col_stride（= 块的字节宽度），
  // 正好是下一个 k 块；于是内层每次只要 tload，不必再写 base_addr / 算地址。
  ta.set_flags(2);
  // A tile 的 zone 只依赖 m 的行块，col 恒为 32 → 在 m 循环里设一次
  // ws（b_scale 向量）：dtype/len/stride 全程不变，一次性配好
  Vec32 ws{};
  config_vec(ws, b_scale);
  set_len(ws, ZB);
  // 向量描述符自增：每次 vload 后 base += len*stride（= 下一个 n 块的 32 个 scale）
  ws.set_flags(1);
  // B 的 G 个 tile 在函数级创建：zone 全程不变，组内只重写一次 base_addr，
  // 之后靠描述符自增前进（flags=2 = 按列自增一整个块的宽度）。
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
  const std::uint32_t ac = static_cast<std::uint32_t>(a.col_bytes);
  const std::uint32_t bb = static_cast<std::uint32_t>(b.row_bytes);
  const std::uint32_t bc = static_cast<std::uint32_t>(b.col_bytes);

  for (std::uint32_t m = 0; m < a.rows; m += kBlock) {
    const std::uint8_t rows = block_zone(a.rows, m);
    Vec32 as{};
    load_vec_n(as, a_scale, m, rows);
    set_zone(ta, rows, ZB);
    ws.set_data(b_scale.data);          // n 从 0 开始，之后靠自增
    std::uint32_t n = 0;
    // ---- 快路径：K 是 32 的倍数，且这一组 G 个输出块都满 32 列 ----
    // （zone 因此在整个 k 循环里不变，可以提到循环外）
    if (a.cols % kBlock == 0) {
      const std::uint32_t am = m * ab;
      for (; n + G * kBlock <= b.rows; n += G * kBlock) {
        // 组内临时值：活区间只到本组结束
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
        o0.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        o1.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        o2.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        o3.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        o4.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        o5.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        o6.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        o7.set_strides(out.row_bytes, out.col_bytes);   // store 用，与位置无关
        const std::uint32_t bn = n * bb;
        ta.set_data(a.data + am);            // 本组起点写一次
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
        // 其余 K 块：描述符自增已经指到下一块，这里只需要 tload
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
        gemm_epilogue_block(as, ws, o0, b_scale, out, m, n);
        gemm_epilogue_block(as, ws, o1, b_scale, out, m, n + 1 * kBlock);
        gemm_epilogue_block(as, ws, o2, b_scale, out, m, n + 2 * kBlock);
        gemm_epilogue_block(as, ws, o3, b_scale, out, m, n + 3 * kBlock);
        gemm_epilogue_block(as, ws, o4, b_scale, out, m, n + 4 * kBlock);
        gemm_epilogue_block(as, ws, o5, b_scale, out, m, n + 5 * kBlock);
        gemm_epilogue_block(as, ws, o6, b_scale, out, m, n + 6 * kBlock);
        gemm_epilogue_block(as, ws, o7, b_scale, out, m, n + 7 * kBlock);
      }
    }

    // ---- 兜底路径：不足一组的尾块、以及 K 不是 32 倍数的情形 ----
    Tile tb{};
    config_tile(tb, b);
    for (; n < b.rows; n += kBlock)
      gemm_one_n_block(a, b, b_scale, as, out, ta, tb, m, n);
  }
}

} // namespace qwen3_ops

#endif
