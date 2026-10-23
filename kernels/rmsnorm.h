#ifndef YAI_QWEN3_OPS_RMSNORM_H
#define YAI_QWEN3_OPS_RMSNORM_H

// ============================================================================
// rmsnorm_rows：行 RMSNorm
//
//   out[t][c] = x[t][c] * rsqrt(mean_c(x[t][c]^2) + eps) * w[c]
//
// 形状：x/out 是 [rows][cols] f32，w 是 [cols] f32，out 可以与 x 同址（原地）。
//
// 为什么是**两遍**扫描：每一行的缩放系数取决于整行的平方和，而一个 tile 只有
// 32 列；必须先把所有列块的 sumsq 合并成一个 per-row 向量，再回头逐块缩放。
// （host 参考同样是两遍，注释里也强调"不能对每个 32 列块分别归一化"。）
//
// 指令映射（一个 32x32 块）：
//   load_acc         → cfg.setx amN/acN + aload
//   row_sum_sq       → areduce.rows.sumsq（跨块用 vadd 合并）
//   mul(inv)/add(eps)/rsqrt
//                    → vmulx / vaddx / vrsqrt.approx
//   mul(ss, byrow)   → amulv.byrow（每行乘本行的 inv_rms）
//   mul(w, bycol)    → amulv.bycol（每列乘自己的 weight）
//   store_acc        → astore
// ============================================================================
#include "tile.h"

#include "config.h"
#include "io.h"
#include "matrix.h"

namespace qwen3_ops {

// inv 是 1/cols（由调用方传常量）：设备端没有标量浮点除法，如果在这里写
// `1.0f / x.cols` 会拉 libgcc 的 __divsf3，裸机链接里没有它。
YAI_OP_INLINE void rmsnorm_rows(const MatrixView &x, const MatrixView &w,
                         const MatrixView &out, float inv) {
  // 描述符外提：b/wv 的 stride+dtype 与位置无关 → 只配一次；zone 依赖块形状
  // （尾块可能不满），所以放在行块/列块层。
  Acc b{};
  config_acc(b, x);
  Vec32 wv{};
  config_vec(wv, w);      // dtype + stride 只配一次，len/addr 每块写

  for (std::uint32_t m = 0; m < x.rows; m += kBlock) {
    const std::uint8_t nr = block_zone(x.rows, m);
    // ---- 第一遍：整行平方和 ----
    Vec32 ss{};
    #pragma clang loop YAI_LOOP_UNROLL
    for (std::uint32_t k = 0; k < x.cols; k += kBlock) {
      set_zone(b, nr, block_zone(x.cols, k));
      load_acc_at(b, x.data + m * static_cast<std::uint32_t>(x.row_bytes) +
                     k * static_cast<std::uint32_t>(x.col_bytes));
      if (k == 0) {
        ss = b.row_sum_sq();
      } else {
        b.row_sum_sq(ss);
      }
    }
    // ---- 缩放系数：mean → +eps → rsqrt（只用一条 vrsqrt，与参考同式子）----
    ss.mul(inv);
    ss.add(kEps);
    ss.rsqrt();

    // ---- 第二遍：先按行缩放，再按列乘 weight，写回 ----
    #pragma clang loop YAI_LOOP_UNROLL
    for (std::uint32_t k = 0; k < x.cols; k += kBlock) {
      const std::uint8_t nc = block_zone(x.cols, k);
      set_zone(b, nr, nc);
      load_acc_at(b, x.data + m * static_cast<std::uint32_t>(x.row_bytes) +
                     k * static_cast<std::uint32_t>(x.col_bytes));
      // weight：长度只依赖块宽，地址随 k 走
      set_len(wv, nc);
      load_vec_at(wv, w.data + k * static_cast<std::uint32_t>(w.row_bytes));
      b.mul(ss, true);
      b.mul(wv, false);
      store_acc_at(b, out.data + m * static_cast<std::uint32_t>(out.row_bytes) +
                      k * static_cast<std::uint32_t>(out.col_bytes));
    }
  }
}

} // namespace qwen3_ops

#endif
