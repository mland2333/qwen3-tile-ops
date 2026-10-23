#ifndef YAI_QWEN3_OPS_ROPE_H
#define YAI_QWEN3_OPS_ROPE_H

// ============================================================================
// rope_head：单头 RoPE（rotate-half，head_dim=32，tiny 模型用）
//
//   left  = a0*cos - a1*sin
//   right = a0*sin + a1*cos
//
// 其中 a0 = x[0 .. half)、a1 = x[half .. 2*half)，half = head_dim/2 = 16。
// 位置相关的 cos/sin 已经由 host 预先算成表（cos_t/sin_t），设备端不做三角
// 函数（ISA 没有 sin/cos，只有 exp2/rsqrt 这类近似指令）。
//
// 关键约束：**原地写回**。输出的前半会覆盖输入的前半，而右半的计算还要用
// 到输入的前半，所以必须"两个半区都算完再写回"（参考实现里也有相同的注释，
// 因为它同样是原地 rotate-half）。
//
// 为什么 x 的前半被 load 两次（left 一次、right 一次）：Acc 是寄存器引用，
// 没有便宜的拷贝；重新 aload 一次比想办法复制更简单。
//
// 指令映射：load_acc_h → aload（列数由调用方给）/ mul → amul / sub → asub /
// add → aadd / store_acc_h → astore。
// ============================================================================
#include "tile.h"

#include "config.h"
#include "io.h"
#include "matrix.h"

namespace qwen3_ops {

YAI_OP_INLINE void rope_head(const MatrixView &x, const MatrixView &cos_t,
                                 const MatrixView &sin_t) {
  // ---- 1) 载入本行的 cos/sin（只要 half=16 列）----
  Acc co{};
  Acc si{};
  load_acc_h(co, cos_t, 0, 0, kHalf);
  load_acc_h(si, sin_t, 0, 0, kHalf);

  // ---- 2) 左半：a0*cos - a1*sin ----
  Acc left{};
  load_acc_h(left, x, 0, 0, kHalf);
  left.mul(co);
  Acc t{};
  load_acc_h(t, x, 0, kHalf, kHalf);
  t.mul(si);
  left.sub(t);

  // ---- 3) 右半：a0*sin + a1*cos（此时 x 还没被改写）----
  Acc right{};
  load_acc_h(right, x, 0, 0, kHalf);
  right.mul(si);
  Acc t2{};
  load_acc_h(t2, x, 0, kHalf, kHalf);
  t2.mul(co);
  right.add(t2);

  // ---- 4) 两个输出都算完再写回，避免第二半覆盖第一半的输入 ----
  store_acc_h(left, x, 0, 0);
  store_acc_h(right, x, 0, kHalf);
}

} // namespace qwen3_ops

#endif
