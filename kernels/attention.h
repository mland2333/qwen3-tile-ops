#ifndef YAI_QWEN3_OPS_ATTENTION_H
#define YAI_QWEN3_OPS_ATTENTION_H

// ============================================================================
// attention_head / attention_heads：单 KV 块（≤32 token）的多头 attention
//
// 这是 tiny 模型用的"最小版本"：q/k/v 都是一个 32x32 的 i8 块，head_dim=32，
// 整条流水线如下（每一步都对应一条/几条硬件指令）：
//
//   S    = q8 · k8ᵀ              mma.nt.zero.i8.i32（K 只有一块）
//   S   *= q_scale(byrow)        amulv.byrow    —— per query token 的量化 scale
//   S   *= k_scale(bycol)        amulv.bycol    —— per KV token 的量化 scale
//   mask: 上三角填 -inf          amaskx.tril（fill5 code 8 = -inf）
//   softmax: -row_max, exp2, /row_sum
//                                areduce.rows.max / vsub / vexp2 / areduce.rows.sum / vdiv
//   S   *= v_scale(bycol)        amulv.bycol    —— 把 V 的量化 scale 折进权重
//   P    = 量化(S)               areduce.rows.absmax → vmulx(1/127) → tquantx
//   O    = P · V                 mma.nn.zero.i8.i32
//   O   *= p_scale(byrow)        amulv.byrow    —— 反量化
//
// 为什么 V 的 scale 能折进 softmax 权重：O = Σ_j P_j * v8_j * vscale_j，
// 而 P 是逐行量化的（同一个 absmax），所以把 vscale 乘在 P 上、最后再乘一次
// 行 scale，数学上完全等价；这样就不需要"每列一个 scale"的 epilogue。
//
// 更通用、支持 KV 分块的版本见 attention_gqa.h（0.6B 用的是那个）。
// ============================================================================
#include "tile.h"

#include "config.h"
#include "io.h"
#include "matrix.h"

namespace qwen3_ops {

YAI_OP_INLINE void attention_head(const MatrixView &q8, const MatrixView &k8,
                                      const MatrixView &v8,
                                      const MatrixView &q_scale,
                                      const MatrixView &k_scale,
                                      const MatrixView &v_scale,
                                      const MatrixView &out, bool causal) {
  // ---- 1) 载入 q/k/v 三个 tile（都是 32x32 i8）----
  Tile qt{};
  Tile kt{};
  Tile vt{};
  load_tile(qt, q8, 0, 0);
  load_tile(kt, k8, 0, 0);
  load_tile(vt, v8, 0, 0);
  // ---- 2) 三条 per-token scale 向量 ----
  Vec32 qs{};
  Vec32 ks{};
  Vec32 vs{};
  load_vec(qs, q_scale, 0);
  load_vec(ks, k_scale, 0);
  load_vec(vs, v_scale, 0);

  // ---- 3) S = q8·k8ᵀ（右操作数逻辑转置：rt=true），i32 累加器 ----
  Acc s{};
  init_i32_acc(s, block_zone(q8.rows, 0), block_zone(k8.rows, 0));
  qt.matmul(kt, s, /*zero=*/true, /*lt=*/false, /*rt=*/true);

  // ---- 4) 转 f32，乘 Q/K 两个量化 scale ----
  s.to_fp32();
  s.mul(qs, true);
  s.mul(ks, false);

  // ---- 5) 因果 mask：保留 j <= i（tril），上三角填 -inf ----
  if (causal) {
    s.mask(0);
  }

  // ---- 6) softmax（减行最大值 → exp2 → 除行和）----
  Vec32 m = s.row_max();
  s.sub(m, true);
  s.exp();
  Vec32 den = s.row_sum();
  s.div(den, true);

  // ---- 7) 折叠 V 的 scale（见文件头说明）----
  s.mul(vs, false);

  // ---- 8) 把 P 量化成 i8（每个 query 行一个 scale）----
  Vec32 ps = s.row_absmax();
  ps.mul(kInv127);
  s.quant_ext(ps, qt);       // P 写回 qt（qt 已经用完）

  // ---- 9) O = P·V（nn：都不转置），再乘 P 的行 scale 反量化 ----
  Acc o{};
  init_i32_acc(o, block_zone(q8.rows, 0), block_zone(v8.cols, 0));
  qt.matmul(vt, o, /*zero=*/true, /*lt=*/false, /*rt=*/false);
  o.to_fp32();
  o.mul(ps, true);
  store_acc(o, out, 0, 0);
}

// ---------------------------------------------------------------------------
// attention_heads：按 head 切列块，逐个 head 调 attention_head。
//
// scale 的布局是 **head-major**：q_scale[head*32 + token]，每个头各 32 个
// per-token scale（tiny 模型就是这么量化的：每个头单独算 absmax）。
// ---------------------------------------------------------------------------
YAI_OP_INLINE void attention_heads(const MatrixView &q8,
                                       const MatrixView &k8,
                                       const MatrixView &v8,
                                       const MatrixView &q_scale,
                                       const MatrixView &k_scale,
                                       const MatrixView &v_scale,
                                       const MatrixView &out, int heads,
                                       int head_dim, bool causal) {
  for (int h = 0; h < heads; ++h) {
    const std::uint32_t c0 = static_cast<std::uint32_t>(h * head_dim);
    const std::uint32_t s0 = static_cast<std::uint32_t>(h * kBlock);
    const std::uint32_t hd = static_cast<std::uint32_t>(head_dim);
    attention_head(sub_view(q8, 0, c0, q8.rows, hd),
                   sub_view(k8, 0, c0, k8.rows, hd),
                   sub_view(v8, 0, c0, v8.rows, hd),
                   sub_view(q_scale, s0, 0, q8.rows, 1),
                   sub_view(k_scale, s0, 0, k8.rows, 1),
                   sub_view(v_scale, s0, 0, v8.rows, 1),
                   sub_view(out, 0, c0, out.rows, hd), causal);
  }
}

} // namespace qwen3_ops

#endif
