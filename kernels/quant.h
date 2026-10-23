#ifndef YAI_QWEN3_OPS_QUANT_H
#define YAI_QWEN3_OPS_QUANT_H

// ============================================================================
// quant_rows：per-token int8 量化（带 per-token f32 scale）
//
//   scale[t] = absmax_c(|x[t][c]|) / 127
//   q[t][c]  = round_rne(x[t][c] / scale[t])        （由 tquantx 一条指令完成）
//
// 形状：x [rows][cols] f32 → qi8 [rows][cols] i8 + scale [rows] f32。
//
// 为什么整行只用一个 scale（而不是每 32 列一个）：host 参考的
// quant(x, q8, s) 就是"整行一个 absmax"，后续 gemm 的 epilogue 也只乘一个
// per-row scale。如果改成 per-32 组，量化值会与参考不同，28 层之后 logits
// 就对不上了（int8 会把差异放大）。
//
// 指令映射：
//   row_absmax  → areduce.rows.absmax（跨块用 vmax 合并）
//   mul(kInv127) / store_vec
//               → vmulx / vstore（写 per-token scale）
//   tquantx     → 用**给定的** scale 量化（quant_ext）
//   store_tile  → tstore
// ============================================================================
#include "tile.h"

#include "config.h"
#include "io.h"
#include "matrix.h"

namespace qwen3_ops {

YAI_OP_INLINE void quant_rows(const MatrixView &x, const MatrixView &qi8,
                                  const MatrixView &scale) {
  // 描述符外提：b 的 stride/dtype、输出 tile 的 dtype、scale 向量的 dtype/stride
  // 都与位置无关 → 只配一次。
  Acc b{};
  config_acc(b, x);
  Tile t{};
  config_tile(t, qi8);      // strides + dtype（store_tile_at 只写 base）
  Vec32 sv{};
  sv.set_dtype(Vec32DType::F32);
  sv.set_stride(scale.row_bytes);

  for (std::uint32_t m = 0; m < x.rows; m += kBlock) {
    const std::uint8_t nr = block_zone(x.rows, m);

    // ---- 第一遍：整行 absmax（跨列块合并）----
    #pragma clang loop YAI_LOOP_UNROLL
    for (std::uint32_t k = 0; k < x.cols; k += kBlock) {
      set_zone(b, nr, block_zone(x.cols, k));
      load_acc_at(b, x.data + m * static_cast<std::uint32_t>(x.row_bytes) +
                     k * static_cast<std::uint32_t>(x.col_bytes));
      if (k == 0) {
        sv = b.row_absmax();      // def 形态：覆盖写 + 配好 vc.len
      } else {
        b.row_absmax(sv);         // merge 形态：vmax
      }
    }
    // ---- scale = absmax/127，写回（只写 nr 个 lane）----
    sv.mul(kInv127);
    set_len(sv, nr);
    store_vec_at(sv, scale.data + m * static_cast<std::uint32_t>(scale.row_bytes));

    // ---- 第二遍：用同一个 scale 逐块量化 ----
    #pragma clang loop YAI_LOOP_UNROLL
    for (std::uint32_t k = 0; k < x.cols; k += kBlock) {
      const std::uint8_t nc = block_zone(x.cols, k);
      set_zone(b, nr, nc);
      load_acc_at(b, x.data + m * static_cast<std::uint32_t>(x.row_bytes) +
                     k * static_cast<std::uint32_t>(x.col_bytes));
      set_zone(t, nr, nc);
      b.quant_ext(sv, t);
      store_tile_at(t, qi8.data + m * static_cast<std::uint32_t>(qi8.row_bytes) +
                       k * static_cast<std::uint32_t>(qi8.col_bytes));
    }
  }
}

} // namespace qwen3_ops

#endif
