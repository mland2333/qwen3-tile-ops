#ifndef YAI_QWEN3_OPS_MLP_H
#define YAI_QWEN3_OPS_MLP_H

// ============================================================================
// SwiGLU 与残差加法
//
//   swiglu   : out[t][j] = silu(gate[t][j]) * up[t][j]
//   add_rows : dst[t][c] += src[t][c]
//
// 这两块都是"逐元素、无跨行依赖"，所以按 32x32 块独立处理即可；行块循环
// 只是为了让 rows > 32 时也能全覆盖（不是乘法意义上的分块）。
// ============================================================================
#include "tile.h"

#include "config.h"
#include "io.h"
#include "matrix.h"

namespace qwen3_ops {

// ---------------------------------------------------------------------------
// swiglu：out = silu(gate) * up
//
// silu(x) = x / (1 + e^-x)。这里手写成 host 参考（qwen3_by_tile.cpp）里的
// f32 式子：  x / (1 + exp2(-x * log2e))
// 而不是直接调 `Acc::silu()`（ISA 的 asilu）。原因：Tile-NPU-Sim 把 asilu
// 建模成"double 里算 x/(1+exp(-x))"，与参考的 float 式子差 ~1 ulp；int8
// 量化会把 1 ulp 放大成 LSB 翻转，28 层后 logits 就不可比了。
//
// 手写版本的 4 步与参考逐位对应：
//   mul(-1)          → -x（乘 -1 在 f32 里精确）
//   exp()            → amulx(log2e) + aexp2 = exp2(-x*log2e) = e^-x
//   add(1.0f)        → 1 + e^-x
//   div(e)           → x / (1 + e^-x)  = silu(x)
//
// 为什么 gate 读两次：Acc 是寄存器引用，没有"免费的 acc 拷贝"，重新 load
// 一次（一个 aload）比想办法复制更简单也更便宜。
// ---------------------------------------------------------------------------
YAI_OP_INLINE void swiglu(const MatrixView &gate, const MatrixView &up,
                              const MatrixView &out) {
  // 描述符外提：三个 acc 的 stride/dtype 只依赖视图，配置一次即可；
  // 每块只写 zone（尾块）和 base。
  Acc g{};
  Acc e{};
  Acc u{};
  config_acc(g, gate);
  config_acc(e, gate);
  config_acc(u, up);

  for (std::uint32_t m = 0; m < gate.rows; m += kBlock) {
    for (std::uint32_t n = 0; n < gate.cols; n += kBlock) {
      const std::uint8_t nr = block_zone(gate.rows, m);
      const std::uint8_t nc = block_zone(gate.cols, n);
      const std::uint32_t off_g =
          m * static_cast<std::uint32_t>(gate.row_bytes) +
          n * static_cast<std::uint32_t>(gate.col_bytes);
      const std::uint32_t off_u =
          m * static_cast<std::uint32_t>(up.row_bytes) +
          n * static_cast<std::uint32_t>(up.col_bytes);
      set_zone(g, nr, nc);
      load_acc_at(g, gate.data + off_g);
      set_zone(e, nr, nc);
      load_acc_at(e, gate.data + off_g);
      set_zone(u, nr, nc);
      load_acc_at(u, up.data + off_u);
      e.mul(-1.0f);
      e.exp();
      e.add(1.0f);
      g.div(e);
      g.mul(u);
      store_acc_at(g, out.data + m * static_cast<std::uint32_t>(out.row_bytes) +
                      n * static_cast<std::uint32_t>(out.col_bytes));
    }
  }
}

YAI_OP_INLINE void add_rows(const MatrixView &dst, const MatrixView &src) {
  Acc a{};
  Acc b{};
  config_acc(a, dst);
  config_acc(b, src);
  for (std::uint32_t m = 0; m < dst.rows; m += kBlock) {
    for (std::uint32_t n = 0; n < dst.cols; n += kBlock) {
      const std::uint8_t nr = block_zone(dst.rows, m);
      const std::uint8_t nc = block_zone(dst.cols, n);
      const std::uint32_t off_d =
          m * static_cast<std::uint32_t>(dst.row_bytes) +
          n * static_cast<std::uint32_t>(dst.col_bytes);
      set_zone(a, nr, nc);
      load_acc_at(a, dst.data + off_d);
      set_zone(b, nr, nc);
      load_acc_at(b, src.data + m * static_cast<std::uint32_t>(src.row_bytes) +
                      n * static_cast<std::uint32_t>(src.col_bytes));
      a.add(b);
      store_acc_at(a, dst.data + off_d);
    }
  }
}

} // namespace qwen3_ops

#endif
