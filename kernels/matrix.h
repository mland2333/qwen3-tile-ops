#ifndef YAI_QWEN3_OPS_MATRIX_H
#define YAI_QWEN3_OPS_MATRIX_H

// ============================================================================
// MatrixView：算子之间的唯一"数据接口"
//
// 为什么需要它：Tile/Acc/Vec32 是寄存器值，不能跨函数/跨循环存活（后端禁止
// spill，见 LLVM改造清单 R11）。所以算子之间的边界上只允许出现**纯标量**：
// 一个基址 + 两个字节跨步 + 两个逻辑尺寸 + 一个 dtype 码。算子内部再把这种
// 大矩阵拆成 32x32 的 tile/acc 块（load_acc / load_tile / load_vec …）。
//
// 于是"编排层（main_big.cpp）"可以自由地：
//   * 把 MatrixView 当参数传（普通 C ABI 就行）；
//   * 在运行期循环里反复构造（28 层、head 循环、KV 块循环）；
//   * 用 sub_view 做任意行列切片（切 head、切 KV cache 的第 pos 行、切 rope
//     表的绝对位置行、切 1 行的尾部块……）。
//
// 单位约定：跨步都是**字节**（不是元素），所以 i8/f32 混用时不用改代码；
// rows/cols 是逻辑尺寸，允许不是 32 的倍数（尾块由 block_zone 裁剪）。
// ============================================================================
#include "tile.h"

#include "config.h"

namespace qwen3_ops {

struct MatrixView {
  std::byte *data;         // 第 (0,0) 个元素的地址
  std::int32_t row_bytes;  // 行跨步（字节）：元素 (r+1,c) 相对 (r,c) 的位移
  std::int32_t col_bytes;  // 列跨步（字节）：通常 = sizeof(元素)，即连续的 4/1
  std::uint32_t rows;      // 逻辑行数（可以是 1；可以是 96 这种非 32 倍数）
  std::uint32_t cols;      // 逻辑列数
  std::uint8_t dtype;      // kDt*（i8/i32/f32 …），决定 load 时配什么 dtype
};

// ----------------------------------------------------------------------------
// sub_view(v, r0, c0, rows, cols)：从 (r0,c0) 开始的子视图。
//
// 只做地址偏移，不拷贝数据；跨步和 dtype 继承父视图。三个典型用法：
//   * 切 head：sub_view(q, 0, h*hd, rows, hd)
//   * KV cache 追加：sub_view(kcache, pos, 0, rows, KW)（把本块 token 的 k
//     量化到 cache 的第 pos 行起）
//   * 绝对位置：sub_view(cos_t, pos, 0, rows, half)（rope 表按位置取行）
// 注意 rows/cols 是**新视图**的逻辑尺寸，父视图的 rows 不再参与裁剪。
// ----------------------------------------------------------------------------
YAI_ALWAYS_INLINE MatrixView sub_view(const MatrixView &v, std::uint32_t r0,
                                      std::uint32_t c0, std::uint32_t rows,
                                      std::uint32_t cols) {
  MatrixView s{v.data + r0 * static_cast<std::uint32_t>(v.row_bytes) +
                   c0 * static_cast<std::uint32_t>(v.col_bytes),
               v.row_bytes, v.col_bytes, rows, cols, v.dtype};
  return s;
}

// ----------------------------------------------------------------------------
// block_zone(total, off)：从 off 开始、长度 ≤32 的块有多少个元素。
//
// 它是所有 load_* 的 valid_zone 来源（硬件只会搬运有效区，区外不动）：
//   rows=96, off=32 → 32；rows=96, off=64 → 32；rows=80, off=64 → 16。
// 尾块（不足 32）就是靠这个函数让 32x32 的 tile 只算有效的那部分，
// 所以"行数/列数不是 32 的倍数"是被支持的（实验过 80 = 2×32 + 16）。
// ----------------------------------------------------------------------------
YAI_ALWAYS_INLINE std::uint8_t block_zone(std::uint32_t total,
                                          std::uint32_t off) {
  const std::uint32_t left = total - off;
  return static_cast<std::uint8_t>(left > static_cast<std::uint32_t>(kBlock)
                                       ? static_cast<std::uint32_t>(kBlock)
                                       : left);
}

} // namespace qwen3_ops

#endif
