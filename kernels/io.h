#ifndef YAI_QWEN3_OPS_IO_H
#define YAI_QWEN3_OPS_IO_H

// ============================================================================
// MatrixView <-> tile 值的唯一通道
//
// 硬件搬运（tload/tstore/aload/astore/vload/vstore/tgather/vgather）不是"读一个
// 指针"，而是读**配置寄存器**：amN/tmN/vmN 里的基址+跨步，acN/tcN/vcN 里的
// rows/cols/dtype。配置寄存器按数据寄存器的物理编号索引，所以"值"= 数据寄存器
// 的内容，而"形状/地址"= 另一套旁路状态。
//
// 由此得到本文件的核心约定（写新算子必须遵守）：
//   ① load_* 会把**4 个字段全写一遍**（base / row_stride / col_stride / zone+dtype），
//      所以从任意视图 load 都是自洽的；
//   ② store_* 只写 base/row_stride（**故意不重写 zone/dtype**）：它沿用刚刚
//      load 到同一个物理寄存器上的 valid_zone，这样"读-改-写"一个块时形状
//      不会丢。前提是块的生命周期内那个物理寄存器没被别人重配（由寄存器
//      分配保证，因为值一直活着）；
//   ③ 用 `X x{}` 新建的 Vec32/Acc **没有任何配置**（vc.len=0 / ac.rows=0），
//      必须先 init_* 或 load 才能用，否则模拟器报 invalid_configuration 或者
//      按陈旧的 len 静默算错；
//   ④ 需要"长度不是 32"的向量（decode 只有 1 行、尾块只有 16 行）时必须用
//      load_vec_n，因为 byrow/bycol 广播要求 vec.len 恰好等于矩阵的 rows/cols。
// ============================================================================
#include "tile.h"

#include "config.h"
#include "matrix.h"

namespace qwen3_ops {

// ---------------------------------------------------------------------------
// MatrixView::dtype（kDt*）-> 硬件 dtype 枚举
//   Acc  : 只有 i32 和 f32 两种语义（u32 当 i32 用）
//   Tile : i8/u8（int8 矩阵）
//   Vec32: i32/u32/f32（argmax 下标用 i32，scale 用 f32）
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE AccDType to_acc_dtype(std::uint8_t d) {
  return d == kDtI32 ? AccDType::I32 : AccDType::F32;
}
YAI_ALWAYS_INLINE TileDType to_tile_dtype(std::uint8_t d) {
  return d == kDtU8 ? TileDType::U8 : TileDType::I8;
}
YAI_ALWAYS_INLINE Vec32DType to_vec_dtype(std::uint8_t d) {
  return d == kDtI32 ? Vec32DType::I32
                     : (d == kDtU32 ? Vec32DType::U32 : Vec32DType::F32);
}

// ---------------------------------------------------------------------------
// set_acc：只配置、不搬运（形状由调用方给定，可以 <32）。
//
// 用途是"我要往这个位置写一个新算出来的块"——比如 embedding 直接把 gather+
// widen 的结果写成 32x32，或者 quant 里给 tquantx 的输出 tile 配形状。
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE void set_acc(Acc &a, const MatrixView &v, std::uint32_t r0,
                               std::uint32_t c0, std::uint32_t rows,
                               std::uint32_t cols) {
  a.set_data(v.data + r0 * static_cast<std::uint32_t>(v.row_bytes) +
             c0 * static_cast<std::uint32_t>(v.col_bytes));
  a.set_strides(v.row_bytes, v.col_bytes);
  a.set_valid_zone(static_cast<std::uint8_t>(rows),
                   static_cast<std::uint8_t>(cols));
  a.set_dtype(to_acc_dtype(v.dtype));
}

// ---------------------------------------------------------------------------
// load_acc：把 v 从 (r0,c0) 起的 32x32（尾块自动裁剪）搬进 Acc。
//
// 对应 6 条指令：cfg.setx amN,{base,row_stride,col_stride} + cfg.setx acN,
// {rows,cols,dtype} + aload aN。zone 由 block_zone 裁剪到剩余形状。
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE void load_acc(Acc &a, const MatrixView &v, std::uint32_t r0,
                                std::uint32_t c0) {
  const std::uint8_t rows = block_zone(v.rows, r0);
  const std::uint8_t cols = block_zone(v.cols, c0);
  a.set_data(v.data + r0 * static_cast<std::uint32_t>(v.row_bytes) +
             c0 * static_cast<std::uint32_t>(v.col_bytes));
  a.set_strides(v.row_bytes, v.col_bytes);
  a.set_valid_zone(rows, cols);
  a.set_dtype(to_acc_dtype(v.dtype));
  a.load();
}

// ---------------------------------------------------------------------------
// store_acc：把 Acc 写回 v 的 (r0,c0)。
//
// 只重写 base/row_stride（见文件头约定②）：zone/dtype 沿用同一寄存器上一次
// load（或 set_acc/init_*）的设置。所以典型写法是
//     load_acc(b, x, m, k);  ...改 b...;  store_acc(b, out, m, k);
// 读和写用同一个 zone。
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE void store_acc(Acc &a, const MatrixView &v, std::uint32_t r0,
                                 std::uint32_t c0) {
  a.set_data(v.data + r0 * static_cast<std::uint32_t>(v.row_bytes) +
             c0 * static_cast<std::uint32_t>(v.col_bytes));
  a.set_strides(v.row_bytes, v.col_bytes);
  a.store();
}

// ---------------------------------------------------------------------------
// load_acc_h / store_acc_h：列数由调用方指定的版本。
//
// rope 用它在"半区"里按 32 列推进：`load_acc_h(co, cos_t, m, off, n)` 表示
// 取 m 行起、off 列起、**n 列**（不是 block_zone 自动裁剪的 32 列），
// 这样 half=64 的最后一块仍然可以按 32 对齐推进。
// 行方向仍然按 block_zone(v.rows, r0) 裁剪。
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE void load_acc_h(Acc &a, const MatrixView &v, std::uint32_t r0,
                                  std::uint32_t c0, std::uint8_t cols) {
  a.set_data(v.data + r0 * static_cast<std::uint32_t>(v.row_bytes) +
             c0 * static_cast<std::uint32_t>(v.col_bytes));
  a.set_strides(v.row_bytes, v.col_bytes);
  a.set_valid_zone(block_zone(v.rows, r0), cols);
  a.set_dtype(to_acc_dtype(v.dtype));
  a.load();
}

YAI_ALWAYS_INLINE void store_acc_h(Acc &a, const MatrixView &v,
                                   std::uint32_t r0, std::uint32_t c0) {
  a.set_data(v.data + r0 * static_cast<std::uint32_t>(v.row_bytes) +
             c0 * static_cast<std::uint32_t>(v.col_bytes));
  a.set_strides(v.row_bytes, v.col_bytes);
  a.store();
}

// ---------------------------------------------------------------------------
// load_tile / store_tile：i8 的 32x32 搬运（tload/tstore）。
//
// GEMM 的 A/B 操作数、量化输出、P·V 的 V 都用它。zone 同样按剩余形状裁剪，
// 所以 K 方向最后一块不足 32 时 mma 只算有效列。
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE void load_tile(Tile &t, const MatrixView &v, std::uint32_t r0,
                                 std::uint32_t c0) {
  t.set_data(v.data + r0 * static_cast<std::uint32_t>(v.row_bytes) +
             c0 * static_cast<std::uint32_t>(v.col_bytes));
  t.set_strides(v.row_bytes, v.col_bytes);
  t.set_dtype(to_tile_dtype(v.dtype));
  t.set_valid_zone(block_zone(v.rows, r0), block_zone(v.cols, c0));
  t.load();
}

YAI_ALWAYS_INLINE void store_tile(Tile &t, const MatrixView &v,
                                  std::uint32_t r0, std::uint32_t c0) {
  t.set_data(v.data + r0 * static_cast<std::uint32_t>(v.row_bytes) +
             c0 * static_cast<std::uint32_t>(v.col_bytes));
  t.set_strides(v.row_bytes, v.col_bytes);
  t.store();
}

// ---------------------------------------------------------------------------
// load_vec：从第 index 个元素起载入 **32** 个 f32/i32（per-token scale 之类）。
//
// 只用于"长度确实是 32"的场景（一块 token 的 scale）。stride 取 row_bytes，
// 所以 scale 张量可以按 [32][1] 的跨步排布（每 4 字节一个）。
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE void load_vec(Vec32 &x, const MatrixView &v,
                                std::uint32_t index) {
  x.set_dtype(to_vec_dtype(v.dtype));
  x.set_valid_zone(static_cast<std::uint8_t>(kBlock));
  x.set_data(v.data + index * static_cast<std::uint32_t>(v.row_bytes));
  x.set_stride(v.row_bytes);
  x.load();
}

// ---------------------------------------------------------------------------
// store_vec：把 Vec32 当前有效长度的那些 lane 写回（长度沿用 x 的 vc.len）。
//
// 所以 quant_rows 里 `store_vec(sv, scale, m)` 会写 sv.len 个 scale（可能是
// 16 个，如果尾块只有 16 行）——这正是我们要的"只写本块的行"。
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE void store_vec(Vec32 &x, const MatrixView &v,
                                 std::uint32_t index) {
  x.set_data(v.data + index * static_cast<std::uint32_t>(v.row_bytes));
  x.set_stride(v.row_bytes);
  x.store();
}

// ---------------------------------------------------------------------------
// load_vec_n：带显式长度的载入。
//
// 必需场景：
//   * decode 时矩阵只有 1 行 → byrow 广播要求 vec.len == 1（用 load_vec 会
//     得到 32，直接 invalid_configuration）；
//   * 行/列块是尾块（例如 80 行模型的最后 16 行）。
// 这是把"长度不是 32 的倍数"跑通的关键函数。
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE void load_vec_n(Vec32 &x, const MatrixView &v,
                                  std::uint32_t index, std::uint8_t len) {
  x.set_dtype(to_vec_dtype(v.dtype));
  x.set_valid_zone(static_cast<std::int32_t>(len));
  x.set_data(v.data + index * static_cast<std::uint32_t>(v.row_bytes));
  x.set_stride(v.row_bytes);
  x.load();
}

// ===========================================================================
// "配置一次 + 只动地址"形态（性能关键）
//
// 描述符有 5 个字段，但只有 base_addr 依赖块位置：
//   tm/am : base_addr（变）  row_stride_bytes / col_stride_bytes（只依赖 MatrixView）
//   tc/ac : dtype（只依赖 MatrixView）  rows / cols（只依赖块形状）
// 所以内核应该把 stride/dtype 提到最外层、zone 提到"形状不变的那一层"，内层每次
// 只写 base_addr：
//
//   config_tile(ta, a);            // 循环外一次：stride + dtype
//   set_zone(ta, rows, kBlock);    // (m,n) 层：rows/cols
//   for (k ...) load_tile_at(ta, a.data + m*a.row_bytes + k*a.col_bytes);
//                                 //        每块只有 1 条 cfg.setx + tload
//
// 实测：gemm_nt 这样改之后，0.6B prefill 的指令数 12.39M → 6.68M（-46%），
// logits 逐位不变。为什么不能指望编译器做：这些 cfg 指令都带 hasSideEffects=1
// （cfg 是全局状态），LLVM 的 CSE/LICM 不敢动它们。
//
// 安全前提：被提升的 tile/acc/vec 值必须**跨整个循环存活**，这样寄存器分配
// 不会把它和别的值分到同一个物理寄存器，配置也就不会被别人重写。
// 反例：把 zone 提到 m/n 循环外会让 lm_head 的 1 行 A 与 dst zone 不匹配
// （运行期 invalid_configuration）——zone 只能提到"形状确实不变"的那一层。
// ===========================================================================
YAI_ALWAYS_INLINE void config_tile(Tile &t, const MatrixView &v) {
  t.set_strides(v.row_bytes, v.col_bytes);
  t.set_dtype(to_tile_dtype(v.dtype));
}
YAI_ALWAYS_INLINE void set_zone(Tile &t, std::uint8_t rows, std::uint8_t cols) {
  t.set_valid_zone(rows, cols);
}
YAI_ALWAYS_INLINE void load_tile_at(Tile &t, std::byte *addr) {
  t.set_data(addr);
  t.load();
}
YAI_ALWAYS_INLINE void store_tile_at(Tile &t, std::byte *addr) {
  t.set_data(addr);
  t.store();
}

YAI_ALWAYS_INLINE void config_acc(Acc &a, const MatrixView &v) {
  a.set_strides(v.row_bytes, v.col_bytes);
  a.set_dtype(to_acc_dtype(v.dtype));
}
YAI_ALWAYS_INLINE void set_zone(Acc &a, std::uint8_t rows, std::uint8_t cols) {
  a.set_valid_zone(rows, cols);
}
YAI_ALWAYS_INLINE void load_acc_at(Acc &a, std::byte *addr) {
  a.set_data(addr);
  a.load();
}
YAI_ALWAYS_INLINE void store_acc_at(Acc &a, std::byte *addr) {
  a.set_data(addr);
  a.store();
}

YAI_ALWAYS_INLINE void config_vec(Vec32 &x, const MatrixView &v) {
  x.set_dtype(to_vec_dtype(v.dtype));
  x.set_stride(v.row_bytes);
}
YAI_ALWAYS_INLINE void set_len(Vec32 &x, std::uint8_t len) {
  x.set_valid_zone(static_cast<std::int32_t>(len));
}
YAI_ALWAYS_INLINE void load_vec_at(Vec32 &x, std::byte *addr) {
  x.set_data(addr);
  x.load();
}
YAI_ALWAYS_INLINE void store_vec_at(Vec32 &x, std::byte *addr) {
  x.set_data(addr);
  x.store();
}

// ---------------------------------------------------------------------------
// 新建值的配置初始化
//
// 这三/四个函数解决"新值没有旁路配置"的问题（见文件头约定③）：
//   init_f32_vec      ：给 running max / denominator 这种纯计算的 Vec32 配 dtype+len
//   init_f32_acc_zero ：给输出累加器配 f32 + zone 并清零（online softmax 的 O）
//   init_i32_acc      ：给 mma 的 int32 目的 acc 配 dtype+zone（不搬运）
//   init_i8_tile      ：32x32 i8 tile 的默认形状
// 少了任何一步，轻则 invalid_configuration，重则按上一轮残留的 shape 算。
// ---------------------------------------------------------------------------
YAI_ALWAYS_INLINE void init_f32_vec(Vec32 &x, std::uint8_t len) {
  x.set_dtype(Vec32DType::F32);
  x.set_valid_zone(static_cast<std::int32_t>(len));
}

YAI_ALWAYS_INLINE void init_f32_acc_zero(Acc &a, std::uint8_t rows,
                                         std::uint8_t cols) {
  a.set_dtype(AccDType::F32);
  a.set_valid_zone(rows, cols);
  a.fill(0.0f);
}

YAI_ALWAYS_INLINE void init_i8_tile(Tile &t) {
  t.set_dtype(TileDType::I8);
  t.set_valid_zone(static_cast<std::uint8_t>(kBlock),
                   static_cast<std::uint8_t>(kBlock));
}

YAI_ALWAYS_INLINE void init_i32_acc(Acc &a, std::uint8_t rows,
                                    std::uint8_t cols) {
  a.set_dtype(AccDType::I32);
  a.set_valid_zone(rows, cols);
}

} // namespace qwen3_ops

#endif
