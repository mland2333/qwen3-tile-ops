#ifndef YAI_QWEN3_OPS_EMBEDDING_H
#define YAI_QWEN3_OPS_EMBEDDING_H

// ============================================================================
// embedding_rows：词表查表 + 反量化（把 token id 变成 f32 激活）
//
//   out[t][c] = table[ids[t]][c] * scale[ids[t]]
//
//   table : [vocab][hidden] i8（PQWF 的 int8 词表）
//   scale : [vocab] f32（每行一个 per-channel scale）
//   ids   : [rows] i32（token id）
//   out   : [rows][hidden] f32
//
// 关键点：id 是**设备值**（来自 blob 的 ids 数组），所以查表不能用标量解引用，
// 而是走硬件的 gather：
//   vgather idv → 用 id 向量去 gather 标量数组（这里 gather scale）
//   tgather idv → 用 id 向量去 gather 矩阵的行（这里 gather 词表的一行块）
// 参考实现（qwen3_by_tile.cpp::embedding）也是这么做的：先 gather scale 到
// out_scale，再逐列块 gather 词表。
//
// 指令映射（一个 32x32 块）：
//   tgather        → tgather tN, vN（按 id 取 32 行）
//   dequant(scale) → tdequant.rows.f32（i8 → f32 并乘 per-token scale）
//                    等价于旧 ISA 的 twiden + acvt + amulv.byrow，但只发一条
//                    指令、只写一次 Acc（见 ISA《量化与反量化》）。
//   store_acc      → astore
// ============================================================================
#include "tile.h"

#include "config.h"
#include "io.h"
#include "matrix.h"

namespace qwen3_ops {

YAI_OP_INLINE void embedding_rows(const MatrixView &table,
                                      const MatrixView &scale,
                                      const MatrixView &ids,
                                      const MatrixView &out) {
  // 描述符外提：gather tile / 输出 acc / 两个向量只依赖视图，配置一次即可；
  // 循环里只写 zone（尾块）和地址。
  Tile t{};
  config_tile(t, table);                       // table 的 row/col stride + dtype(i8)
  Acc af{};                                    // tdequant 的目的，固定 f32
  af.set_strides(out.row_bytes, out.col_bytes); // store 用的步长（out 是 f32 宽）
  af.set_dtype(AccDType::F32);                 // tdequant 要求 dst=f32，全程不用再改
  Vec32 idv{};
  config_vec(idv, ids);
  // 新 ISA 要求 gather 的索引向量 dtype 为 u32（非负），不是 i32。
  idv.set_dtype(Vec32DType::U32);
  Vec32 esv{};
  esv.set_dtype(Vec32DType::F32);
  esv.set_stride(scale.row_bytes);

  for (std::uint32_t m = 0; m < ids.rows; m += kBlock) {
    const std::uint8_t nr = block_zone(ids.rows, m);

    // ---- 1) 取本块 id，并 gather 出每行的 scale ----
    set_len(idv, nr);
    load_vec_at(idv, ids.data + m * static_cast<std::uint32_t>(ids.row_bytes));
    set_len(esv, nr);
    esv.set_data(scale.data);
    esv.gather_from_vec(idv);                  // vgather：esv[i] = scale[idv[i]]

    // ---- 2) 逐列块 gather 词表行 → tdequant（i8 × scale[i] → f32）----
    for (std::uint32_t k = 0; k < table.cols; k += kBlock) {
      const std::uint8_t nc = block_zone(table.cols, k);
      set_zone(t, nr, nc);
      t.set_data(table.data + k * static_cast<std::uint32_t>(table.col_bytes));
      t.gather_from_vec(idv);                  // tgather
      set_zone(af, nr, nc);
      // 一条 tdequant = 旧 ISA 的 twiden.i8.i32 + acvt.i32.f32 + amulv.byrow；
      // 数值逐位相同（value_ops 的 tile dequant case 用 ones 版做了对拍）。
      t.dequant(esv, af);                      // af = f32(t) * esv[i]
      store_acc_at(af, out.data + m * static_cast<std::uint32_t>(out.row_bytes) +
                      k * static_cast<std::uint32_t>(out.col_bytes));
    }
  }
}


// ---------------------------------------------------------------------------
// embedding + rmsnorm 融合。
//
// 未融合时：embedding 写 x（f32），rmsnorm 再**读一遍 x** 求整行 sumsq，然后
// 又读一遍 x 做缩放。融合后第一遍扫描在 gather 的同时就已经把每个 32x32 块的
// f32 值放进了 Acc，于是 sumsq 可以顺手累积（块顺序与 rmsnorm 第一遍完全一致
// → 逐位不变），省掉 rmsnorm 对 tmp 的那一趟读（每块 128 KB）。
//
// 为什么不能连 tmp 也省掉：一行有 cols/32 个块（hidden=1024 → 32 块），
// 而 Acc 只有 16 个，缩放系数又必须等整行 sumsq 算完才知道 → 必须两遍，
// 第一遍的结果只能落回内存。
//
//   table/scale/ids : 词表（i8 + per-token scale）与 token id
//   tmp             : 中间缓冲（embedding 的 f32 结果，同时是残差流的起点）
//   w/out           : rmsnorm 的 weight 与输出
//   inv             : 1/cols
// ---------------------------------------------------------------------------
YAI_OP_INLINE void embedding_rmsnorm_rows(const MatrixView &table,
                                          const MatrixView &scale,
                                          const MatrixView &ids,
                                          const MatrixView &tmp,
                                          const MatrixView &w,
                                          const MatrixView &out, float inv) {
  Tile t{};
  config_tile(t, table);
  Acc af{};
  af.set_strides(tmp.row_bytes, tmp.col_bytes);
  af.set_dtype(AccDType::F32);                 // tdequant 的 dst 固定 f32
  Vec32 idv{};
  config_vec(idv, ids);
  // 新 ISA 要求 gather 的索引向量 dtype 为 u32（非负），不是 i32。
  idv.set_dtype(Vec32DType::U32);
  Vec32 esv{};
  esv.set_dtype(Vec32DType::F32);
  esv.set_stride(scale.row_bytes);
  Vec32 ss{};
  ss.set_dtype(Vec32DType::F32);
  Vec32 wv{};
  config_vec(wv, w);
  Acc b{};
  config_acc(b, tmp);

  for (std::uint32_t m = 0; m < ids.rows; m += kBlock) {
    const std::uint8_t nr = block_zone(ids.rows, m);
    set_len(ss, nr);
    ss.fill(0.0f);

    // ---- id 与 per-token scale ----
    set_len(idv, nr);
    load_vec_at(idv, ids.data + m * static_cast<std::uint32_t>(ids.row_bytes));
    set_len(esv, nr);
    esv.set_data(scale.data);
    esv.gather_from_vec(idv);

    // ---- 第一遍：gather → tdequant（i8×scale→f32）→ 写 tmp + 累积 sumsq ----
    for (std::uint32_t k = 0; k < table.cols; k += kBlock) {
      const std::uint8_t nc = block_zone(table.cols, k);
      set_zone(t, nr, nc);
      t.set_data(table.data + k * static_cast<std::uint32_t>(table.col_bytes));
      t.gather_from_vec(idv);
      set_zone(af, nr, nc);
      // 一条 tdequant 代替 twiden + acvt + amulv.byrow，数值逐位相同。
      t.dequant(esv, af);                      // af = f32(t) * esv[i]
      store_acc_at(af, tmp.data + m * static_cast<std::uint32_t>(tmp.row_bytes) +
                      k * static_cast<std::uint32_t>(tmp.col_bytes));
      // 统一用 merge 形式（ss += 行平方和），ss 已在循环外清零并配好 len。
      af.row_sum_sq(ss);
    }
    ss.mul_(inv);
    ss.add_(kEps);
    ss.rsqrt_();

    // ---- 第二遍：读 tmp → ×ss(byrow) → ×w(bycol) → 写 out ----
    for (std::uint32_t k = 0; k < tmp.cols; k += kBlock) {
      const std::uint8_t nc = block_zone(tmp.cols, k);
      set_zone(b, nr, nc);
      load_acc_at(b, tmp.data + m * static_cast<std::uint32_t>(tmp.row_bytes) +
                     k * static_cast<std::uint32_t>(tmp.col_bytes));
      set_len(wv, nc);
      load_vec_at(wv, w.data + k * static_cast<std::uint32_t>(w.row_bytes));
      b.mul_(ss, true);
      b.mul_(wv, false);
      store_acc_at(b, out.data + m * static_cast<std::uint32_t>(out.row_bytes) +
                      k * static_cast<std::uint32_t>(out.col_bytes));
    }
  }
}

} // namespace qwen3_ops

#endif
