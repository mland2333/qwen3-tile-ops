#ifndef YAI_QWEN3_OPS_ROPE_GQA_H
#define YAI_QWEN3_OPS_ROPE_GQA_H

// ============================================================================
// rope_head_hd：任意 head_dim 的 RoPE（rotate-half），按 32 列 / 32 行分块
//
//   x      : [rows][head_dim] f32（单个 head 的切片，调用方用 sub_view 切好）
//   cos_t  : [rows][half] f32    —— 已经按**绝对位置**切好行（cos_t 的行就是
//                                   token 的位置；decode 时调用方传 pos 那一行）
//   sin_t  : 同上
//   half   = head_dim/2（Qwen3-0.6B 是 64，所以每个 head 有 2 个半区）
//
//   out[j]      = x[j]*cos[j] - x[half+j]*sin[j]
//   out[half+j] = x[j]*sin[j] + x[half+j]*cos[j]
//
// 三层结构：
//   m 行块（token 分批，rows 可以 >32 或者 =1）
//     off 列块（在"半区"的宽度上按 32 推进；head_dim=128 时 half=64 → 两块）
//
// 与 rope.h 的区别：那个是 hd=32 的单块特化（half=16 装得下一个 tile），
// 这个要处理 half 不是 32 的倍数/大于 32 的情况，所以列方向要分块。
//
// 原地安全：同一列块内，先算完 left/right 再写回；两个半区（off 与 half+off）
// 是互不重叠的列段，跨块之间也不会互相覆盖。
// ============================================================================
#include "tile.h"

#include "config.h"
#include "io.h"
#include "matrix.h"

namespace qwen3_ops {

YAI_OP_INLINE void rope_head_hd(const MatrixView &x, const MatrixView &cos_t,
                                    const MatrixView &sin_t) {
  const std::uint32_t half = x.cols / 2;
  const std::uint32_t hb = static_cast<std::uint32_t>(x.row_bytes);
  const std::uint32_t cb = static_cast<std::uint32_t>(x.col_bytes);

  // 描述符外提：6 个 acc 的步长/dtype 只依赖三个视图，配置一次即可。
  Acc co{}; Acc si{}; Acc l{}; Acc t{}; Acc r{}; Acc t2{};
  config_acc(co, cos_t);
  config_acc(si, sin_t);
  config_acc(l, x);
  config_acc(t, x);
  config_acc(r, x);
  config_acc(t2, x);

  for (std::uint32_t m = 0; m < x.rows; m += kBlock) {
    const std::uint8_t nr = block_zone(x.rows, m);
    for (std::uint32_t off = 0; off < half; off += kBlock) {
      const std::uint32_t left = half - off;
      const std::uint32_t len = left > static_cast<std::uint32_t>(kBlock)
                                    ? static_cast<std::uint32_t>(kBlock)
                                    : left;
      const std::uint8_t n = static_cast<std::uint8_t>(len);
      std::byte *p0 = x.data + m * hb + off * cb;
      std::byte *p1 = x.data + m * hb + (half + off) * cb;

      set_zone(co, nr, n);
      load_acc_at(co, cos_t.data + m * static_cast<std::uint32_t>(cos_t.row_bytes) +
                          off * static_cast<std::uint32_t>(cos_t.col_bytes));
      set_zone(si, nr, n);
      load_acc_at(si, sin_t.data + m * static_cast<std::uint32_t>(sin_t.row_bytes) +
                          off * static_cast<std::uint32_t>(sin_t.col_bytes));

      set_zone(l, nr, n);
      load_acc_at(l, p0);
      l.mul(co);
      set_zone(t, nr, n);
      load_acc_at(t, p1);
      t.mul(si);
      l.sub(t);

      set_zone(r, nr, n);
      load_acc_at(r, p0);
      r.mul(si);
      set_zone(t2, nr, n);
      load_acc_at(t2, p1);
      t2.mul(co);
      r.add(t2);

      store_acc_at(l, p0);
      store_acc_at(r, p1);
    }
  }
}

} // namespace qwen3_ops

#endif
