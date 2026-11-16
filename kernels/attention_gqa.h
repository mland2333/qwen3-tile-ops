#ifndef YAI_QWEN3_OPS_ATTENTION_GQA_H
#define YAI_QWEN3_OPS_ATTENTION_GQA_H

// ============================================================================
// 泛化 attention（GQA + 任意 head_dim + KV 分块）
//
// 三个层次：
//   1) attention_head_hd          单 (q头, kv头)，KV 只有一块（≤32）
//   2) attention_heads_gqa        按 head 切片的 driver（head-major scale）
//   3) attention_head_hd128_blocks + attention_heads_gqa_tokscale
//                                 KV 分块 + online softmax（0.6B 用这个）
//
// 布局（调用方用 sub_view 切好的单头视图）：
//   q8 : [q_rows][head_dim] i8    k8/v8 : [kv_len][head_dim] i8
//   scale: [*] f32                out   : [q_rows][head_dim] f32
//
// 数学：O = softmax(Q·Kᵀ * qk_scale + causal_mask) · V。Q/K/V 都是量化值，
// 所以 scale 按 "S *= q_scale(byrow) * k_scale(bycol)" 还原，V 的 scale 折进
// softmax 权重（见 attention.h 的说明）。
// ============================================================================
#include "tile.h"

#include "config.h"
#include "io.h"
#include "matrix.h"

namespace qwen3_ops {

// ---------------------------------------------------------------------------
// attention_head_hd：单 KV 块版（kv_len ≤ 32），head_dim 可以是 32 的倍数。
//   K 方向：S = Σ_d q8_d · k8_dᵀ（int32 累加，首块 zero）；
//   V 方向：O 按 32 列输出块，每块一条 mma.nn。
// ---------------------------------------------------------------------------
YAI_OP_INLINE void attention_head_hd(const MatrixView &q8,
                                         const MatrixView &k8,
                                         const MatrixView &v8,
                                         const MatrixView &q_scale,
                                         const MatrixView &k_scale,
                                         const MatrixView &v_scale,
                                         const MatrixView &out, bool causal,
                                         float qk_scale) {
  const std::uint32_t rows = block_zone(q8.rows, 0);
  const std::uint32_t kdim = q8.cols;

  // ---- 1) S = Σ_{d} q8_d · k8_dᵀ ----
  Acc s{};
  init_i32_acc(s, rows, block_zone(k8.rows, 0));
  {
    Tile qt{};
    Tile kt{};
    config_tile(qt, q8);
    config_tile(kt, k8);
    set_zone(qt, static_cast<std::uint8_t>(rows), static_cast<std::uint8_t>(kBlock));
    set_zone(kt, block_zone(k8.rows, 0), static_cast<std::uint8_t>(kBlock));
    load_tile_at(qt, q8.data);
    load_tile_at(kt, k8.data);
    qt.matmul(kt, s, /*zero=*/true, /*lt=*/false, /*rt=*/true);
    for (std::uint32_t k = kBlock; k < kdim; k += kBlock) {
      load_tile_at(qt, q8.data + k * static_cast<std::uint32_t>(q8.col_bytes));
      load_tile_at(kt, k8.data + k * static_cast<std::uint32_t>(k8.col_bytes));
      qt.matmul(kt, s, /*zero=*/false, /*lt=*/false, /*rt=*/true);
    }
  }

  // ---- 2) 还原 Q/K 的量化 scale，再乘 1/sqrt(head_dim) ----
  // 新 ISA 无 acvt：一条 adequant 做 f32(s) * ksv[j] * qsv[i]（再乘 qk_scale）。
  Vec32 qsv{};
  Vec32 ksv{};
  Vec32 vsv{};
  load_vec(qsv, q_scale, 0);
  load_vec(ksv, k_scale, 0);
  load_vec(vsv, v_scale, 0);
  Vec32 ones{};
  config_ones(ones);            // O 的 epilogue 还要用（单 scale + den）
  Acc sf{};
  init_f32_acc(sf, static_cast<std::uint8_t>(rows),
               static_cast<std::uint8_t>(block_zone(k8.rows, 0)));
  // 一条 adequant：f32(s) * ksv[j] * qsv[i]（acvt + 两次广播乘已合并）
  adequant(sf, s, ksv, qsv);
  sf.mul_(qk_scale);

  // ---- 3) 因果 mask（tril；单块时偏移恒为 0）----
  if (causal) {
    sf.mask_(0);
  }

  // ---- 4) softmax ----
  Vec32 m = sf.row_max();
  sf.sub_(m, true);
  sf.exp_();
  Vec32 den = sf.row_sum();
  // 参考把 1/den 留到最后；这里先不除，只折 V scale（否则多一次舍入、差 1 ulp）
  sf.mul_(vsv, false);

  // ---- 5) P = 量化(S) ----
  Vec32 ps = sf.row_absmax();
  ps.mul_(kInv127);
  Tile pt{};
  pt.set_dtype(TileDType::I8);
  pt.set_valid_zone(static_cast<std::uint8_t>(rows),
                    static_cast<std::uint8_t>(kBlock));
  sf.quant_ext(ps, pt);

  // ---- 6) O = P·Vᵀ，每 32 维输出块一条 mma.nt ----
  // 右操作数必须是 Vᵀ（[hd][kv]）：用转置视图（row_bytes=1、col_bytes=原行跨步）
  // 按块载入。den 的倒数在循环外算一次（循环内会把它再倒一次）。
  const std::uint32_t kv = v8.rows;
  const MatrixView vT{v8.data, 1, v8.row_bytes,
                      static_cast<std::uint32_t>(v8.cols), kv, v8.dtype};
  const std::uint8_t nc = block_zone(kv, 0);
  den.rcp_();
  for (std::uint32_t n = 0; n < v8.cols; n += kBlock) {
    Tile vt{};
    config_tile(vt, vT);
    set_zone(vt, block_zone(vT.rows, n), nc);
    load_tile_at(vt, vT.data + n);
    Acc o{};
    init_i32_acc(o, static_cast<std::uint8_t>(rows), block_zone(v8.cols, n));
    // acvt + byrow(ps) + byrow(den) → adequant(ones, ps) + byrow(den)
    const std::uint8_t ncols = block_zone(v8.cols, n);
    Acc of{};
    init_f32_acc(of, static_cast<std::uint8_t>(rows), ncols);
    set_len(ones, ncols);
    pt.matmul_dequant(vt, o, of, ones, ps, /*zero=*/true, /*lt=*/false,
                      /*rt=*/true);
    of.mul_(den, true);
    store_acc(of, out, 0, n);
  }
}

// ---------------------------------------------------------------------------
// attention_heads_gqa：GQA driver（head-major scale）
//   group = heads / kv_heads；q 头 h 对应 kv 头 h / group。
//   scale 布局 head-major：q_scale[h*32 + token]、k/v_scale[kv*32 + token]
//   （tiny/small 模型的量化粒度：每个头单独算 absmax）。
// ---------------------------------------------------------------------------
YAI_OP_INLINE void attention_heads_gqa(const MatrixView &q8,
                                           const MatrixView &k8,
                                           const MatrixView &v8,
                                           const MatrixView &q_scale,
                                           const MatrixView &k_scale,
                                           const MatrixView &v_scale,
                                           const MatrixView &out, int heads,
                                           int kv_heads, int head_dim,
                                           bool causal, float qk_scale) {
  const int group = heads / kv_heads;
  for (int h = 0; h < heads; ++h) {
    const int kv = h / group;
    const std::uint32_t qc = static_cast<std::uint32_t>(h * head_dim);
    const std::uint32_t kvc = static_cast<std::uint32_t>(kv * head_dim);
    const std::uint32_t qs0 = static_cast<std::uint32_t>(h * kBlock);
    const std::uint32_t ks0 = static_cast<std::uint32_t>(kv * kBlock);
    const std::uint32_t hd = static_cast<std::uint32_t>(head_dim);
    attention_head_hd(sub_view(q8, 0, qc, q8.rows, hd),
                      sub_view(k8, 0, kvc, k8.rows, hd),
                      sub_view(v8, 0, kvc, v8.rows, hd),
                      sub_view(q_scale, qs0, 0, q8.rows, 1),
                      sub_view(k_scale, ks0, 0, k8.rows, 1),
                      sub_view(v_scale, ks0, 0, v8.rows, 1),
                      sub_view(out, 0, qc, out.rows, hd), causal, qk_scale);
  }
}

// ---------------------------------------------------------------------------
// 下面两个是 0.6B 实际用的版本：scale 是 per-token（整行一个，所有头共用），
// KV 按 32 行分块 + online softmax，q 也按 32 行分块。
//
// 每个 (head, q 块) 维护 4 个状态：maximum / den / r0..r3。
// 每个 KV 块 j：
//   1. S_j = q8·k8ᵀ（4 个 K 块，int32）→ f32 → ×q scale → ×k scale → ×qk_scale
//   2. mask(pos0 - j)
//   3. 本块行 max → nmm = max(row_max, maximum)
//   4. corr = exp(maximum - nmm)，修正 den 与 r0..r3
//   5. P = exp(S - nmm)，den += Σ P
//   6. 折 V scale → 量化 P → 4 个输出块各做一次 P·V
//
// KV 只有一块时 corr = exp(-FLT_MAX - m) = 0，res/den 初值为 0，0*0 + x == x，
// 所以结果与单块实现逐位一致。
// ---------------------------------------------------------------------------

// 一个 32 宽输出块：res += (P·V_d) * pmax
YAI_ALWAYS_INLINE void pv_block(Acc &res, Tile &pt, Tile &vt, Acc &part,
                                const MatrixView &v8, std::uint32_t j,
                                std::uint32_t d, const Vec32 &pmax,
                                Vec32 &ones, std::uint8_t rows) {
  // 新 ISA 只保留 mma.nt（D[m,n] = Σ_k A[m,k]·B[n,k]），PV 需要 B = Vᵀ：
  // 调用方传进来的 v8 已经是转置视图（row_bytes=1、col_bytes=原行跨步），
  // 所以这里按 (d, j) 取块，块内就是 Vᵀ。
  load_tile_at(vt, v8.data + d * static_cast<std::uint32_t>(v8.row_bytes) +
                       j * static_cast<std::uint32_t>(v8.col_bytes));
  part.set_dtype(AccDType::I32);   // 上一轮计算会把它改成 f32
  // 新 ISA 的 drain 序列：mma 后紧跟 adequant，由 Tile API 的
  // matmul_dequant 生成一个 pseudo，post-RA 展开成相邻两条。
  Acc pf{};
  init_f32_acc(pf, rows, static_cast<std::uint8_t>(kBlock));
  set_len(ones, static_cast<std::uint8_t>(kBlock));
  pt.matmul_dequant(vt, part, pf, ones, pmax, /*zero=*/true, /*lt=*/false,
                    /*rt=*/true);
  res.add_(pf);                    // 累加
}

// KV 分块版单头 attention（head_dim 固定 128 = 4 个输出块）
YAI_OP_INLINE void attention_head_hd128_blocks(
    const MatrixView &q8, const MatrixView &k8, const MatrixView &v8,
    const MatrixView &q_scale, const MatrixView &k_scale,
    const MatrixView &v_scale, const MatrixView &o8, const MatrixView &os,
    std::uint32_t q_rows, std::uint32_t kv_len, std::int32_t pos0,
    bool causal, float qk_scale) {
  const std::uint8_t nr = static_cast<std::uint8_t>(q_rows);
  constexpr std::uint8_t ZB = static_cast<std::uint8_t>(kBlock);

  // ---- 0) 状态初始化 ----
  Vec32 qsv{};
  load_vec_n(qsv, q_scale, 0, nr);
  Vec32 maximum{};
  init_f32_vec(maximum, nr);
  maximum.fill(-3.402823466e+38F);      // -FLT_MAX
  Vec32 den{};
  init_f32_vec(den, nr);
  den.fill(0.0f);
  Acc r0{}, r1{}, r2{}, r3{};           // 4 个独立变量（不能放数组）
  init_f32_acc_zero(r0, nr, ZB);
  init_f32_acc_zero(r1, nr, ZB);
  init_f32_acc_zero(r2, nr, ZB);
  init_f32_acc_zero(r3, nr, ZB);

  // ---- 1) 逐 KV 块 ----
  Tile qt{};
  config_tile(qt, q8);
  set_zone(qt, nr, ZB);
  Tile kt{};
  config_tile(kt, k8);
  Tile vt{};
  config_tile(vt, v8);
  Vec32 ksv{};
  config_vec(ksv, k_scale);
  Vec32 vsv{};
  config_vec(vsv, v_scale);
  Acc part{};
  part.set_strides(q8.row_bytes, q8.col_bytes);
  set_zone(part, nr, ZB);
  Acc s{};
  // epilogue 的 "不缩放" 列向量（新 ISA 无 acvt，adequant 需要 col scale）
  Vec32 ones{};
  config_ones(ones);
  Tile pt{};
  pt.set_dtype(TileDType::I8);

  // 因果时整块被 mask 的 KV 块直接跳过（贡献恰好为 0，逐位不变）：
  // 本 q 块能看到的最大 KV 位置是 pos0 + q_rows - 1，所以 j >= pos0+q_rows 全被 mask。
  std::uint32_t kv_end = kv_len;
  if (causal) {
    const std::uint32_t need = static_cast<std::uint32_t>(pos0) + q_rows;
    if (need < kv_end) kv_end = need;
  }
  for (std::uint32_t j = 0; j < kv_end; j += kBlock) {
    const std::uint8_t nc = block_zone(kv_len, j);   // 尾块可能是 16 / 1

    // 1a) S = q8 · k8ᵀ（4 个 K 块在 i32 里累加）
    s.set_dtype(AccDType::I32);
    set_zone(s, nr, nc);
    set_zone(kt, nc, ZB);
    #pragma clang loop YAI_LOOP_UNROLL
    for (std::uint32_t d = 0; d < static_cast<std::uint32_t>(kHeadDim128); d += kBlock) {
      load_tile_at(qt, q8.data + d * static_cast<std::uint32_t>(q8.col_bytes));
      load_tile_at(kt, k8.data + j * static_cast<std::uint32_t>(k8.row_bytes) +
                           d * static_cast<std::uint32_t>(k8.col_bytes));
      qt.matmul(kt, s, /*zero=*/d == 0, /*lt=*/false, /*rt=*/true);
    }

    // 1b) 还原量化 scale + qk_scale：一条 adequant 做 f32(s) * ksv[j] * qsv[i]
    set_len(ksv, nc);
    load_vec_at(ksv, k_scale.data + j * static_cast<std::uint32_t>(k_scale.row_bytes));
    Acc sf{};
    init_f32_acc(sf, nr, nc);
    adequant(sf, s, ksv, qsv);
    sf.mul_(qk_scale);

    // 1c) 因果 mask：偏移 pos0 - j
    if (causal) {
      sf.mask_(pos0 - static_cast<std::int32_t>(j));
    }

    // 1d) running max + correction
    //
    // 写法说明：corr 是**派生值**（本来就需要一份副本），所以第一步用函数式
    // `maximum.sub(nmm)` 直接表达意图（内部 1 次 vcopy），后续再原地 exp_。
    // 不要写成 `maximum.sub(nmm).exp()`：链条的每一步都会各拷一次（现在是 2 次）。
    // 等后端给运算加上 def 之后，链条才会变成 0 拷贝。
    Vec32 nmm = sf.row_max();
    nmm.max_(maximum);
    // corr 是派生值，直接用值形式：maximum - nmm 写到新寄存器（post-RA 只补
    // 一条 cfg.copy，不搬 128 B 数据）；随后在 corr 上原地 exp。
    Vec32 corr = maximum.sub(nmm);
    corr.exp_();
    den.mul_(corr);
    r0.mul_(corr, true);
    r1.mul_(corr, true);
    r2.mul_(corr, true);
    r3.mul_(corr, true);
    sf.sub_(nmm, true);
    sf.exp_();
    sf.row_sum(den);                // 分母累加必须在乘 V scale 之前
    maximum.copy(nmm);   // 禁止隐式赋值（会变别名）

    // 1e) 折 V 的 scale → 量化 P
    set_len(vsv, nc);
    load_vec_at(vsv, v_scale.data + j * static_cast<std::uint32_t>(v_scale.row_bytes));
    sf.mul_(vsv, false);
    Vec32 pmax = sf.row_absmax();
    pmax.mul_(kInv127);
    set_zone(pt, nr, nc);
    sf.quant_ext(pmax, pt);

    // 1f) 4 个输出块累加
    set_zone(vt, ZB, nc);   // 转置视图：行=head_dim 块、列=KV 块
    pv_block(r0, pt, vt, part, v8, j, 0 * kBlock, pmax, ones, nr);
    pv_block(r1, pt, vt, part, v8, j, 1 * kBlock, pmax, ones, nr);
    pv_block(r2, pt, vt, part, v8, j, 2 * kBlock, pmax, ones, nr);
    pv_block(r3, pt, vt, part, v8, j, 3 * kBlock, pmax, ones, nr);
  }

  // ---- 2) 最后才除分母 ----
  // ISA 没有 div：用 rcp.approx（相对误差 ≤ 2^-10）再按行乘。
  den.rcp_();
  r0.mul_(den, true);
  r1.mul_(den, true);
  r2.mul_(den, true);
  r3.mul_(den, true);

  // epilogue 直接量化：per-head absmax，写 i8 tile + 每行一个 head scale，
  // 不再把 f32 attention 输出写回内存再 quant_rows_heads。
  Vec32 m0 = r0.row_absmax();
  Vec32 m1 = r1.row_absmax();
  Vec32 m2 = r2.row_absmax();
  Vec32 m3 = r3.row_absmax();
  Vec32 m = m0.max(m1).max(m2.max(m3));
  m.mul_(kInv127);
  m.set_stride(os.row_bytes);
  Tile ot{};
  config_tile(ot, o8);
  set_zone(ot, nr, ZB);
  r0.quant_ext(m, ot);
  store_tile_at(ot, o8.data + 0 * ZB * static_cast<std::uint32_t>(o8.col_bytes));
  r1.quant_ext(m, ot);
  store_tile_at(ot, o8.data + 1 * ZB * static_cast<std::uint32_t>(o8.col_bytes));
  r2.quant_ext(m, ot);
  store_tile_at(ot, o8.data + 2 * ZB * static_cast<std::uint32_t>(o8.col_bytes));
  r3.quant_ext(m, ot);
  store_tile_at(ot, o8.data + 3 * ZB * static_cast<std::uint32_t>(o8.col_bytes));
  store_vec_at(m, os.data);
}

// ---------------------------------------------------------------------------
// GQA driver（per-token scale + KV 分块，0.6B 用）
//   q8: [q_rows][heads*hd]，k8/v8: [kv_len][kv_heads*hd]（通常就是 KV cache 前 kv_len 行）
//   scale: per-token（整行一个，所有头共用）
// 因果偏移 pos0 = kv_len - q_rows，对 prefill 和 decode 都成立。
// ---------------------------------------------------------------------------
YAI_OP_INLINE void attention_heads_gqa_tokscale(const MatrixView &q8,
                                                   const MatrixView &k8,
                                                   const MatrixView &v8,
                                                   const MatrixView &q_scale,
                                                   const MatrixView &k_scale,
                                                   const MatrixView &v_scale,
                                                   const MatrixView &o8,
                                                   const MatrixView &os, int heads,
                                                   int kv_heads, int head_dim,
                                                   bool causal, float qk_scale) {
  const int group = heads / kv_heads;
  const std::uint32_t q_rows = q8.rows;
  const std::uint32_t kv_len = k8.rows;
  const std::int32_t pos0 =
      causal ? static_cast<std::int32_t>(kv_len) - static_cast<std::int32_t>(q_rows) : 0;
  for (int h = 0; h < heads; ++h) {
    const int kv = h / group;
    const std::uint32_t qc = static_cast<std::uint32_t>(h * head_dim);
    const std::uint32_t kvc = static_cast<std::uint32_t>(kv * head_dim);
    const std::uint32_t hd = static_cast<std::uint32_t>(head_dim);
    for (std::uint32_t qi = 0; qi < q_rows; qi += kBlock) {
      const std::uint8_t nr = block_zone(q_rows, qi);
      const MatrixView qs_h = sub_view(q_scale, qi, h, nr, 1);
      const MatrixView ks_h = sub_view(k_scale, 0, kv, kv_len, 1);
      const MatrixView vs_h = sub_view(v_scale, 0, kv, kv_len, 1);
      attention_head_hd128_blocks(
          sub_view(q8, qi, qc, nr, hd), sub_view(k8, 0, kvc, kv_len, hd),
          MatrixView{v8.data + static_cast<std::size_t>(kvc) * static_cast<std::size_t>(v8.col_bytes),
                     1, v8.row_bytes, hd, kv_len, v8.dtype},
          qs_h, ks_h, vs_h,
          sub_view(o8, qi, qc, nr, hd), sub_view(os, qi, h, nr, 1),
          nr, kv_len,
          causal ? pos0 + static_cast<std::int32_t>(qi) : 0, causal, qk_scale);
    }
  }
}

} // namespace qwen3_ops

#endif
