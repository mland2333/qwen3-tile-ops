#include "../kernels/attention_gqa.h"
#include "../kernels/group_gemm.h"
#include "../kernels/lm_argmax.h"
#include "../kernels/norm_rope_quant.h"
#include "../kernels/swiglu_quant.h"
#include "../kernels/embedding.h"
#include "../kernels/qwen3.h"
#include "../kernels/rope_gqa.h"
#include "model_big.h"

// 真实 Qwen3-0.6B 自回归推理：hidden=1024、28 层、16 q 头 / 8 kv 头、hd=128、
// inter=3072、vocab=151936。
//
// 布局与 tools/convert_pqwf.py 一致。控制块 CTRL = [n_prompt, n_gen, max_seq, mode]：
//   mode=0：融合路径——prefill/decode 循环里 lm_head 与 argmax 融合，不写 logits。
//   mode=1：非融合参考路径——同样的循环，但 lm_head 写完整 logits，再用
//           argmax_logits 选下一个 token（用于对拍/调试）。
//   * prefill：prompt 按 kSeq(32) 个 token 一块前向，k/v 量化后写进 KV cache；
//   * decode ：每次 1 个 token（rows=1），attention 走 online-softmax 分块，
//               KV 长度 = pos+1，因果偏移 pos0 = pos（driver 里自动算）；
//   * 生成   ：mode 0 下 lm_head 与 argmax 融合，每个 32 列块直接并进
//               (best, bidx)，不写 logits；mode 1 下写 logits 再 argmax。
//               最后把 id 写回 ids[pos]，下一步用它当输入。
//
// mode 1 的 logits 与 host 参考逐位一致；mode 0 的 fused best/bidx 与参考
// logits 的 max/argmax 逐位一致（make check 对两种模式各跑一遍）。
//
// 视图构造：blob 的每张表都是稠密行主序，所以统一用 matrix.h 的
// f32_view/i8_view/i32_view（自动推 row_bytes/col_bytes），切片用
// rows_of/cols_of；真正非稠密的转置视图（attention 的 Vᵀ）仍手写 MatrixView。
namespace {
constexpr bool kDebug = false;   // 保留给将来 dump（当前构建不含转储代码）
}

namespace qwen3_ops {

// 模型的全部视图（地址/形状/跨步），按块传给前向函数。
struct ModelViews {
  std::byte *blob;
  MatrixView ids, x, emb, emb_s, lm, lm_s, nf, cos_t, sin_t;
  MatrixView h, h8, hs, q, k, v, q8, qs, attn, o, o8, os;
  MatrixView g, u, sw, s8, ss, logits, argf, argi;
};

// 一层的权重/量化 scale/norm 与 KV cache 视图。偏移只在这里算一次，
// 层循环里不再出现 kLK-kLQ 这类减法。
struct LayerViews {
  MatrixView wq, wk, wv, wo, wg, wu, wd;              // [out][in] i8
  MatrixView wqs, wks, wvs, wos, wgs, wus, wds;       // [out][1] f32
  MatrixView n1, nq, nk, n2;                          // [len][1] f32
  MatrixView kcache, kcscale, vcache, vcscale;        // KV cache
};

// 全局缓冲：一张表一行，形状全部来自 model_big.h。
YAI_ALWAYS_INLINE void make_model_views(ModelViews &w, std::byte *blob) {
  constexpr std::uint32_t H = mb::kHidden;
  constexpr std::uint32_t HD = mb::kHeadDim;
  constexpr std::uint32_t QW = mb::kHeads * HD;
  constexpr std::uint32_t KW = mb::kKvHeads * HD;
  constexpr std::uint32_t N = mb::kSeq;
  constexpr std::uint32_t V = mb::kVocab;
  constexpr std::uint32_t I = mb::kInter;
  constexpr std::uint32_t HALF = mb::kHalf;
  constexpr std::uint32_t MS = mb::kMaxSeq;

  w = ModelViews{};
  w.blob  = blob;
  w.ids   = i32_view(blob + mb::kIDS, MS, 1);
  w.x     = f32_view(blob + mb::kX, N, H);
  w.emb   = i8_view(blob + mb::kEMB, V, H);
  w.emb_s = f32_view(blob + mb::kEMB + V * H, V, 1);
  w.lm    = i8_view(blob + mb::kLM, V, H);
  w.lm_s  = f32_view(blob + mb::kLM + V * H, V, 1);
  w.nf    = f32_view(blob + mb::kNF, H, 1);
  w.cos_t = f32_view(blob + mb::kCOS, MS, HALF);
  w.sin_t = f32_view(blob + mb::kSIN, MS, HALF);
  w.h     = f32_view(blob + mb::kH, N, H);
  w.h8    = i8_view(blob + mb::kH8, N, H);
  w.hs    = f32_view(blob + mb::kHS, N, mb::kHidden / 256);
  w.q     = f32_view(blob + mb::kQ, N, QW);
  w.k     = f32_view(blob + mb::kK, N, KW);
  w.v     = f32_view(blob + mb::kV, N, KW);
  w.q8    = i8_view(blob + mb::kQ8, N, QW);
  w.qs    = f32_view(blob + mb::kQS, N, mb::kHeads);
  w.attn  = f32_view(blob + mb::kATTN, N, QW);
  w.o     = f32_view(blob + mb::kO, N, H);
  w.o8    = i8_view(blob + mb::kO8, N, QW);
  w.os    = f32_view(blob + mb::kOS, N, mb::kHeads);
  w.g     = f32_view(blob + mb::kG, N, I);
  w.u     = f32_view(blob + mb::kU, N, I);
  w.sw    = f32_view(blob + mb::kSW, N, I);
  w.s8    = i8_view(blob + mb::kS8, N, I);
  w.ss    = f32_view(blob + mb::kSS, N, I / 256);
  w.logits = f32_view(blob + mb::kLOGITS, N, V);
  w.argf  = f32_view(blob + mb::kARGF, N, 1);
  w.argi  = i32_view(blob + mb::kARGI, N, 1);
}

// 第 L 层的全部视图：weight 段 / norm 段 / KV 段各一个基址。
YAI_ALWAYS_INLINE void make_layer_views(LayerViews &v, std::byte *blob, int L) {
  constexpr std::uint32_t H = mb::kHidden;
  constexpr std::uint32_t HD = mb::kHeadDim;
  constexpr std::uint32_t QW = mb::kHeads * HD;
  constexpr std::uint32_t KW = mb::kKvHeads * HD;
  constexpr std::uint32_t I = mb::kInter;
  constexpr std::uint32_t MS = mb::kMaxSeq;

  std::byte *lw = blob + mb::kLQ + L * mb::kLAYERW_STRIDE;
  std::byte *ln = blob + mb::kN1 + L * mb::kLAYERN_STRIDE;
  std::byte *kv = blob + mb::kKV_BASE + L * mb::kKV_STRIDE;

  v = LayerViews{};
  v.wq = i8_view(lw, QW, H);
  v.wk = i8_view(lw + (mb::kLK - mb::kLQ), KW, H);
  v.wv = i8_view(lw + (mb::kLV - mb::kLQ), KW, H);
  v.wo = i8_view(lw + (mb::kLO - mb::kLQ), H, QW);
  v.wg = i8_view(lw + (mb::kLG - mb::kLQ), I, H);
  v.wu = i8_view(lw + (mb::kLU - mb::kLQ), I, H);
  v.wd = i8_view(lw + (mb::kLD - mb::kLQ), H, I);
  v.wqs = f32_view(lw + (mb::kLQ_SCALE - mb::kLQ), QW, 1);
  v.wks = f32_view(lw + (mb::kLK_SCALE - mb::kLQ), KW, 1);
  v.wvs = f32_view(lw + (mb::kLV_SCALE - mb::kLQ), KW, 1);
  v.wos = f32_view(lw + (mb::kLO_SCALE - mb::kLQ), H, 1);
  v.wgs = f32_view(lw + (mb::kLG_SCALE - mb::kLQ), I, 1);
  v.wus = f32_view(lw + (mb::kLU_SCALE - mb::kLQ), I, 1);
  v.wds = f32_view(lw + (mb::kLD_SCALE - mb::kLQ), H, 1);
  v.n1 = f32_view(ln, H, 1);
  v.nq = f32_view(ln + (mb::kNQ - mb::kN1), HD, 1);
  v.nk = f32_view(ln + (mb::kNK - mb::kN1), HD, 1);
  v.n2 = f32_view(ln + (mb::kN2 - mb::kN1), H, 1);
  // KV cache：[max_seq][KW] i8 + [max_seq][kv_heads] f32 scale（k 段 + v 段）
  v.kcache  = i8_view(kv, MS, KW);
  v.kcscale = f32_view(kv + mb::kKV_SOFF, MS, mb::kKvHeads);
  v.vcache  = i8_view(kv + mb::kKV_VOFF, MS, KW);
  v.vcscale = f32_view(kv + mb::kKV_VOFF + mb::kKV_SOFF, MS, mb::kKvHeads);
}

// lm_head 的输出形式：
//   None   ：不跑 final-norm/lm_head（prefill 的非最后一块）
//   Logits ：写完整 logits（mode 1 非融合参考路径）
//   Argmax ：融合 lm_head+argmax，只把 (best, bidx) 写进 argf/argi（生产路径）
enum class LmHeadOut { None, Logits, Argmax };

// 一块 token（rows <= kSeq）的完整前向。pos 是本块第一个 token 的绝对位置。
YAI_ALWAYS_INLINE void forward_chunk(const ModelViews &w, std::uint32_t pos,
                                     std::uint32_t rows, LmHeadOut lm_out,
                                     bool optimize_tail) {
  constexpr int H = mb::kHidden;
  constexpr int HD = mb::kHeadDim;

  // 本块 token 的切片：激活缓冲只取前 rows 行。
  const MatrixView x = rows_of(w.x, 0, rows);
  const MatrixView h = rows_of(w.h, 0, rows);
  const MatrixView h8 = rows_of(w.h8, 0, rows);
  const MatrixView hs = rows_of(w.hs, 0, rows);
  // rope 表按绝对位置取行
  const MatrixView cos_p = rows_of(w.cos_t, pos, rows);
  const MatrixView sin_p = rows_of(w.sin_t, pos, rows);
  const MatrixView ids_p = rows_of(w.ids, pos, rows);

  // embedding 与「第 0 层的 rmsnorm」融合：第一遍 gather 时顺手累积 sumsq，
  // 省掉 rmsnorm 对 x 的那一趟读（逐位不变，块顺序一致）。
  {
    const MatrixView n1_0 = f32_view(w.blob + mb::kN1, H, 1);
    embedding_rmsnorm_rows(w.emb, w.emb_s, ids_p, x, n1_0, h, 1.0f / H);
  }

  for (int L = 0; L < mb::kLayers; ++L) {
    LayerViews lv;
    make_layer_views(lv, w.blob, L);

    // 第 0 层的 norm 已经在 embedding_rmsnorm_rows 里算完（h 就绪）
    if (L != 0) {
      rmsnorm_rows(x, lv.n1, h, 1.0f / H);
    }
    quant_rows_heads(h, h8, hs, 256);

    // 每一层都必须给所有 rows 算 K/V 并写进 KV cache：后续 token 的
    // attention 要用。K/V 写完之前不能停。
    const MatrixView kk = rows_of(w.k, 0, rows);
    const MatrixView vv = rows_of(w.v, 0, rows);
    MatrixView vc8 = rows_of(lv.vcache, pos, rows);
    MatrixView vcs = rows_of(lv.vcscale, pos, rows);
    gemm_nt_group_store(h8, hs, lv.wv, lv.wvs, vv, 256, nullptr, &vc8, &vcs);

    // K：GEMM 写一次 f32 k，norm/rope/quant 读一次；没有额外存储，
    // 保留 norm/rope 融合，避免中间 f32 往返。
    MatrixView kc = rows_of(lv.kcache, pos, rows);
    MatrixView kcs = rows_of(lv.kcscale, pos, rows);
    gemm_nt_group_store(h8, hs, lv.wk, lv.wks, kk, 256, nullptr, &kc, &kcs,
                        &lv.nk, &cos_p, &sin_p, 1.0f / HD);
    // V 的 GEMM 输出已经在 epilogue 里 per-head 量化并写进 KV cache。

    // 最后一层里，只有需要 logits 的行才需要 Q/attention/O/MLP：
    //   - 非最后一层：所有 rows 都要，作为下一层的输入；
    //   - 最后一层 + 非最终 chunk：一行都不要，K/V 写完就可以停；
    //   - 最后一层 + 最终 chunk：优化路径只算最后一行，参考路径算所有 rows。
    const bool is_last = (L == mb::kLayers - 1);
    const bool do_tail = !is_last || (lm_out != LmHeadOut::None);
    if (!do_tail) {
      continue;
    }
    const bool tail_last_row = optimize_tail && is_last;
    const std::uint32_t tr0 = tail_last_row ? (rows - 1) : 0;
    const std::uint32_t tr = tail_last_row ? 1 : rows;

    MatrixView h8t = rows_of(h8, tr0, tr);
    MatrixView hst = rows_of(hs, tr0, tr);
    MatrixView xt = rows_of(w.x, tr0, tr);
    MatrixView ht = rows_of(w.h, tr0, tr);
    MatrixView qt = rows_of(w.q, tr0, tr);
    MatrixView q8t = rows_of(w.q8, tr0, tr);
    MatrixView qst = rows_of(w.qs, tr0, tr);
    MatrixView o8t = rows_of(w.o8, tr0, tr);
    MatrixView ost = rows_of(w.os, tr0, tr);
    MatrixView ot = rows_of(w.o, tr0, tr);
    MatrixView s8t = rows_of(w.s8, tr0, tr);
    MatrixView sst = rows_of(w.ss, tr0, tr);
    // rope 表按绝对位置取行；tail 可能从 rows-1 开始
    MatrixView cos_t = rows_of(w.cos_t, pos + tr0, tr);
    MatrixView sin_t = rows_of(w.sin_t, pos + tr0, tr);

    // Q：GEMM 写一次 f32 q，norm/rope/quant 读一次；没有额外存储。
    gemm_nt_group_store(h8t, hst, lv.wq, lv.wqs, qt, 256, nullptr, &q8t, &qst,
                        &lv.nq, &cos_t, &sin_t, 1.0f / HD);

    // attention：KV 长度 = pos + rows。
    // 单请求先用 identity page table 走 paged attention：page=32 token，
    // 与当前连续 cache 布局一一对应；多请求时 page table 指向共享 page pool，
    // K/V 写入改为按 request page 直接写或 scatter。
    const std::uint32_t kv_len = pos + rows;
    constexpr std::uint32_t kPageTokens = 32;
    constexpr std::uint32_t kPages = mb::kMaxSeq / kPageTokens;
    unsigned short ptab[kPages];
    for (std::uint32_t p = 0; p < kPages; ++p) ptab[p] = static_cast<unsigned short>(p);
    const std::uint32_t kv_row = static_cast<std::uint32_t>(lv.kcache.row_bytes);
    const std::uint32_t ks_row = static_cast<std::uint32_t>(lv.kcscale.row_bytes);
    PagedKV paged{lv.kcache.data, lv.vcache.data, lv.kcscale.data,
                  lv.vcscale.data, ptab, kPageTokens,
                  kPageTokens * kv_row, kv_row,
                  kPageTokens * ks_row, ks_row};
    attention_heads_gqa_tokscale_paged(
        q8t, qst, paged, kv_len, o8t, ost, mb::kHeads, mb::kKvHeads, HD,
        /*causal=*/true, /*qk_scale=*/kInvSqrt128);

    gemm_nt_group_store(o8t, ost, lv.wo, lv.wos, ot, 128, &xt);

    // SwiGLU
    rmsnorm_rows(xt, lv.n2, ht, 1.0f / H);
    quant_rows_heads(ht, h8t, hst, 256);
    gemm_gate_up_swiglu_quant(h8t, hst, lv.wg, lv.wgs, lv.wu, lv.wus, s8t, sst);
    gemm_nt_group_store(s8t, sst, lv.wd, lv.wds, ot, 256, &xt);
  }

  if (lm_out != LmHeadOut::None) {
    // 只算最后一行的 final-norm + lm_head（省掉整块 32 行）
    const std::uint32_t last = rows - 1;
    MatrixView xl = rows_of(w.x, last, 1);
    MatrixView hl = rows_of(w.o, 0, 1);     // o 已用完，借来放 h
    // q8 是 [N][QW] 的宽缓冲，这里只借前 H 列，不能继承它的 cols
    MatrixView h8l = sub_view(w.q8, 0, 0, 1, H);
    // final norm 也按 128 列一组；hsl 借用 qs 的第 0 行（q 已用完）
    MatrixView hsl{w.qs.data, w.qs.row_bytes, 4, 1, H / 256, kDtF32};
    rmsnorm_rows(xl, w.nf, hl, 1.0f / H);
    quant_rows_heads(hl, h8l, hsl, 256);
    if (lm_out == LmHeadOut::Logits) {
      gemm_nt_group_store(h8l, hsl, w.lm, w.lm_s, rows_of(w.logits, 0, 1), 256);
    } else {
      // 生产路径：per-group lm_head + 每 32 列块就地并进 (best, bidx)
      static_assert(mb::kVocab % 128 == 0 && H % 128 == 0,
                    "gemm_nt_group_argmax needs N and K multiples of 128");
      Vec32 best{};
      best.set_dtype(Vec32DType::F32);
      best.set_stride(4);
      set_len(best, 1);
      best.fill(-3.0e38f);
      Vec32 bidx{};
      bidx.set_dtype(Vec32DType::I32);
      bidx.set_stride(4);
      set_len(bidx, 1);
      bidx.fill(0.0f);
      gemm_nt_group_argmax(h8l, hsl, w.lm, w.lm_s, best, bidx, 256);
      store_vec(best, w.argf, 0);
      store_vec(bidx, w.argi, 0);
    }
  }
}

// 贪心 argmax（对 logits 第 0 行）。跨 32 列块时把 f32 位模式映射成单调整型
// key 再比较：-march=rv64im 没有 F 扩展，标量浮点比较会拉 libgcc 软浮点 helper。
YAI_ALWAYS_INLINE std::uint32_t argmax_logits(const ModelViews &w) {
  const MatrixView logits1 = rows_of(w.logits, 0, 1);
  std::uint32_t best = 0;
  std::uint32_t best_key = 0;
  bool first = true;
  for (std::uint32_t n = 0; n < mb::kVocab; n += kBlock) {
    Acc a{};
    load_acc(a, logits1, 0, n);
    Vec32 mx = a.row_max();      // f32，长度 = ac.rows = 1
    Vec32 ix = a.row_argmax();   // i32，长度 = 1（块内下标）
    store_vec(mx, w.argf, 0);
    store_vec(ix, w.argi, 0);
    const std::uint32_t bits = *reinterpret_cast<const std::uint32_t *>(w.argf.data);
    const std::uint32_t idx = *reinterpret_cast<const std::uint32_t *>(w.argi.data);
    const std::uint32_t key = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    if (first || key > best_key) {
      best_key = key;
      best = n + idx;
      first = false;
    }
  }
  return best;
}

// 生产路径的 argmax：gemm_nt_argmax 已经把 (best, bidx) 写回 argf/argi，
// 这里只读一次 bidx，不再扫 logits。
YAI_ALWAYS_INLINE std::uint32_t fused_argmax_id(const ModelViews &w) {
  return *reinterpret_cast<const std::uint32_t *>(w.argi.data);
}

} // namespace qwen3_ops

int main() {
  std::uintptr_t base;
  asm volatile("mv %0, x31" : "=r"(base));
  auto *blob = reinterpret_cast<std::byte *>(base);
  using namespace qwen3_ops;
  constexpr int N = mb::kSeq;

  ModelViews w;
  make_model_views(w, blob);

  auto *ctrl = reinterpret_cast<volatile std::int32_t *>(blob + mb::kCTRL);
  const std::uint32_t n_prompt = static_cast<std::uint32_t>(ctrl[0]);
  const std::uint32_t n_gen = static_cast<std::uint32_t>(ctrl[1]);
  const std::uint32_t mode = static_cast<std::uint32_t>(ctrl[3]);
  if (mode > 1) {                        // 只保留 0=融合、1=非融合
    return 2;
  }
  const bool fused = (mode == 0);

  const std::uint32_t total = n_prompt + n_gen;

  std::uint32_t pos = 0;
  while (pos < total) {
    const std::uint32_t rows =
        (pos < n_prompt) ? ((n_prompt - pos < N) ? (n_prompt - pos) : N) : 1;
    const bool need_out = (pos + rows >= n_prompt);
    const LmHeadOut lm_out =
        need_out ? (fused ? LmHeadOut::Argmax : LmHeadOut::Logits)
                 : LmHeadOut::None;
    forward_chunk(w, pos, rows, lm_out, fused);
    pos += rows;
    if (pos >= n_prompt && pos < total) {
      const std::uint32_t id = fused ? fused_argmax_id(w) : argmax_logits(w);
      *reinterpret_cast<volatile std::uint32_t *>(w.ids.data + pos * 4) = id;
    }
  }
  return 0;
}
