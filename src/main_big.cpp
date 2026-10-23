#include "../kernels/attention_gqa.h"
#include "../kernels/embedding.h"
#include "../kernels/qwen3.h"
#include "../kernels/rope_gqa.h"
#include "model_big.h"

// 真实 Qwen3-0.6B 自回归推理：hidden=1024、28 层、16 q 头 / 8 kv 头、hd=128、
// inter=3072、vocab=151936。
//
// 布局与 tools/convert_pqwf.py 一致。控制块 CTRL = [n_prompt, n_gen, max_seq, mode]：
//   mode=0：内置生成循环（prefill 分块 + 贪心 argmax，一次跑完 n_gen 个 token）
//   mode=1：单步模式——只处理绝对位置 CTRL[0] 的 1 个 token（ids[CTRL[0]]），
//           写 logits 后退出。KV cache 在 blob 里，host 可以读 logits 自己采样，
//           再把采样结果写回 ids、以本 blob 为输入跑下一步。
//   * prefill：prompt 按 kSeq(32) 个 token 一块前向，k/v 量化后写进 KV cache；
//   * decode ：每次 1 个 token（rows=1），attention 走 online-softmax 分块，
//               KV 长度 = pos+1，因果偏移 pos0 = pos（driver 里自动算）；
//   * 生成   ：每个位置前向结束后取最后一行 logits 做 argmax（贪心），
//               把 id 写回 ids[pos]，下一步用它当输入。
//
// logits 与 host 参考 qwen3_by_tile.cpp 逐位一致（max|Δ|=0，见 tools/known_issues.md）。
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

// 一块 token（rows <= kSeq）的完整前向。pos 是本块第一个 token 的绝对位置。
YAI_ALWAYS_INLINE void forward_chunk(const ModelViews &w, std::uint32_t pos,
                                     std::uint32_t rows, bool want_logits) {
  constexpr int H = mb::kHidden;
  constexpr int HD = mb::kHeadDim;
  constexpr int QW = mb::kHeads * HD;
  constexpr int KW = mb::kKvHeads * HD;

  const MatrixView x = sub_view(w.x, 0, 0, rows, H);
  const MatrixView h = sub_view(w.h, 0, 0, rows, H);
  const MatrixView h8 = sub_view(w.h8, 0, 0, rows, H);
  const MatrixView hs = sub_view(w.hs, 0, 0, rows, 1);
  const MatrixView q = sub_view(w.q, 0, 0, rows, QW);
  const MatrixView q8 = sub_view(w.q8, 0, 0, rows, QW);
  const MatrixView qs = sub_view(w.qs, 0, 0, rows, 1);
  const MatrixView attn = sub_view(w.attn, 0, 0, rows, QW);
  const MatrixView o8 = sub_view(w.o8, 0, 0, rows, QW);
  const MatrixView os = sub_view(w.os, 0, 0, rows, 1);
  const MatrixView o = sub_view(w.o, 0, 0, rows, H);
  const MatrixView g = sub_view(w.g, 0, 0, rows, mb::kInter);
  const MatrixView u = sub_view(w.u, 0, 0, rows, mb::kInter);
  const MatrixView sw = sub_view(w.sw, 0, 0, rows, mb::kInter);
  const MatrixView s8 = sub_view(w.s8, 0, 0, rows, mb::kInter);
  const MatrixView ss = sub_view(w.ss, 0, 0, rows, 1);
  // rope 表按绝对位置取行
  const MatrixView cos_p = sub_view(w.cos_t, pos, 0, rows, mb::kHalf);
  const MatrixView sin_p = sub_view(w.sin_t, pos, 0, rows, mb::kHalf);
  const MatrixView ids_p = sub_view(w.ids, pos, 0, rows, 1);

  // embedding 与「第 0 层的 rmsnorm」融合：第一遍 gather 时顺手累积 sumsq，
  // 省掉 rmsnorm 对 x 的那一趟读（逐位不变，块顺序一致）。
  {
    const MatrixView n1_0{w.blob + mb::kN1, 4, 4, H, 1, kDtF32};
    embedding_rmsnorm_rows(w.emb, w.emb_s, ids_p, x, n1_0, h, 1.0f / H);
  }

  for (int L = 0; L < mb::kLayers; ++L) {
    std::byte *lw = w.blob + mb::kLQ + L * mb::kLAYERW_STRIDE;
    std::byte *ln = w.blob + mb::kN1 + L * mb::kLAYERN_STRIDE;
    const MatrixView wq{lw, H, 1, QW, H, kDtI8};
    const MatrixView wk{lw + (mb::kLK - mb::kLQ), H, 1, KW, H, kDtI8};
    const MatrixView wv{lw + (mb::kLV - mb::kLQ), H, 1, KW, H, kDtI8};
    const MatrixView wo{lw + (mb::kLO - mb::kLQ), QW, 1, H, QW, kDtI8};
    const MatrixView wg{lw + (mb::kLG - mb::kLQ), H, 1, mb::kInter, H, kDtI8};
    const MatrixView wu{lw + (mb::kLU - mb::kLQ), H, 1, mb::kInter, H, kDtI8};
    const MatrixView wd{lw + (mb::kLD - mb::kLQ), mb::kInter, 1, H, mb::kInter, kDtI8};
    const MatrixView wqs{lw + (mb::kLQ_SCALE - mb::kLQ), 4, 4, QW, 1, kDtF32};
    const MatrixView wks{lw + (mb::kLK_SCALE - mb::kLQ), 4, 4, KW, 1, kDtF32};
    const MatrixView wvs{lw + (mb::kLV_SCALE - mb::kLQ), 4, 4, KW, 1, kDtF32};
    const MatrixView wos{lw + (mb::kLO_SCALE - mb::kLQ), 4, 4, H, 1, kDtF32};
    const MatrixView wgs{lw + (mb::kLG_SCALE - mb::kLQ), 4, 4, mb::kInter, 1, kDtF32};
    const MatrixView wus{lw + (mb::kLU_SCALE - mb::kLQ), 4, 4, mb::kInter, 1, kDtF32};
    const MatrixView wds{lw + (mb::kLD_SCALE - mb::kLQ), 4, 4, H, 1, kDtF32};
    const MatrixView n1{ln, 4, 4, H, 1, kDtF32};
    const MatrixView nq{ln + (mb::kNQ - mb::kN1), 4, 4, HD, 1, kDtF32};
    const MatrixView nk{ln + (mb::kNK - mb::kN1), 4, 4, HD, 1, kDtF32};
    const MatrixView n2{ln + (mb::kN2 - mb::kN1), 4, 4, H, 1, kDtF32};
    // 本层 KV cache：[max_seq][KW] i8 + [max_seq] f32 scale（k 段 + v 段）
    std::byte *kv = w.blob + mb::kKV_BASE + L * mb::kKV_STRIDE;
    const MatrixView kcache{kv, KW, 1, mb::kMaxSeq, KW, kDtI8};
    const MatrixView kcscale{kv + mb::kKV_SOFF, 4, 4, mb::kMaxSeq, 1, kDtF32};
    const MatrixView vcache{kv + mb::kKV_VOFF, KW, 1, mb::kMaxSeq, KW, kDtI8};
    const MatrixView vcscale{kv + mb::kKV_VOFF + mb::kKV_SOFF, 4, 4, mb::kMaxSeq, 1, kDtF32};

    // 第 0 层的 norm 已经在 embedding_rmsnorm_rows 里算完（h 就绪）
    if (L != 0) {
      rmsnorm_rows(x, n1, h, 1.0f / H);
    }
    quant_rows(h, h8, hs);
    gemm_nt(h8, wq, wqs, hs, q);
    MatrixView kk = sub_view(w.k, 0, 0, rows, KW);
    MatrixView vv = sub_view(w.v, 0, 0, rows, KW);
    gemm_nt(h8, wk, wks, hs, kk);
    gemm_nt(h8, wv, wvs, hs, vv);

    // k/v：按 kv 头做 q/k-norm + rope
    for (int kvm = 0; kvm < mb::kKvHeads; ++kvm) {
      const std::uint32_t c0 = static_cast<std::uint32_t>(kvm * HD);
      MatrixView kh = sub_view(kk, 0, c0, rows, HD);
      rmsnorm_rows(kh, nk, kh, 1.0f / HD);
      rope_head_hd(kh, cos_p, sin_p);
    }
    // k/v 都是 per-token 量化（整行一个 scale，和 host 参考的 quant(k)/quant(v)
    // 粒度一致），直接写进 KV cache 的 [pos, pos+rows) 行。
    quant_rows(kk, sub_view(kcache, pos, 0, rows, KW),
               sub_view(kcscale, pos, 0, rows, 1));
    quant_rows(vv, sub_view(vcache, pos, 0, rows, KW),
               sub_view(vcscale, pos, 0, rows, 1));

    // q：按 q 头 q/k-norm + rope，再 per-token 量化
    for (int hh = 0; hh < mb::kHeads; ++hh) {
      const std::uint32_t c0 = static_cast<std::uint32_t>(hh * HD);
      MatrixView qh = sub_view(q, 0, c0, rows, HD);
      rmsnorm_rows(qh, nq, qh, 1.0f / HD);
      rope_head_hd(qh, cos_p, sin_p);
    }
    quant_rows(q, q8, qs);

    // attention：KV 长度 = pos + rows
    const std::uint32_t kv_len = pos + rows;
    attention_heads_gqa_tokscale(
        q8, sub_view(kcache, 0, 0, kv_len, KW), sub_view(vcache, 0, 0, kv_len, KW),
        qs, kcscale, vcscale, attn, mb::kHeads, mb::kKvHeads, HD,
        /*causal=*/true, /*qk_scale=*/kInvSqrt128);

    quant_rows(attn, o8, os);
    gemm_nt(o8, wo, wos, os, o);
    add_rows(x, o);

    // SwiGLU
    rmsnorm_rows(x, n2, h, 1.0f / H);
    quant_rows(h, h8, hs);
    gemm_nt(h8, wg, wgs, hs, g);
    gemm_nt(h8, wu, wus, hs, u);
    swiglu(g, u, sw);
    quant_rows(sw, s8, ss);
    gemm_nt(s8, wd, wds, ss, o);
    add_rows(x, o);
  }

  if (want_logits) {
    // 只算最后一行的 final-norm + lm_head（省掉整块 32 行）
    const std::uint32_t last = rows - 1;
    MatrixView xl = sub_view(w.x, last, 0, 1, H);
    MatrixView hl = sub_view(w.o, 0, 0, 1, H);     // o 已用完，借来放 h
    MatrixView h8l = sub_view(w.q8, 0, 0, 1, H);   // q8 已用完
    MatrixView hsl = sub_view(w.qs, 0, 0, 1, 1);   // qs 已用完
    rmsnorm_rows(xl, w.nf, hl, 1.0f / H);
    quant_rows(hl, h8l, hsl);
    gemm_nt(h8l, w.lm, w.lm_s, hsl, sub_view(w.logits, 0, 0, 1, mb::kVocab));
  }
}

// 贪心 argmax（对 logits 第 0 行）。跨 32 列块时把 f32 位模式映射成单调整型
// key 再比较：-march=rv64im 没有 F 扩展，标量浮点比较会拉 libgcc 软浮点 helper。
YAI_ALWAYS_INLINE std::uint32_t argmax_logits(const ModelViews &w) {
  const MatrixView logits1 = sub_view(w.logits, 0, 0, 1, mb::kVocab);
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

} // namespace qwen3_ops

int main() {
  std::uintptr_t base;
  asm volatile("mv %0, x31" : "=r"(base));
  auto *blob = reinterpret_cast<std::byte *>(base);
  using namespace qwen3_ops;
  constexpr int H = mb::kHidden;
  constexpr int QW = mb::kHeads * mb::kHeadDim;
  constexpr int KW = mb::kKvHeads * mb::kHeadDim;
  constexpr int N = mb::kSeq;

  ModelViews w{};
  w.blob = blob;
  w.ids = MatrixView{blob + mb::kIDS, 4, 4, mb::kMaxSeq, 1, kDtI32};
  w.x = MatrixView{blob + mb::kX, H * 4, 4, N, H, kDtF32};
  w.emb = MatrixView{blob + mb::kEMB, H, 1, mb::kVocab, H, kDtI8};
  w.emb_s = MatrixView{blob + mb::kEMB + mb::kVocab * H, 4, 4, mb::kVocab, 1, kDtF32};
  w.lm = MatrixView{blob + mb::kLM, H, 1, mb::kVocab, H, kDtI8};
  w.lm_s = MatrixView{blob + mb::kLM + mb::kVocab * H, 4, 4, mb::kVocab, 1, kDtF32};
  w.nf = MatrixView{blob + mb::kNF, 4, 4, H, 1, kDtF32};
  w.cos_t = MatrixView{blob + mb::kCOS, mb::kHalf * 4, 4, mb::kMaxSeq, mb::kHalf, kDtF32};
  w.sin_t = MatrixView{blob + mb::kSIN, mb::kHalf * 4, 4, mb::kMaxSeq, mb::kHalf, kDtF32};
  w.h = MatrixView{blob + mb::kH, H * 4, 4, N, H, kDtF32};
  w.h8 = MatrixView{blob + mb::kH8, H, 1, N, H, kDtI8};
  w.hs = MatrixView{blob + mb::kHS, 4, 4, N, 1, kDtF32};
  w.q = MatrixView{blob + mb::kQ, QW * 4, 4, N, QW, kDtF32};
  w.k = MatrixView{blob + mb::kK, KW * 4, 4, N, KW, kDtF32};
  w.v = MatrixView{blob + mb::kV, KW * 4, 4, N, KW, kDtF32};
  w.q8 = MatrixView{blob + mb::kQ8, QW, 1, N, QW, kDtI8};
  w.qs = MatrixView{blob + mb::kQS, 4, 4, N, 1, kDtF32};
  w.attn = MatrixView{blob + mb::kATTN, QW * 4, 4, N, QW, kDtF32};
  w.o = MatrixView{blob + mb::kO, H * 4, 4, N, H, kDtF32};
  w.o8 = MatrixView{blob + mb::kO8, QW, 1, N, QW, kDtI8};
  w.os = MatrixView{blob + mb::kOS, 4, 4, N, 1, kDtF32};
  w.g = MatrixView{blob + mb::kG, mb::kInter * 4, 4, N, mb::kInter, kDtF32};
  w.u = MatrixView{blob + mb::kU, mb::kInter * 4, 4, N, mb::kInter, kDtF32};
  w.sw = MatrixView{blob + mb::kSW, mb::kInter * 4, 4, N, mb::kInter, kDtF32};
  w.s8 = MatrixView{blob + mb::kS8, mb::kInter, 1, N, mb::kInter, kDtI8};
  w.ss = MatrixView{blob + mb::kSS, 4, 4, N, 1, kDtF32};
  w.logits = MatrixView{blob + mb::kLOGITS, mb::kVocab * 4, 4, N, mb::kVocab, kDtF32};
  w.argf = MatrixView{blob + mb::kARGF, 4, 4, N, 1, kDtF32};
  w.argi = MatrixView{blob + mb::kARGI, 4, 4, N, 1, kDtI32};

  auto *ctrl = reinterpret_cast<volatile std::int32_t *>(blob + mb::kCTRL);
  const std::uint32_t n_prompt = static_cast<std::uint32_t>(ctrl[0]);
  const std::uint32_t n_gen = static_cast<std::uint32_t>(ctrl[1]);
  const std::uint32_t mode = static_cast<std::uint32_t>(ctrl[3]);
  if (mode == 1) {                       // 单步：host 侧采样用
    forward_chunk(w, n_prompt, 1, /*want_logits=*/true);
    return 0;
  }
  const std::uint32_t total = n_prompt + n_gen;

  std::uint32_t pos = 0;
  while (pos < total) {
    const std::uint32_t rows =
        (pos < n_prompt) ? ((n_prompt - pos < N) ? (n_prompt - pos) : N) : 1;
    const bool want_logits = (pos + rows >= n_prompt);
    forward_chunk(w, pos, rows, want_logits);
    pos += rows;
    if (pos >= n_prompt && pos < total) {
      const std::uint32_t id = argmax_logits(w);
      *reinterpret_cast<volatile std::uint32_t *>(w.ids.data + pos * 4) = id;
    }
  }
  return 0;
}
