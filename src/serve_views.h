// Shared model views for the single-request driver and the serve guests.
// Layout constants come from model_big.h; every view is dense row-major.
#pragma once

#include <cstddef>
#include <cstdint>

#include "../kernels/matrix.h"
#include "model_big.h"

namespace qwen3_ops {

// 模型的全部视图（地址/形状/跨步），按块传给前向函数。
struct ModelViews {
  std::byte *blob;
  MatrixView ids, x, emb, emb_s, lm, lm_s, nf, cos_t, sin_t;
  MatrixView h, h8, hs, q, k, v, q8, qs, attn, o, o8, os;
  MatrixView k8, v8, ks, vs;   // contiguous K/V staging for batch scatter
  MatrixView g, u, sw, s8, ss, logits, argf, argi;
};

// 一层的权重/量化 scale/norm 与 KV cache 视图。
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
  w.k8    = i8_view(blob + mb::kK8, N, KW);
  w.v8    = i8_view(blob + mb::kV8, N, KW);
  w.ks    = f32_view(blob + mb::kKS, N, mb::kKvHeads);
  w.vs    = f32_view(blob + mb::kVS, N, mb::kKvHeads);
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

} // namespace qwen3_ops
