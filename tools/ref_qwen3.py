#!/usr/bin/env python3
"""Reference implementation of the Qwen3 tile kernels (numpy, f32 semantics).

The reference mirrors ``kernels/`` instruction by instruction, including the
f32 rounding points: it is meant to explain device numbers, not to be an
independent implementation of Qwen3.  Every helper below corresponds to one
kernel, and the f32 operations are applied in the same order as the ISA
instructions the kernel emits (``amulx``/``aexp2``/``vrsqrt``/``areduce`` ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

F = np.float32
LOG2E = np.uint32(0x3FB8AA3B).view(F)          # 1.4426950216293335f
KINV127 = (F(1.0) / F(127.0)).astype(F)
EPS = F(1e-6)
FLT_MAX = F(3.402823466e38)


# ---------------------------------------------------------------------------
# f32 primitives (same rounding points as the instructions)
# ---------------------------------------------------------------------------
def exp_f32(x: np.ndarray) -> np.ndarray:
    """``exp`` lowered as ``amulx(log2e)`` + ``aexp2``."""
    return np.exp2((x.astype(F) * LOG2E).astype(F)).astype(F)


def rcp_f32(x: np.ndarray) -> np.ndarray:
    """``rcp.approx``; the simulator models it as an IEEE 1/x."""
    return (F(1.0) / x.astype(F)).astype(F)


def rsqrt_f32(x: np.ndarray) -> np.ndarray:
    return (F(1.0) / np.sqrt(x.astype(F))).astype(F)


def tree_reduce_rows(matrix: np.ndarray, op) -> np.ndarray:
    """The ISA reduction tree over the column axis, one row at a time."""
    nodes = np.asarray(matrix, dtype=F)
    while nodes.shape[1] > 1:
        n = nodes.shape[1]
        nxt = op(nodes[:, 0:n - 1:2], nodes[:, 1:n:2])
        if n % 2:
            nxt = np.concatenate([nxt, nodes[:, -1:]], axis=1)
        nodes = nxt.astype(F)
    return nodes[:, 0].astype(F)


def tree_sum_rows(matrix: np.ndarray) -> np.ndarray:
    return tree_reduce_rows(matrix, lambda a, b: (a + b).astype(F))


def tree_max_rows(matrix: np.ndarray) -> np.ndarray:
    return tree_reduce_rows(matrix, lambda a, b: np.where(b > a, b, a).astype(F))


def row_absmax(matrix: np.ndarray) -> np.ndarray:
    """``areduce.rows.absmax``: |x| leaves (NaN -> +0), then the max tree."""
    return tree_max_rows(np.abs(matrix).astype(F)).astype(F)


# ---------------------------------------------------------------------------
# quantisation
# ---------------------------------------------------------------------------
def quant_with_scale(matrix: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``aquant.rows`` with an external scale: rne(x / scale), sat [-127, 127]."""
    with np.errstate(invalid="ignore", divide="ignore"):
        quotient = (matrix.astype(F) / scale[:, None].astype(F)).astype(F)
    rounded = np.rint(quotient).astype(F)
    rounded = np.where(np.isnan(quotient), F(0.0), rounded)
    rounded = np.clip(rounded, F(-127.0), F(127.0))
    return rounded.astype(np.int8)


def quant_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``quant_rows``: per-row absmax, ``scale = absmax * (1/127)``, tquantx."""
    absmax = np.max(np.abs(matrix), axis=1).astype(F)
    scale = (absmax * KINV127).astype(F)
    return quant_with_scale(matrix, scale), scale


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def imatmul(a8: np.ndarray, b8: np.ndarray) -> np.ndarray:
    """Exact i8 x i8 accumulate, ``(M,K) @ (K,N)``.

    The i8 products and their sums are integers far below 2^53, so a float64
    BLAS matmul followed by a round is bit-identical to the integer path while
    being much faster than numpy's integer matmul.
    """
    return np.rint(a8.astype(np.float64) @ b8.astype(np.float64))


def imatmul_nt(a8: np.ndarray, b8: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Exact i8 x i8^T accumulate, ``(M,K) @ (N,K)^T``.

    N is chunked so the f64 copy of a large weight matrix (the 151936-row
    lm_head would be 1.2 GB) never has to exist in one piece.
    """
    a64 = a8.astype(np.float64)
    n = b8.shape[0]
    if n <= chunk:
        return np.rint(a64 @ b8.astype(np.float64).T)
    out = np.empty((a64.shape[0], n), dtype=np.float64)
    for n0 in range(0, n, chunk):
        n1 = min(n, n0 + chunk)
        out[:, n0:n1] = a64 @ b8[n0:n1].astype(np.float64).T
    return np.rint(out)


def gemm_nt(a8: np.ndarray, a_scale: np.ndarray,
            w8: np.ndarray, w_scale: np.ndarray) -> np.ndarray:
    """``gemm_nt``: i32 accumulate, then ``acvt`` + two f32 scale multiplies."""
    acc = imatmul_nt(a8, w8).astype(np.int32)
    out = acc.astype(F)
    out = (out * a_scale[:, None]).astype(F)
    out = (out * w_scale[None, :]).astype(F)
    return out


def rmsnorm_from_values(x: np.ndarray, w: np.ndarray, inv: F) -> np.ndarray:
    rows, cols = x.shape
    ss = np.zeros(rows, dtype=F)
    for k in range(0, cols, 32):
        block = x[:, k:k + 32]
        sq = (block * block).astype(F)
        ss = (ss + tree_sum_rows(sq)).astype(F)
    ss = (ss * F(inv)).astype(F)
    ss = (ss + EPS).astype(F)
    rinv = rsqrt_f32(ss)
    out = (x * rinv[:, None]).astype(F)
    out = (out * w[None, :]).astype(F)
    return out


def rmsnorm_rows(x: np.ndarray, w: np.ndarray, inv) -> np.ndarray:
    return rmsnorm_from_values(x, w, inv)


def rope_head_hd(x: np.ndarray, cos_t: np.ndarray, sin_t: np.ndarray) -> np.ndarray:
    half = x.shape[1] // 2
    left = x[:, :half].copy()
    right = x[:, half:].copy()
    out = np.empty_like(x)
    l = (left * cos_t).astype(F)
    t = (right * sin_t).astype(F)
    out[:, :half] = (l - t).astype(F)
    r = (left * sin_t).astype(F)
    t2 = (right * cos_t).astype(F)
    out[:, half:] = (r + t2).astype(F)
    return out


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    e = (gate * F(-1.0)).astype(F)
    e = exp_f32(e)
    e = (e + F(1.0)).astype(F)
    e = rcp_f32(e)
    out = (gate * e).astype(F)
    return (out * up).astype(F)


@dataclass
class AttentionState:
    maximum: np.ndarray
    den: np.ndarray
    res: list[np.ndarray]


def attention_head_blocks(q8: np.ndarray, q_scale: np.ndarray,
                          k8: np.ndarray, k_scale: np.ndarray,
                          v8: np.ndarray, v_scale: np.ndarray,
                          q_rows: int, kv_len: int, pos0: int, causal: bool,
                          qk_scale: F, hd: int) -> np.ndarray:
    """``attention_head_hd128_blocks``: online-softmax over 32-row KV blocks."""
    nr = q_rows
    blocks = hd // 32
    maximum = np.full(nr, -FLT_MAX, dtype=F)
    den = np.zeros(nr, dtype=F)
    res = [np.zeros((nr, 32), dtype=F) for _ in range(blocks)]

    kv_end = kv_len
    if causal:
        kv_end = min(kv_len, pos0 + nr)

    for j in range(0, kv_end, 32):
        nc = min(32, kv_len - j)
        # 1a) S = q8 . k8^T, i32
        s = imatmul_nt(q8, k8[j:j + nc]).astype(np.int32)
        # 1b) dequantise + qk scale
        s = s.astype(F)
        s = (s * q_scale[:, None]).astype(F)
        s = (s * k_scale[j:j + nc][None, :]).astype(F)
        s = (s * qk_scale).astype(F)
        # 1c) causal mask
        if causal:
            diag = pos0 - j
            cols = np.arange(nc, dtype=np.int64)[None, :]
            rows = np.arange(nr, dtype=np.int64)[:, None]
            keep = (cols - rows) <= diag
            s = np.where(keep, s, F(-np.inf)).astype(F)
        # 1d) running max + correction
        nmm = tree_max_rows(s)
        nmm = np.maximum(nmm, maximum).astype(F)
        corr = exp_f32((maximum - nmm).astype(F))
        den = (den * corr).astype(F)
        for i in range(blocks):
            res[i] = (res[i] * corr[:, None]).astype(F)
        p = exp_f32((s - nmm[:, None]).astype(F))
        den = (den + tree_sum_rows(p)).astype(F)
        maximum = nmm
        # 1e) fold the V scale, quantise P
        p = (p * v_scale[j:j + nc][None, :]).astype(F)
        pmax = (row_absmax(p) * KINV127).astype(F)
        pq = quant_with_scale(p, pmax)
        # 1f) accumulate P . V blocks
        for i in range(blocks):
            if i * 32 >= hd:
                break
            part = imatmul(pq, v8[j:j + nc, i * 32:(i + 1) * 32]).astype(F)
            part = (part * pmax[:, None]).astype(F)
            res[i] = (res[i] + part).astype(F)

    den = rcp_f32(den)
    out = np.concatenate([(res[i] * den[:, None]).astype(F) for i in range(blocks)], axis=1)
    return out


def attention_heads_gqa_tokscale(q8: np.ndarray, q_scale: np.ndarray,
                                 k8: np.ndarray, k_scale: np.ndarray,
                                 v8: np.ndarray, v_scale: np.ndarray,
                                 heads: int, kv_heads: int, hd: int,
                                 causal: bool, qk_scale: F) -> np.ndarray:
    rows = q8.shape[0]
    kv_len = k8.shape[0]
    group = heads // kv_heads
    pos0 = kv_len - rows if causal else 0
    out = np.zeros((rows, heads * hd), dtype=F)
    for h in range(heads):
        kv = h // group
        qc = slice(h * hd, (h + 1) * hd)
        kvc = slice(kv * hd, (kv + 1) * hd)
        for qi in range(0, rows, 32):
            nr = min(32, rows - qi)
            out[qi:qi + nr, qc] = attention_head_blocks(
                q8[qi:qi + nr, qc], q_scale[qi:qi + nr],
                k8[:, kvc], k_scale, v8[:, kvc], v_scale,
                nr, kv_len, pos0 + qi, causal, qk_scale, hd)
    return out


def embedding_rmsnorm(emb8: np.ndarray, emb_scale: np.ndarray, ids: np.ndarray,
                      w: np.ndarray, inv) -> tuple[np.ndarray, np.ndarray]:
    """``embedding_rmsnorm_rows``: gather + widen + dequantise + fused norm."""
    x = (emb8[ids].astype(F) * emb_scale[ids][:, None]).astype(F)
    return rmsnorm_from_values(x, w, inv), x


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
@dataclass
class LayerWeights:
    wq: np.ndarray
    wq_s: np.ndarray
    wk: np.ndarray
    wk_s: np.ndarray
    wv: np.ndarray
    wv_s: np.ndarray
    wo: np.ndarray
    wo_s: np.ndarray
    wg: np.ndarray
    wg_s: np.ndarray
    wu: np.ndarray
    wu_s: np.ndarray
    wd: np.ndarray
    wd_s: np.ndarray
    n1: np.ndarray
    nq: np.ndarray
    nk: np.ndarray
    n2: np.ndarray


@dataclass
class Model:
    hidden: int
    layers: list[LayerWeights]
    heads: int
    kv_heads: int
    head_dim: int
    inter: int
    vocab: int
    seq: int
    max_seq: int
    emb8: np.ndarray
    emb_s: np.ndarray
    lm8: np.ndarray
    lm_s: np.ndarray
    nf: np.ndarray
    cos_t: np.ndarray
    sin_t: np.ndarray


@dataclass
class Cache:
    k8: np.ndarray
    k_s: np.ndarray
    v8: np.ndarray
    v_s: np.ndarray


def new_cache(max_seq: int, kv_width: int) -> Cache:
    return Cache(np.zeros((max_seq, kv_width), np.int8), np.zeros(max_seq, F),
                 np.zeros((max_seq, kv_width), np.int8), np.zeros(max_seq, F))


def new_caches(model: "Model") -> list[Cache]:
    """One KV cache per layer, like the device's per-layer cache blocks."""
    return [new_cache(model.max_seq, model.kv_heads * model.head_dim)
            for _ in range(len(model.layers))]


def forward_chunk(model: Model, ids: np.ndarray, caches: list[Cache], pos: int,
                  rows: int, want_logits: bool,
                  record: dict | None = None) -> np.ndarray | None:
    """One prefill/decode chunk; mirrors ``forward_chunk`` in main_big.cpp."""
    H = model.hidden
    HD = model.head_dim
    QW = model.heads * HD
    KW = model.kv_heads * HD
    inv_h = F(1.0) / F(H)
    inv_hd = F(1.0) / F(HD)
    qk_scale = F(1.0 / np.sqrt(float(HD)))

    cos_p = model.cos_t[pos:pos + rows]
    sin_p = model.sin_t[pos:pos + rows]

    # layer 0 fuses the first rmsnorm into the embedding pass; x is the
    # residual stream (the embedding kernel writes it to the x buffer)
    h, x = embedding_rmsnorm(model.emb8, model.emb_s, ids, model.layers[0].n1, inv_h)
    for idx, layer in enumerate(model.layers):
        cache = caches[idx]
        if idx != 0:
            h = rmsnorm_rows(x, layer.n1, inv_h)
        h8, hs = quant_rows(h)
        q = gemm_nt(h8, hs, layer.wq, layer.wq_s)
        k = gemm_nt(h8, hs, layer.wk, layer.wk_s)
        v = gemm_nt(h8, hs, layer.wv, layer.wv_s)
        # q/k norm + rope, per head
        for kvm in range(model.kv_heads):
            c0 = kvm * HD
            kh = k[:, c0:c0 + HD]
            kh = rmsnorm_rows(kh, layer.nk, inv_hd)
            k[:, c0:c0 + HD] = rope_head_hd(kh, cos_p, sin_p)
        for hh in range(model.heads):
            c0 = hh * HD
            qh = q[:, c0:c0 + HD]
            qh = rmsnorm_rows(qh, layer.nq, inv_hd)
            q[:, c0:c0 + HD] = rope_head_hd(qh, cos_p, sin_p)
        k8, ks = quant_rows(k)
        v8, vs = quant_rows(v)
        cache.k8[pos:pos + rows] = k8
        cache.k_s[pos:pos + rows] = ks
        cache.v8[pos:pos + rows] = v8
        cache.v_s[pos:pos + rows] = vs
        q8, qs = quant_rows(q)
        kv_len = pos + rows
        attn = attention_heads_gqa_tokscale(
            q8, qs, cache.k8[:kv_len], cache.k_s[:kv_len],
            cache.v8[:kv_len], cache.v_s[:kv_len],
            model.heads, model.kv_heads, HD, True, qk_scale)
        o8, os = quant_rows(attn)
        o = gemm_nt(o8, os, layer.wo, layer.wo_s)
        x = (x + o).astype(F)

        h = rmsnorm_rows(x, layer.n2, inv_h)
        h8, hs = quant_rows(h)
        g = gemm_nt(h8, hs, layer.wg, layer.wg_s)
        u = gemm_nt(h8, hs, layer.wu, layer.wu_s)
        sw = swiglu(g, u)
        s8, ss = quant_rows(sw)
        o = gemm_nt(s8, ss, layer.wd, layer.wd_s)
        x = (x + o).astype(F)

        if record is not None:
            record.clear()
            record.update(dict(h=h, h8=h8, hs=hs, q=q, k=k, v=v, q8=q8, qs=qs,
                               kc=cache.k8[:kv_len].copy(), ks=cache.k_s[:kv_len].copy(),
                               vc=cache.v8[:kv_len].copy(), vs=cache.v_s[:kv_len].copy(),
                               attn=attn, o8=o8, os=os, o=o, g=g, u=u, sw=sw,
                               s8=s8, ss=ss, x=x, layer=idx))

    logits = None
    if want_logits:
        last = x[rows - 1:rows]
        hl = rmsnorm_rows(last, model.nf, inv_h)
        h8l, hsl = quant_rows(hl)
        logits = gemm_nt(h8l, hsl, model.lm8, model.lm_s)
        if record is not None:
            record["logits"] = logits
    return logits


def argmax_logits(logits: np.ndarray, vocab: int) -> int:
    """Greedy pick mirroring ``argmax_logits`` (per 32-column block)."""
    row = logits.reshape(-1)
    best = 0
    best_key = 0
    first = True
    for n in range(0, vocab, 32):
        block = row[n:n + 32].astype(F)
        # row_max / row_argmax over the block
        idx = int(np.argmax(block))
        mx = block[idx]
        bits = int(np.asarray(mx).view(np.uint32))
        key = (~bits) & 0xFFFFFFFF if bits & 0x80000000 else (bits | 0x80000000)
        if first or key > best_key:
            best_key = key
            best = n + idx
            first = False
    return best


def generate(model: Model, ids: list[int], n_prompt: int, n_gen: int,
             record: dict | None = None) -> tuple[np.ndarray, list[int]]:
    """The built-in greedy loop from ``main()``."""
    caches = new_caches(model)
    ids = list(ids) + [0] * (n_prompt + n_gen - len(ids))
    x = np.zeros((model.seq, model.hidden), dtype=F)
    total = n_prompt + n_gen
    pos = 0
    generated: list[int] = []
    last_logits = None
    while pos < total:
        rows = min(model.seq, n_prompt - pos) if pos < n_prompt else 1
        want_logits = pos + rows >= n_prompt
        chunk_ids = np.asarray(ids[pos:pos + rows], dtype=np.int64)
        logits = forward_chunk(model, chunk_ids, caches, pos, rows, want_logits,
                               record if (record is not None and want_logits) else None)
        if logits is not None:
            last_logits = logits
        pos += rows
        if pos >= n_prompt and pos < total:
            nxt = argmax_logits(last_logits, model.vocab)
            ids[pos] = nxt
            generated.append(nxt)
    return last_logits, generated
