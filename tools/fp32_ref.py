#!/usr/bin/env python3
"""Plain f32 Qwen3 forward (numpy) — an implementation that shares nothing
with the tile kernels: no int8 quantisation, no 32x32 blocking, no online
softmax.  It is used to check that the tile pipeline implements the *model*
(not just itself) and how far the int8 path drifts from f32.

    python3 tools/fp32_ref.py --ids 151644,9707
    python3 tools/fp32_ref.py --prompt "你好"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pqwf import SafeTensors  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT.parent / "assets" / "qwen3-0.6b"


class F32Model:
    def __init__(self, model_dir: Path):
        self.cfg = json.loads((model_dir / "config.json").read_text())
        st = SafeTensors(model_dir / "model.safetensors")
        self.st = st
        self.hidden = int(self.cfg["hidden_size"])
        self.heads = int(self.cfg["num_attention_heads"])
        self.kv_heads = int(self.cfg["num_key_value_heads"])
        self.head_dim = int(self.cfg["head_dim"])
        self.inter = int(self.cfg["intermediate_size"])
        self.layers = int(self.cfg["num_hidden_layers"])
        self.eps = float(self.cfg["rms_norm_eps"])
        self.theta = float(self.cfg.get("rope_theta", 1e6))
        self.emb = st.block_f32("model.embed_tokens.weight", 0, 10**9)
        self.lm = st.block_f32("lm_head.weight", 0, 10**9) if "lm_head.weight" in st.header else self.emb
        self.nf = st.vector_f32("model.norm.weight")
        self._wcache: dict[str, np.ndarray] = {}
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.theta ** (np.arange(half, dtype=np.float64) / half))
        self.inv_freq = inv_freq

    def w(self, idx: int, name: str) -> np.ndarray:
        key = f"{idx}.{name}"
        if key not in self._wcache:
            # bf16 -> f32 conversion is expensive; the f32 copy is ~15 MB/layer
            self._wcache[key] = self.st.block_f32(
                f"model.layers.{idx}.{name}.weight", 0, 10**9)
        return self._wcache[key]

    def v(self, idx: int, name: str) -> np.ndarray:
        return self.st.vector_f32(f"model.layers.{idx}.{name}.weight")

    @staticmethod
    def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
        x = x.astype(np.float64)
        var = np.mean(x * x, axis=-1, keepdims=True)
        return (x / np.sqrt(var + eps) * weight).astype(np.float32)

    def forward(self, ids: list[int]) -> np.ndarray:
        seq = len(ids)
        x = self.emb[ids].astype(np.float32)
        half = self.head_dim // 2
        pos = np.arange(seq, dtype=np.float64)[:, None]
        ang = pos * self.inv_freq[None, :]
        cos = np.cos(ang).astype(np.float32)
        sin = np.sin(ang).astype(np.float32)
        scale = 1.0 / np.sqrt(float(self.head_dim))

        for layer in range(self.layers):
            h = self.rmsnorm(x, self.v(layer, "input_layernorm"), self.eps)
            wq = self.w(layer, "self_attn.q_proj")
            wk = self.w(layer, "self_attn.k_proj")
            wv = self.w(layer, "self_attn.v_proj")
            wo = self.w(layer, "self_attn.o_proj")
            q = (h @ wq.T).reshape(seq, self.heads, self.head_dim)
            k = (h @ wk.T).reshape(seq, self.kv_heads, self.head_dim)
            v = (h @ wv.T).reshape(seq, self.kv_heads, self.head_dim)
            # per-head q/k norm (over head_dim)
            q = self.rmsnorm(q, self.v(layer, "self_attn.q_norm"), self.eps)
            k = self.rmsnorm(k, self.v(layer, "self_attn.k_norm"), self.eps)
            # rotate-half rope
            def rope(t: np.ndarray) -> np.ndarray:
                left = t[..., :half]
                right = t[..., half:]
                out = np.empty_like(t)
                out[..., :half] = left * cos[:, None, :] - right * sin[:, None, :]
                out[..., half:] = left * sin[:, None, :] + right * cos[:, None, :]
                return out

            q = rope(q)
            k = rope(k)
            group = self.heads // self.kv_heads
            kx = np.repeat(k, group, axis=1).transpose(1, 0, 2)  # (heads, seq, hd)
            vx = np.repeat(v, group, axis=1).transpose(1, 0, 2)
            qx = q.transpose(1, 0, 2)                             # (heads, seq, hd)
            scores = (qx @ kx.transpose(0, 2, 1)).astype(np.float32) * scale
            mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
            scores = np.where(mask[None, :, :], -np.inf, scores)
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores).astype(np.float32)
            probs /= probs.sum(axis=-1, keepdims=True)
            attn = (probs @ vx).transpose(1, 0, 2).reshape(seq, self.heads * self.head_dim)
            x = x + attn @ wo.T

            h = self.rmsnorm(x, self.v(layer, "post_attention_layernorm"), self.eps)
            gate = h @ self.w(layer, "mlp.gate_proj").T
            up = h @ self.w(layer, "mlp.up_proj").T
            silu = gate / (1.0 + np.exp(-gate.astype(np.float32)))
            x = x + (silu * up) @ self.w(layer, "mlp.down_proj").T

        x = self.rmsnorm(x, self.nf, self.eps)
        return x @ self.lm.T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    ap.add_argument("--ids", default="")
    ap.add_argument("--model-dir", default=str(ASSETS))
    ap.add_argument("--tokenizer", default=str(ASSETS / "tokenizer.json"))
    ap.add_argument("--raw", action="store_true", help="do not wrap the prompt in ChatML")
    ap.add_argument("--gen", type=int, default=0, help="greedy generation steps")
    args = ap.parse_args()

    if args.ids:
        ids = [int(t) for t in args.ids.split(",") if t.strip()]
    else:
        text = args.prompt if args.raw else f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
        ids = Tokenizer(args.tokenizer).encode(text)
    model = F32Model(Path(args.model_dir))
    logits = model.forward(ids)
    last = logits[-1]
    top = np.argsort(-last)[:5]
    print(f"ids ({len(ids)}): {ids}")
    print(f"fp32 top-5: {[(int(i), float(last[i])) for i in top]}")
    if args.gen:
        generated = []
        for _ in range(args.gen):
            nxt = int(np.argmax(last))
            if nxt in (151645, 151643):
                break
            generated.append(nxt)
            ids.append(nxt)
            last = model.forward(ids)[-1]
        print(f"fp32 greedy ids: {generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
