#!/usr/bin/env python3
"""PQWF -> guest blob (``model_big.h`` + ``qwen3_big_data.bin``) and reference.

The blob layout comes from :mod:`tools.pqwf`; the reference forward runs the
numpy mirror of the tile kernels (:mod:`tools.ref_qwen3`) on the *same*
quantised weights read back from the blob, so device logits and reference
logits are directly comparable.

The weights are quantised straight into the (file-backed) blob, so converting
the full 0.6B model does not need to hold the model in memory.

Examples::

    python3 tools/convert_pqwf.py --header build/model_big.h \\
        --blob build/qwen3_big_data.bin --ref build/ref.npz

    # fast iteration: 1 layer, 1024 vocab rows
    python3 tools/convert_pqwf.py --layers 1 --vocab 1024 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pqwf import Layout, Blob, SafeTensors, quantize_into  # noqa: E402
import ref_qwen3 as ref  # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parent.parent.parent / "assets" / "qwen3-0.6b"


def parse_ids(text: str, n: int) -> list[int]:
    ids = [int(part) for part in text.split(",") if part.strip()] if text else []
    if not ids:
        ids = [100 + 10 * i for i in range(n)]
    if len(ids) < n:
        ids = ids + [0] * (n - len(ids))
    return ids[:n]


def rope_tables(max_seq: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    half = head_dim // 2
    freqs = np.arange(half, dtype=np.float64) / float(half)
    inv_freq = 1.0 / (float(theta) ** freqs)
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    ang = pos * inv_freq[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


class BlobView:
    """Typed views on the finished blob file (used by the reference)."""

    def __init__(self, path: str | Path, size: int):
        self.buf = np.memmap(path, dtype=np.uint8, mode="r", shape=(size,))

    def i8(self, offset: int, rows: int, cols: int) -> np.ndarray:
        return self.buf[offset:offset + rows * cols].view(np.int8).reshape(rows, cols)

    def f32(self, offset: int, count: int) -> np.ndarray:
        return self.buf[offset:offset + 4 * count].view(np.float32).reshape(-1)


DATA_KEY = {"q_proj": "kLQ", "k_proj": "kLK", "v_proj": "kLV", "o_proj": "kLO",
            "gate_proj": "kLG", "up_proj": "kLU", "down_proj": "kLD"}
SCALE_KEY = {"q_proj": "kLQ_SCALE", "k_proj": "kLK_SCALE", "v_proj": "kLV_SCALE",
             "o_proj": "kLO_SCALE", "gate_proj": "kLG_SCALE", "up_proj": "kLU_SCALE",
             "down_proj": "kLD_SCALE"}
ATTN_PROJ = {"q_proj", "k_proj", "v_proj", "o_proj"}


def reference_model(blob_path: str | Path, layout: Layout, cos_t: np.ndarray,
                    sin_t: np.ndarray) -> ref.Model:
    """Build the numpy reference model from the quantised weights in a blob."""
    off = layout.offsets
    view = BlobView(blob_path, off["kSCRATCH_END"])
    hidden = layout.hidden
    head_dim = layout.head_dim
    q_width = layout.heads * head_dim
    kv_width = layout.kv_heads * head_dim
    weight_shapes = {
        "q_proj": (q_width, hidden), "k_proj": (kv_width, hidden),
        "v_proj": (kv_width, hidden), "o_proj": (hidden, q_width),
        "gate_proj": (layout.inter, hidden), "up_proj": (layout.inter, hidden),
        "down_proj": (hidden, layout.inter),
    }
    layers = []
    for idx in range(layout.layers):
        base = off["kLQ"] + idx * off["kLAYERW_STRIDE"]
        nbase = off["kN1"] + idx * off["kLAYERN_STRIDE"]

        def w(name: str) -> tuple[np.ndarray, np.ndarray]:
            rows, cols = weight_shapes[name]
            return (view.i8(base + (off[DATA_KEY[name]] - off["kLQ"]), rows, cols),
                    view.f32(base + (off[SCALE_KEY[name]] - off["kLQ"]), rows))

        wq, wq_s = w("q_proj")
        wk, wk_s = w("k_proj")
        wv, wv_s = w("v_proj")
        wo, wo_s = w("o_proj")
        wg, wg_s = w("gate_proj")
        wu, wu_s = w("up_proj")
        wd, wd_s = w("down_proj")
        layers.append(ref.LayerWeights(
            wq=wq, wq_s=wq_s, wk=wk, wk_s=wk_s, wv=wv, wv_s=wv_s, wo=wo, wo_s=wo_s,
            wg=wg, wg_s=wg_s, wu=wu, wu_s=wu_s, wd=wd, wd_s=wd_s,
            n1=view.f32(nbase, hidden),
            nq=view.f32(nbase + (off["kNQ"] - off["kN1"]), head_dim),
            nk=view.f32(nbase + (off["kNK"] - off["kN1"]), head_dim),
            n2=view.f32(nbase + (off["kN2"] - off["kN1"]), hidden)))
    return ref.Model(hidden=hidden, layers=layers, heads=layout.heads,
                     kv_heads=layout.kv_heads, head_dim=head_dim, inter=layout.inter,
                     vocab=layout.vocab, seq=layout.seq, max_seq=layout.max_seq,
                     emb8=view.i8(off["kEMB"], layout.vocab, hidden),
                     emb_s=view.f32(off["kEMB"] + layout.vocab * hidden, layout.vocab),
                     lm8=view.i8(off["kLM"], layout.vocab, hidden),
                     lm_s=view.f32(off["kLM"] + layout.vocab * hidden, layout.vocab),
                     nf=view.f32(off["kNF"], hidden), cos_t=cos_t, sin_t=sin_t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--layers", type=int, default=0, help="number of layers (0 = all)")
    ap.add_argument("--vocab", type=int, default=0, help="vocab rows to keep (0 = all)")
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--n-prompt", type=int, default=32)
    ap.add_argument("--n-gen", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--header", default="build/model_big.h")
    ap.add_argument("--blob", default="build/qwen3_big_data.bin")
    ap.add_argument("--ref", default="build/ref.npz")
    ap.add_argument("--no-ref", action="store_true", help="skip the numpy reference")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    inter = int(cfg["intermediate_size"])
    vocab = int(cfg["vocab_size"])
    n_layers = int(cfg["num_hidden_layers"])
    if args.layers:
        n_layers = args.layers
    if args.vocab:
        vocab = args.vocab
    seq, max_seq = args.seq, args.max_seq
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim

    layout = Layout(hidden=hidden, layers=n_layers, heads=heads, kv_heads=kv_heads,
                    head_dim=head_dim, inter=inter, vocab=vocab, seq=seq,
                    max_seq=max_seq).compute()
    off = layout.offsets
    layer_stride = off["kLAYERW_STRIDE"]
    norm_stride = off["kLAYERN_STRIDE"]

    print(f"layout: {n_layers} layers, hidden={hidden}, vocab={vocab}, seq={seq}, "
          f"max_seq={max_seq}, blob={off['kSCRATCH_END']} bytes", flush=True)
    print(f"loading {model_dir / 'model.safetensors'} ...", flush=True)
    st = SafeTensors(model_dir / "model.safetensors")

    Path(args.blob).parent.mkdir(parents=True, exist_ok=True)
    Path(args.header).parent.mkdir(parents=True, exist_ok=True)
    blob = Blob(args.blob, off["kSCRATCH_END"])

    # ---- control block / rowwise tables ------------------------------------
    ids = parse_ids(args.ids, args.n_prompt + args.n_gen)
    blob.put(off["kIDS"], np.asarray(ids, dtype=np.int32))
    blob.put(off["kCTRL"], np.asarray([args.n_prompt, args.n_gen, max_seq, 0], dtype=np.int32))
    blob.put(off["kNF"], st.vector_f32("model.norm.weight"))
    cos_t, sin_t = rope_tables(max_seq, head_dim, float(cfg.get("rope_theta", 1e6)))
    blob.put(off["kCOS"], cos_t)
    blob.put(off["kSIN"], sin_t)

    # ---- embedding / lm_head (per-row int8 + per-row f32 scale) ------------
    print("quantising embedding / lm_head ...", flush=True)
    quantize_into(st, "model.embed_tokens.weight", blob,
                  off["kEMB"], off["kEMB"] + vocab * hidden, vocab, hidden)
    lm_name = "lm_head.weight" if "lm_head.weight" in st.header else "model.embed_tokens.weight"
    quantize_into(st, lm_name, blob,
                  off["kLM"], off["kLM"] + vocab * hidden, vocab, hidden)

    # ---- per-layer weights / norms ----------------------------------------
    weight_shapes = {
        "q_proj": (q_width, hidden),
        "k_proj": (kv_width, hidden),
        "v_proj": (kv_width, hidden),
        "o_proj": (hidden, q_width),
        "gate_proj": (inter, hidden),
        "up_proj": (inter, hidden),
        "down_proj": (hidden, inter),
    }
    for idx in range(n_layers):
        prefix = f"model.layers.{idx}."
        base = off["kLQ"] + idx * layer_stride
        for proj, (rows, cols) in weight_shapes.items():
            name = prefix + ("self_attn." if proj in ATTN_PROJ else "mlp.") + proj + ".weight"
            quantize_into(st, name, blob,
                          base + (off[DATA_KEY[proj]] - off["kLQ"]),
                          base + (off[SCALE_KEY[proj]] - off["kLQ"]), rows, cols)
        nbase = off["kN1"] + idx * norm_stride
        blob.put(nbase, st.vector_f32(prefix + "input_layernorm.weight"))
        blob.put(nbase + (off["kNQ"] - off["kN1"]), st.vector_f32(prefix + "self_attn.q_norm.weight"))
        blob.put(nbase + (off["kNK"] - off["kN1"]), st.vector_f32(prefix + "self_attn.k_norm.weight"))
        blob.put(nbase + (off["kN2"] - off["kN1"]),
                 st.vector_f32(prefix + "post_attention_layernorm.weight"))
        if n_layers > 4 and (idx + 1) % 4 == 0:
            print(f"  ... {idx + 1}/{n_layers} layers", flush=True)
    del st

    blob.flush()
    Path(args.header).write_text(layout.header_text())
    print(f"blob:   {args.blob} ({off['kSCRATCH_END']} bytes)")
    print(f"header: {args.header}")
    print(f"ids:    {ids[:args.n_prompt + args.n_gen]}")

    if args.no_ref:
        return 0

    # ---- reference: read the quantised weights back from the blob ----------
    print("running reference ...", flush=True)
    model = reference_model(args.blob, layout, cos_t, sin_t)

    record: dict = {}
    logits, generated = ref.generate(model, ids, args.n_prompt, args.n_gen, record)
    payload = {f"ref_{key}": value for key, value in record.items()}
    payload["ref_logits"] = logits if logits is not None else np.zeros((1, vocab), np.float32)
    payload["ids"] = np.asarray(ids, dtype=np.int32)
    payload["generated"] = np.asarray(generated, dtype=np.int32)
    Path(args.ref).parent.mkdir(parents=True, exist_ok=True)
    np.savez(Path(args.ref), **payload)
    print(f"ref:    {args.ref} (logits {payload['ref_logits'].shape})")
    top = int(np.argmax(payload["ref_logits"].reshape(-1)))
    print(f"ref top-1: {top} (logit {float(payload['ref_logits'].reshape(-1)[top]):.6f})")
    if generated:
        print(f"ref generated: {generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
