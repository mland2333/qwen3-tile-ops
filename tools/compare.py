#!/usr/bin/env python3
"""Compare a finished guest blob against the numpy reference.

    python3 tools/compare.py --header build/model_big.h --out build/out.bin \\
                             --ref build/ref.npz

The reference npz holds the *last* layer's intermediates (``ref_h``, ``ref_q``,
...) and the logits; the blob dump holds the same buffers at the addresses the
generated header names, so a per-stage diff points at the kernel that is off.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def read_layout(header: Path) -> tuple[dict[str, int], dict[str, int]]:
    ints: dict[str, int] = {}
    longs: dict[str, int] = {}
    for line in header.read_text().splitlines():
        m = re.match(r"\s*inline constexpr int (\w+)\s*=\s*(\d+);", line)
        if m:
            ints[m.group(1)] = int(m.group(2))
        m = re.match(r"\s*inline constexpr long (\w+)\s*=\s*(\d+);", line)
        if m:
            longs[m.group(1)] = int(m.group(2))
    return ints, longs


DTYPES = {
    "i8": np.int8,
    "f32": np.float32,
    "i32": np.int32,
    "u32": np.uint32,
}


def stage_regions(lay: dict[str, int], consts: dict[str, int]) -> list[tuple[str, str, int, tuple[int, ...]]]:
    """(ref key, dtype, offset, shape) for every recorded stage."""
    hidden = consts["kHidden"]
    inter = consts["kInter"]
    heads = consts["kHeads"]
    kv_heads = consts["kKvHeads"]
    hd = consts["kHeadDim"]
    vocab = consts["kVocab"]
    qw = heads * hd
    kw = kv_heads * hd
    rows = None  # filled from the reference arrays
    regions = [
        ("h", "f32", lay["kH"], (rows, hidden)),
        ("h8", "i8", lay["kH8"], (rows, hidden)),
        ("hs", "f32", lay["kHS"], (rows,)),
        ("q", "f32", lay["kQ"], (rows, qw)),
        ("k", "f32", lay["kK"], (rows, kw)),
        ("v", "f32", lay["kV"], (rows, kw)),
        ("q8", "i8", lay["kQ8"], (rows, qw)),
        ("qs", "f32", lay["kQS"], (rows,)),
        ("attn", "f32", lay["kATTN"], (rows, qw)),
        ("o", "f32", lay["kO"], (rows, hidden)),
        ("o8", "i8", lay["kO8"], (rows, qw)),
        ("os", "f32", lay["kOS"], (rows,)),
        ("g", "f32", lay["kG"], (rows, inter)),
        ("u", "f32", lay["kU"], (rows, inter)),
        ("sw", "f32", lay["kSW"], (rows, inter)),
        ("s8", "i8", lay["kS8"], (rows, inter)),
        ("ss", "f32", lay["kSS"], (rows,)),
        ("x", "f32", lay["kX"], (rows, hidden)),
        ("logits", "f32", lay["kLOGITS"], (1, vocab)),
    ]
    # KV cache of the last layer
    kv_base = lay["kKV_BASE"] + (consts["kLayers"] - 1) * lay["kKV_STRIDE"]
    regions += [
        ("kc", "i8", kv_base, (None, kw)),
        ("ks", "f32", kv_base + lay["kKV_SOFF"], (None,)),
        ("vc", "i8", kv_base + lay["kKV_VOFF"], (None, kw)),
        ("vs", "f32", kv_base + lay["kKV_VOFF"] + lay["kKV_SOFF"], (None,)),
    ]
    return regions


def describe(a: np.ndarray, b: np.ndarray) -> str:
    if a.shape != b.shape:
        return f"shape {a.shape} != {b.shape}"
    if a.size == 0:
        return "empty"
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    if np.issubdtype(a.dtype, np.integer) or np.issubdtype(b.dtype, np.integer):
        diff = flat_a.astype(np.int64) - flat_b.astype(np.int64)
        bad = int(np.count_nonzero(diff))
        worst = int(np.argmax(np.abs(diff)))
        return f"int mismatches {bad}/{flat_a.size}" + (
            f" (worst {flat_a[worst]} vs {flat_b[worst]} @{worst})" if bad else "")
    diff = flat_a.astype(np.float64) - flat_b.astype(np.float64)
    bits_equal = np.array_equal(flat_a.view(np.uint32), flat_b.view(np.uint32))
    finite = np.isfinite(diff)
    absdiff = np.where(finite, np.abs(diff), 0.0)
    worst = int(np.argmax(absdiff))
    scale = float(np.max(np.abs(flat_b.astype(np.float64)))) or 1.0
    return (f"bit-exact {bits_equal} max|d|={float(absdiff[worst]):.3g} "
            f"(rel {float(absdiff[worst]) / scale:.2e}) @{worst}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", default="build/model_big.h")
    ap.add_argument("--out", default="build/out.bin")
    ap.add_argument("--ref", default="build/ref.npz")
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    consts, lay = read_layout(Path(args.header))
    data = np.frombuffer(Path(args.out).read_bytes(), dtype=np.uint8)
    ref = np.load(args.ref)
    if args.layers:
        consts["kLayers"] = args.layers

    failures = 0
    for name, dtype, offset, shape in stage_regions(lay, consts):
        key = "ref_" + name
        if key not in ref:
            continue
        expected = ref[key]
        if any(dim is None for dim in shape):
            shape = tuple(expected.shape)
        count = int(np.prod(shape))
        elem = np.dtype(DTYPES[dtype]).itemsize
        raw = data[offset:offset + count * elem]
        got = raw.view(DTYPES[dtype]).reshape(shape)
        aliased = {"q8": 1, "qs": 1, "o": 1}
        if name in aliased:
            # main_big reuses the first row of these buffers for the final-norm
            # output (h8l/hsl/hl), so row 0 is not the kernel's result.
            if shape[0] <= aliased[name]:
                print(f"skip {name:7s} (aliased with the final-norm buffer)")
                continue
            got = got[aliased[name]:]
            expected = expected[aliased[name]:]
        status = describe(got, expected)
        ok = "bit-exact True" in status or "int mismatches 0/" in status
        if not ok:
            failures += 1
        print(f"{'ok  ' if ok else 'FAIL'} {name:7s} {str(shape):18s} {status}")
    if "ids" in ref and "generated" in ref:
        max_seq = consts["kMaxSeq"]
        ids = data[lay["kIDS"]:lay["kIDS"] + 4 * max_seq].view(np.int32)
        prompt = ref["ids"]
        generated = ref["generated"]
        n_prompt = int(len(prompt) - len(generated)) if len(generated) else int(len(prompt))
        n_prompt = int(np.count_nonzero(prompt)) and n_prompt
        got_gen = ids[len(prompt) - len(generated):len(prompt)]
        ok = np.array_equal(got_gen, generated)
        if not ok:
            failures += 1
        print(f"{'ok  ' if ok else 'FAIL'} ids     device {list(got_gen)} ref {list(generated)}")
    print(f"{failures} stage(s) differ")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
