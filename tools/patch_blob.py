#!/usr/bin/env python3
"""Patch prompt ids / control block into a generated blob, in place.

    python3 tools/patch_blob.py --blob build/qwen3_big_data.bin \\
        --header build/model_big.h --ids 100,200 --n-gen 8 [--mode 0]

The blob's control block is ``CTRL = [n_prompt, n_gen, max_seq, mode]``:
mode 0 runs prefill + the built-in greedy loop, mode 1 runs a single decode
step at absolute position ``n_prompt``.  Activations, logits and the KV cache
are cleared so the file can be re-run without stale state.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", default="build/qwen3_big_data.bin")
    ap.add_argument("--header", default="build/model_big.h")
    ap.add_argument("--ids", default="")
    ap.add_argument("--n-gen", type=int, default=0)
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--keep-state", action="store_true",
                    help="keep activations and the KV cache (host-side decode loop)")
    args = ap.parse_args()

    consts, lay = read_layout(Path(args.header))
    ids = [int(t) for t in args.ids.split(",") if t.strip()]
    if not ids:
        print("no ids given", file=sys.stderr)
        return 2
    max_seq = consts["kMaxSeq"]
    if len(ids) + args.n_gen > max_seq:
        print(f"prompt {len(ids)} + n_gen {args.n_gen} exceeds max_seq {max_seq}", file=sys.stderr)
        return 2

    with open(args.blob, "r+b") as handle:
        if not args.keep_state:
            # activations from kH through the KV cache, plus logits
            start = lay["kH"]
            end = lay["kSCRATCH_END"]
            handle.seek(start)
            remaining = end - start
            chunk = bytes(1 << 20)
            while remaining > 0:
                step = min(remaining, len(chunk))
                handle.write(chunk[:step])
                remaining -= step
            handle.seek(lay["kLOGITS"])
            handle.write(bytes(consts["kSeq"] * consts["kVocab"] * 4))
        padded = ids + [0] * (max_seq - len(ids))
        handle.seek(lay["kIDS"])
        handle.write(struct.pack(f"<{max_seq}i", *padded))
        handle.seek(lay["kCTRL"])
        handle.write(struct.pack("<4i", len(ids), args.n_gen, max_seq, args.mode))
    print(f"patched {args.blob}: {len(ids)} prompt ids, n_gen={args.n_gen}, mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
