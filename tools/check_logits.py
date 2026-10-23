#!/usr/bin/env python3
"""从 guest 跑完的 blob 里取出最后一行的 logits，和参考 f32 比。

参考文件是裸 f32 数组（vocab 个），由 host 侧实现导出。

    python3 tools/check_logits.py --header config/model_big.h \
                                  --out build/out.bin --ref /tmp/ref32.bin
"""
from __future__ import annotations

import argparse
import array
import re
import sys
from pathlib import Path


def read_layout(header: Path) -> dict[str, int]:
    """从生成出来的布局头里抓 `inline constexpr long kX = N;`。"""
    out: dict[str, int] = {}
    for line in header.read_text().splitlines():
        m = re.match(r"\s*inline constexpr long (\w+)\s*=\s*(\d+);", line)
        if m:
            out[m.group(1)] = int(m.group(2))
        m = re.match(r"\s*inline constexpr int (\w+)\s*=\s*(\d+);", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--row", type=int, default=-1,
                    help="第几行 logits（默认最后一行）")
    args = ap.parse_args()

    lay = read_layout(Path(args.header))
    need = {"kLOGITS", "kVocab", "kSeq"}
    if not need <= lay.keys():
        print(f"布局头缺少 {need - lay.keys()}", file=sys.stderr)
        return 2

    vocab = lay["kVocab"]
    raw = Path(args.out).read_bytes()
    got = array.array("f")
    got.frombytes(raw[lay["kLOGITS"]:lay["kLOGITS"] + vocab * 4])
    ref = array.array("f")
    ref.frombytes(Path(args.ref).read_bytes()[:vocab * 4])

    if len(ref) != vocab:
        print(f"参考长度 {len(ref)} != vocab {vocab}", file=sys.stderr)
        return 2

    same = list(got) == list(ref)
    worst = max(range(vocab), key=lambda i: abs(got[i] - ref[i]))
    print(f"logits: 逐位相同 = {same}")
    print(f"        max|Δ| = {abs(got[worst] - ref[worst]):.3g} @ {worst}")
    print(f"        argmax  = guest {max(range(vocab), key=got.__getitem__)}"
          f" / ref {max(range(vocab), key=ref.__getitem__)}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
