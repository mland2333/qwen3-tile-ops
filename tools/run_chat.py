#!/usr/bin/env python3
"""Text in / text out on the Tile-NPU simulator (0.6B, greedy).

    python3 tools/run_chat.py "你好" --max-new 16
    python3 tools/run_chat.py --ids 151644,9707 --max-new 4

The tool copies the blob, patches the ChatML prompt ids, runs ``sim_bridge``
and decodes the generated ids.  Greedy decoding happens on the device
(``argmax_logits``); the host only tokenises and decodes.
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EOS_IDS = {151645, 151643}  # <|im_end|>, <|endoftext|>


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


CHAT_TEMPLATE = "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="你好，请用一句话介绍你自己。")
    ap.add_argument("--tokenizer", default=str(ROOT.parent / "assets" / "qwen3-0.6b" / "tokenizer.json"))
    ap.add_argument("--header", default=str(ROOT / "build" / "model_big.h"))
    ap.add_argument("--blob", default=str(ROOT / "build" / "qwen3_big_data.bin"))
    ap.add_argument("--guest", default=str(ROOT / "build" / "qwen3_big.bin"))
    ap.add_argument("--bridge", default=str(ROOT / "build" / "sim_bridge"))
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--raw", action="store_true", help="do not wrap the text in ChatML")
    ap.add_argument("--ids", default="", help="use raw token ids instead of the text")
    args = ap.parse_args()

    consts, lay = read_layout(Path(args.header))
    max_seq = consts["kMaxSeq"]
    vocab = consts["kVocab"]
    tokenizer = Tokenizer(args.tokenizer)

    if args.ids:
        prompt_ids = [int(t) for t in args.ids.split(",") if t.strip()]
    else:
        text = args.text if args.raw else CHAT_TEMPLATE.format(text=args.text)
        prompt_ids = tokenizer.encode(text)
    if len(prompt_ids) + args.max_new > max_seq:
        print(f"error: prompt {len(prompt_ids)} + max-new {args.max_new} > max_seq {max_seq}",
              file=sys.stderr)
        return 2

    work = ROOT / "build" / "chat_blob.bin"
    shutil.copyfile(args.blob, work)
    subprocess.run([sys.executable, str(ROOT / "tools" / "patch_blob.py"),
                    "--blob", str(work), "--header", args.header,
                    "--ids", ",".join(map(str, prompt_ids)),
                    "--n-gen", str(args.max_new)], check=True)

    ids_dump = ROOT / "build" / "chat_ids.bin"
    logits_dump = ROOT / "build" / "chat_logits.bin"
    cmd = [args.bridge, args.guest, str(work), str(Path("/dev/null")),
           "--dump", f"{ids_dump}@{lay['kIDS']}:{4 * max_seq}",
           "--dump", f"{logits_dump}@{lay['kLOGITS']}:{4 * vocab}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print("guest failed", file=sys.stderr)
        return 1

    ids = np.frombuffer(ids_dump.read_bytes(), dtype=np.int32)
    generated = [int(x) for x in ids[len(prompt_ids):len(prompt_ids) + args.max_new]]
    stopped = []
    for tid in generated:
        if tid in EOS_IDS:
            break
        stopped.append(tid)

    print(f"prompt: {len(prompt_ids)} tokens")
    print(f"new ids: {generated}")
    if logits_dump.exists() and 4 * vocab <= logits_dump.stat().st_size:
        logits = np.frombuffer(logits_dump.read_bytes()[:4 * vocab], dtype=np.float32)
        top = int(np.argmax(logits))
        print(f"device top-1: {top} (logit {float(logits[top]):.4f})")
    text = tokenizer.decode(stopped)
    print("--- output ---")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
