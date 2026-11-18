#!/usr/bin/env python3
"""Run paged-attention vs contiguous-attention guest test."""
from __future__ import annotations
import argparse, struct, subprocess, sys
from pathlib import Path

BLOB_SIZE = 0x20000
RESULT_OFF = 0x1E000

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guest", default="build/paged_attn.bin")
    ap.add_argument("--bridge", default="build/sim_bridge")
    ap.add_argument("--workdir", default="build")
    args = ap.parse_args()
    work = Path(args.workdir)
    blob = work / "paged_attn_data.bin"
    out = work / "paged_attn_out.bin"
    blob.write_bytes(bytes(BLOB_SIZE))
    r = subprocess.run([args.bridge, args.guest, str(blob), str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return 1
    data = out.read_bytes()
    f8, first, fs, _ = struct.unpack_from("<4I", data, RESULT_OFF)
    print(f"paged o8 mismatches: {f8} (first={first})")
    print(f"paged os mismatches: {fs}")
    return 0 if (f8 == 0 and fs == 0) else 1

if __name__ == "__main__":
    sys.exit(main())
