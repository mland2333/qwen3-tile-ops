#!/usr/bin/env python3
"""Run the per-group GEMM guest and read its pass/fail result word."""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

BLOB_SIZE = 0x30000
RESULT_OFF = 0x1D000
RESULT2_OFF = 0x2C300


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guest", default="build/group_gemm.bin")
    ap.add_argument("--bridge", default="build/sim_bridge")
    ap.add_argument("--workdir", default="build")
    args = ap.parse_args()

    work = Path(args.workdir)
    blob = work / "group_gemm_data.bin"
    out = work / "group_gemm_out.bin"
    blob.write_bytes(bytes(BLOB_SIZE))

    result = subprocess.run(
        [args.bridge, args.guest, str(blob), str(out)],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print("guest failed", file=sys.stderr)
        return 1

    data = out.read_bytes()
    rc = 0
    fails, first, got, ref = struct.unpack_from("<4I", data, RESULT_OFF)
    if fails == 0:
        print("ok   per-group gemm  (32x256 @ 128x256, K split 2x128) bit-exact")
    else:
        print(f"FAIL per-group gemm: {fails} words differ, first={first} "
              f"got=0x{got:08x} ref=0x{ref:08x}")
        rc = 1
    qfails, qfirst, qgot, qref = struct.unpack_from("<4I", data, RESULT2_OFF)
    if qfails == 0:
        print("ok   per-head quant (32x256 f32 -> i8, 2 groups of 128) bit-exact")
    else:
        print(f"FAIL per-head quant: {qfails} entries differ, first={qfirst} "
              f"got=0x{qgot:08x} ref=0x{qref:08x}")
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
