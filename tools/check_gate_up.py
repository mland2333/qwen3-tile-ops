#!/usr/bin/env python3
"""Run the isolated gate/up fusion guest and report mismatch counts."""
from __future__ import annotations
import argparse, struct, subprocess, sys
from pathlib import Path

BLOB_SIZE = 0x70000
RESULT_OFF = 0x6B000

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guest", default="build/gate_up.bin")
    ap.add_argument("--bridge", default="build/sim_bridge")
    ap.add_argument("--workdir", default="build")
    args = ap.parse_args()
    work = Path(args.workdir)
    blob = work / "gate_up_data.bin"
    out = work / "gate_up_out.bin"
    blob.write_bytes(bytes(BLOB_SIZE))
    r = subprocess.run([args.bridge, args.guest, str(blob), str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return 1
    data = out.read_bytes()
    gf, uf, sf, sfirst, ssf, _ = struct.unpack_from("<6I", data, RESULT_OFF)
    print(f"s8 mismatches: {sf} (first={sfirst})")
    print(f"ss mismatches: {ssf}")
    return 0 if (sf == ssf == 0) else 1

if __name__ == "__main__":
    sys.exit(main())
