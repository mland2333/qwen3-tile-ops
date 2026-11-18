#!/usr/bin/env python3
from __future__ import annotations
import argparse, struct, subprocess, sys
from pathlib import Path
BLOB=0xB000; RES=0xAA00
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--guest',default='build/paged_kv_batch.bin'); ap.add_argument('--bridge',default='build/sim_bridge'); ap.add_argument('--workdir',default='build'); a=ap.parse_args()
    w=Path(a.workdir); blob=w/'paged_kv_batch_data.bin'; out=w/'paged_kv_batch_out.bin'
    blob.write_bytes(bytes(BLOB))
    r=subprocess.run([a.bridge,a.guest,str(blob),str(out)],capture_output=True,text=True)
    if r.returncode: print(r.stderr,file=sys.stderr); return 1
    fails,=struct.unpack_from('<I',out.read_bytes(),RES)
    print(f'batch kv scatter mismatches: {fails}')
    return 0 if fails==0 else 1
sys.exit(main())
