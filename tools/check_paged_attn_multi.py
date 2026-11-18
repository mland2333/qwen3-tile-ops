#!/usr/bin/env python3
from __future__ import annotations
import argparse, struct, subprocess, sys
from pathlib import Path
BLOB=0x30000; RES=0x2F800
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--guest',default='build/paged_attn_multi.bin'); ap.add_argument('--bridge',default='build/sim_bridge'); ap.add_argument('--workdir',default='build'); a=ap.parse_args()
    w=Path(a.workdir); blob=w/'paged_attn_multi_data.bin'; out=w/'paged_attn_multi_out.bin'
    blob.write_bytes(bytes(BLOB))
    r=subprocess.run([a.bridge,a.guest,str(blob),str(out)],capture_output=True,text=True)
    if r.returncode: print(r.stderr,file=sys.stderr); return 1
    d=out.read_bytes(); f0,s0,f1,s1=struct.unpack_from('<4I',d,RES)
    print(f'req0 o8/os mismatches: {f0}/{s0}'); print(f'req1 o8/os mismatches: {f1}/{s1}')
    return 0 if f0==s0==f1==s1==0 else 1
sys.exit(main())
