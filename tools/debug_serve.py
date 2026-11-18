#!/usr/bin/env python3
"""Compare batch vs single serve guest memory after prefill and decode.

Requires the debug guest (``make serve-debug`` / ``build/serve_debug.bin``):
the scratch layout used here includes sentinel/snap/dump/before.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_serve import Runtime, Serve

DEFAULT_BLOB = Path("build/full/qwen3_big_data.bin")
DEFAULT_HDR = Path("build/full/model_big.h")
DEFAULT_GUEST = Path("build/serve_debug.bin")
DEFAULT_BRIDGE = Path("build/sim_bridge")
DEFAULT_WORK = Path("build/serve_debug")


def kv_pages(out: bytes, h: dict[str, int], pages: list[int]) -> bytes:
    """Logical K/V + scales of one request, gathered from its physical pages."""
    pt = 32
    row = h["kKvHeads"] * h["kHeadDim"]
    srow = h["kKvHeads"] * 4
    kpage, spage = pt * row, pt * srow
    chunks: list[bytes] = []
    for layer in range(h["kLayers"]):
        base = h["kKV_BASE"] + layer * h["kKV_STRIDE"]
        for page in pages:
            slabs = (
                (base, kpage),
                (base + h["kKV_SOFF"], spage),
                (base + h["kKV_VOFF"], kpage),
                (base + h["kKV_VOFF"] + h["kKV_SOFF"], spage),
            )
            for off, size in slabs:
                chunks.append(out[off + page * size:off + (page + 1) * size])
    return b"".join(chunks)


def diff(a: bytes, b: bytes, label: str) -> None:
    if len(a) != len(b):
        print(label, "len", len(a), len(b)); return
    n = sum(x != y for x, y in zip(a, b))
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), -1)
    print(f"{label}: diff={n} first={first}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blob", default=str(DEFAULT_BLOB))
    parser.add_argument("--header", default=str(DEFAULT_HDR))
    parser.add_argument("--guest", default=str(DEFAULT_GUEST))
    parser.add_argument("--bridge", default=str(DEFAULT_BRIDGE))
    parser.add_argument("--workdir", default=str(DEFAULT_WORK))
    args = parser.parse_args()
    blob = Path(args.blob)
    header = Path(args.header)
    guest = Path(args.guest)
    bridge = Path(args.bridge)
    work = Path(args.workdir)

    serve_s = Serve(blob, guest, bridge, header, work, debug=True)
    h = serve_s.h
    rt_s = Runtime(serve_s, max_new=4)
    rs = rt_s.add([1, 2, 3, 4, 5])
    rt_s.prefill(rs)
    s_pre = serve_s.last_out
    rt_s.decode([rs])
    s_dec = serve_s.last_out
    print("single prefill tokens", rs.generated)

    serve_b = Serve(blob, guest, bridge, header, work, debug=True)
    rt_b = Runtime(serve_b, max_new=4)
    rb = rt_b.add([1, 2, 3, 4, 5])
    rt_b.add(list(range(70, 100)))
    rt_b.add([11, 12, 13])
    rt_b.prefill(rb)
    b_pre = serve_b.last_out
    for r in rt_b.requests[1:]:
        rt_b.prefill(r)
    for r in rt_b.requests:
        print("item", r.rid, rt_b._item(r, 1))
    rt_b.decode(rt_b.requests)
    b_dec = serve_b.last_out
    print("batch prefill tokens", rb.generated)
    print("batch request 1 pages", rb.pages)

    # Compare only request 0's pages: the batch cache legitimately contains
    # extra requests (1 and 2), so a whole-KV diff would be non-zero by design.
    pages_s = list(rs.pages)
    pages_b = list(rt_b.requests[0].pages)
    print("req0 pages single/batch", pages_s, pages_b)
    diff(kv_pages(s_pre, h, pages_s), kv_pages(b_pre, h, pages_b),
         "prefill K/V req0")
    diff(kv_pages(s_dec, h, pages_s), kv_pages(b_dec, h, pages_b),
         "decode K/V req0")
    # Control/debug scratch now lives after kSCRATCH_END; read its addresses
    # from Serve instead of recomputing the (previously overlapping) layout.
    pt = serve_s.pt_off
    diff(s_pre[pt:pt + 16], b_pre[pt:pt + 16], "prefill ptab0")
    diff(s_dec[pt:pt + 16], b_dec[pt:pt + 16], "decode ptab0")
    lg = serve_s.log_off
    V = h["kVocab"]
    diff(s_dec[lg:lg + V * 4], b_dec[lg:lg + V * 4], "decode logits0 batch")
    sent = serve_s.sent_off
    # item 0 sentinel: [sum, first16, last16]
    diff(s_dec[sent:sent + 80 * 4], b_dec[sent:sent + 80 * 4], "sentinel item0")
    print("sent item0 xor", hex(struct.unpack_from("<I", b_dec, sent)[0]))
    snap = serve_s.snap_off
    for i in range(3):
        diff(s_dec[lg:lg + V * 4], b_dec[snap + i * V * 4:snap + (i + 1) * V * 4],
             f"snap{i} item0")
    ds = serve_s.dump_off
    db = serve_b.dump_off
    H = h["kHidden"]; QW = h["kHeads"] * h["kHeadDim"]; HS = H // 256
    offs = [0, H*4, H*8, H*8+QW, H*8+QW+h["kHeads"]*4, H*9+QW+h["kHeads"]*4]
    lens = [H*4, H*4, QW, h["kHeads"]*4, H, HS*4]
    names = ["x", "h", "q8", "qs", "h8", "hs"]
    for name, off, ln in zip(names, offs, lens):
        diff(s_dec[ds+off:ds+off+ln], b_dec[db+off:db+off+ln], f"item0 {name}")
    bs = serve_s.before_off; bb = serve_b.before_off
    diff(s_dec[bs:bs+V*4], b_dec[bb:bb+V*4], "item0 logits batch")
    diff(b_dec[bb:bb+V*4], b_dec[snap:snap+V*4], "before vs snap0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
