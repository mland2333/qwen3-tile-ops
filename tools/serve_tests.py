#!/usr/bin/env python3
"""Route-A serve acceptance tests: cross-page KV, scheduler, page budget."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from debug_serve import kv_pages
from engine import SamplingParams
from run_serve import Runtime, Serve
from serve_engine import ServeEngine


def mismatches(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        return -1
    return sum(x != y for x, y in zip(a, b))


def _engine(serve: Serve, num_pages: int = 8, free: list[int] | None = None) -> ServeEngine:
    engine = ServeEngine(serve, num_pages=num_pages)
    if free is not None:
        engine.pool.free = list(free)
    return engine


def _run(serve: Serve, prompts: list[list[int]], max_new: int,
         num_pages: int = 8, free: list[int] | None = None) -> ServeEngine:
    engine = _engine(serve, num_pages, free)
    for prompt in prompts:
        engine.add_request(prompt, SamplingParams(max_new=max_new))
    engine.run()
    return engine


def test_long_kv(serve: Serve) -> None:
    """40-token prompt: prefill spans two pages, decode enters page 2."""
    prompt = list(range(100, 140))
    others = [list(range(200, 240)), [11, 12, 13]]
    # PagePool pops from the tail; this order hands req0 pages 0 and 1.
    free = [7, 6, 5, 4, 3, 2, 1, 0]

    batch = Runtime(serve, num_pages=8, max_new=4)
    batch.pool.free = list(free)
    rb = batch.add(prompt)
    for other in others:
        batch.add(other)
    batch.prefill(rb)
    b_pre = serve.last_out
    for r in batch.requests[1:]:
        batch.prefill(r)
    batch.decode([rb])
    b_dec = serve.last_out

    single = Runtime(serve, num_pages=8, max_new=4)
    single.pool.free = list(free)
    rs = single.add(prompt)
    single.prefill(rs)
    s_pre = serve.last_out
    single.decode([rs])
    s_dec = serve.last_out

    assert rb.pages == rs.pages == [0, 1], (rb.pages, rs.pages)
    assert len(rb.pages) == 2 and len(rs.pages) == 2
    h = serve.h
    assert mismatches(kv_pages(s_pre, h, rs.pages),
                      kv_pages(b_pre, h, rb.pages)) == 0, "prefill K/V"
    assert mismatches(kv_pages(s_dec, h, rs.pages),
                      kv_pages(b_dec, h, rb.pages)) == 0, "decode K/V"
    pt = serve.pt_off
    assert mismatches(s_pre[pt:pt + 16], b_pre[pt:pt + 16]) == 0, "page table"
    print("ok   long KV: req0 pages", rb.pages, "cross-page prefill/decode diff=0")


def test_scheduler_long(serve: Serve, max_new: int = 20) -> None:
    """Batch scheduler with 3 concurrent 40-token requests and long decode."""
    prompts = [list(range(100, 140)), list(range(200, 240)), [11, 12, 13]]
    batch = _run(serve, prompts, max_new)
    singles = [_run(serve, [prompt], max_new).outputs()[0] for prompt in prompts]
    got = batch.outputs()
    assert [got[i] for i in range(len(prompts))] == singles, (got, singles)
    assert batch.pool.free_pages == 8, batch.pool.free
    for rid, pages in batch.sched.page_history.items():
        if rid < 2:
            assert len(pages) >= 2, (rid, pages)
    print("ok   scheduler long: batch == single, pages released",
          len(batch.sched.page_history), "requests")


def test_page_exhaustion(serve: Serve) -> None:
    """3 requests x 3 pages on an 8-page pool: third waits, then reuses pages."""
    prompt = list(range(100, 165))       # 65 tokens + 4 new = 3 pages
    max_new = 4
    engine = _engine(serve)
    reqs = [engine.add_request(prompt, SamplingParams(max_new=max_new))
            for _ in range(3)]
    assert len(engine.sched.waiting) == 3
    assert engine.step() is True
    assert len(engine.sched.active) == 2, "only 2 requests fit (6 pages)"
    assert len(engine.sched.waiting) == 1, "third request must wait"
    while engine.step():
        pass
    assert all(len(r.generated) >= max_new for r in reqs), [r.generated for r in reqs]
    assert engine.sched.admitted == [0, 1, 2], engine.sched.admitted
    assert engine.pool.free_pages == 8, engine.pool.free
    for r in reqs:
        assert len(engine.sched.page_history[r.rid]) == 3, engine.sched.page_history

    for r in reqs:
        single = _run(serve, [prompt], max_new)
        assert single.outputs()[0] == r.generated, (r.rid, r.generated,
                                                    single.outputs()[0])

    # A later request must be able to reuse the released pages.
    fresh = engine.add_request([1, 2, 3], SamplingParams(max_new=2))
    while engine.step():
        pass
    assert len(fresh.generated) >= 2, fresh.generated
    assert engine.pool.free_pages == 8, engine.pool.free
    print("ok   page exhaustion: wait/reuse, batch == single, free=8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blob", default="build/full/qwen3_big_data.bin")
    parser.add_argument("--header", default="build/full/model_big.h")
    parser.add_argument("--guest", default="build/serve.bin")
    parser.add_argument("--bridge", default="build/sim_bridge")
    parser.add_argument("--workdir", default="build/serve_tests")
    parser.add_argument("--case", choices=("all", "long", "pages"), default="all")
    parser.add_argument("--max-new", type=int, default=20)
    args = parser.parse_args()
    serve = Serve(Path(args.blob), Path(args.guest), Path(args.bridge),
                  Path(args.header), Path(args.workdir))
    if args.case in ("all", "long"):
        test_long_kv(serve)
        test_scheduler_long(serve, max_new=args.max_new)
    if args.case in ("all", "pages"):
        test_page_exhaustion(serve)
    print("serve tests ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
