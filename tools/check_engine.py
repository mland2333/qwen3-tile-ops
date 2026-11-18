#!/usr/bin/env python3
"""Unit test for the host engine scaffolding (mock guest logits)."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Batch, Engine, SamplingParams


def mock_guest(batch: Batch) -> dict[int, np.ndarray]:
    out = {}
    for it in batch.items:
        if it.needs_logits:
            x = np.full(32, -100.0, dtype=np.float64)
            # deterministic token per request/step
            tok = (it.req.rid * 7 + len(it.req.generated) + it.rows) % 32
            x[tok] = 100.0
            out[it.req.rid] = x
    return out


def main() -> int:
    eng = Engine(mock_guest, num_pages=8, max_rows=32, vocab=32)
    a = eng.add_request([1, 2, 3, 4, 5], SamplingParams(max_new=4, temperature=0))
    b = eng.add_request(list(range(70)), SamplingParams(max_new=4, temperature=0))
    c = eng.add_request(list(range(33)), SamplingParams(max_new=4, temperature=0))
    eng.run(max_steps=100)
    outs = eng.outputs()
    assert len(outs[a.rid]) == 4, outs
    assert len(outs[b.rid]) == 4, outs
    assert len(outs[c.rid]) == 4, outs
    assert eng.pool.free_pages == 8, eng.pool.free_pages
    assert a.done and b.done and c.done
    print("ok host engine: 3 requests, chunked prefill, decode, pages freed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
