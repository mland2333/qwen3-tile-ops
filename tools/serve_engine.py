#!/usr/bin/env python3
"""Host scheduler + page-budget runtime for the serve guest (route A).

This wires ``tools/engine.py``'s request/sampling model to the real guest call.
Admission reserves ``ceil((prompt + max_new) / 32)`` pages atomically, so a
request never holds a partial reservation and the 8-page pool cannot deadlock.
Requests that do not fit wait in a FIFO queue; finished requests release their
pages and are replaced by the next waiting request.
"""
from __future__ import annotations

from engine import (
    Batch,
    BatchItem,
    Engine,
    PagePool,
    Request,
    Sampler,
    Scheduler,
    State,
)
from run_serve import ITEMS, Serve


class ReservingScheduler(Scheduler):
    """FIFO admission with atomic page reservation."""

    def __init__(self, pool: PagePool, max_rows: int = 32,
                 max_items: int = ITEMS) -> None:
        super().__init__(pool, max_rows)
        self.max_items = max_items
        self.waiting: list[Request] = []
        self.active: list[Request] = []
        self.admitted: list[int] = []   # rids in admission order, for tests
        self.page_history: dict[int, list[int]] = {}

    def add(self, req: Request) -> None:
        self.requests.append(req)
        self.waiting.append(req)

    def finish(self, req: Request) -> None:
        if req.state != State.FINISHED:
            req.state = State.FINISHED
            self.page_history[req.rid] = list(req.pages)
            self.pool.release(req)
        if req in self.active:
            self.active.remove(req)

    def _reservation_tokens(self, req: Request) -> int:
        return req.n_prompt + max(0, req.params.max_new)

    def _admit(self) -> None:
        while self.waiting:
            req = self.waiting[0]
            need = ((self._reservation_tokens(req) + self.pool.page_tokens - 1)
                    // self.pool.page_tokens)
            if need > self.pool.num_pages:
                raise RuntimeError(
                    f"request {req.rid} needs {need} pages, pool has "
                    f"{self.pool.num_pages}")
            if not self.pool.reserve(req, self._reservation_tokens(req)):
                break
            self.waiting.pop(0)
            self.active.append(req)
            self.admitted.append(req.rid)
            req.state = State.PREFILL

    def build_batch(self) -> Batch:
        self._admit()
        items: list[BatchItem] = []
        rows = 0
        for req in list(self.active):
            if req.done:
                continue
            if len(items) >= self.max_items:
                break
            if req.n_fed < req.n_prompt:
                # K/V is stored with a single page/slot base per chunk, so a
                # prefill chunk must start on a 32-token page boundary and not
                # cross one.  Always take a full page except for the final
                # chunk; if it does not fit this batch's row budget, retry next
                # step rather than splitting at an arbitrary position.
                chunk = min(32, req.n_prompt - req.n_fed)
                if chunk > self.max_rows - rows:
                    continue
                items.append(BatchItem(
                    req, rows, chunk, req.n_fed,
                    req.prompt[req.n_fed:req.n_fed + chunk],
                    True, req.n_fed + chunk == req.n_prompt))
                rows += chunk
                req.state = State.PREFILL
                continue
            if len(req.generated) >= req.params.max_new:
                self.finish(req)
                continue
            if rows >= self.max_rows:
                continue
            if not req.generated:
                continue
            items.append(BatchItem(
                req, rows, 1, req.n_fed, [req.generated[-1]], False, True))
            rows += 1
            req.state = State.DECODE
        return Batch(items, rows)


def make_guest_step(serve: Serve):
    """Turn one Batch into one Serve.run (route A: guest runs items in order)."""
    def guest_step(batch: Batch) -> dict[int, list[float]]:
        if not batch.items:
            return {}
        if len(batch.items) > ITEMS:
            raise RuntimeError(
                f"batch has {len(batch.items)} items, ABI max is {ITEMS}")
        items = []
        page_tables = []
        ids = []
        wants = []
        for slot, item in enumerate(batch.items):
            req = item.req
            items.append((
                item.rows,
                item.pos,
                serve.pt_off + slot * 16,
                serve.log_off + slot * serve.V * 4,
                serve.ids_off + slot * serve.ids_stride,
            ))
            page_tables.append(req.pages)
            ids.append((item.pos, item.token_ids))
            wants.append(item.needs_logits)
        outs = serve.run(items, page_tables, ids, wants)
        return {
            item.req.rid: logits
            for item, logits in zip(batch.items, outs)
            if item.needs_logits
        }
    return guest_step


class ServeEngine(Engine):
    def __init__(self, serve: Serve, num_pages: int = 8,
                 page_tokens: int = 32, max_rows: int = 32) -> None:
        self.serve = serve
        self.pool = PagePool(num_pages, page_tokens)
        self.sched = ReservingScheduler(self.pool, max_rows, max_items=ITEMS)
        self.sampler = Sampler()
        self.guest_step = make_guest_step(serve)
        self.vocab = serve.V
        self.next_rid = 0
        self.steps = 0
