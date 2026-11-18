#!/usr/bin/env python3
"""Host-side engine scaffolding: paged KV pool, continuous-batching scheduler,
and sampler.  The NPU guest is called through a `guest_step(batch)` callback;
the production guest will consume the same BatchItem fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Sequence

import numpy as np


class State(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass
class SamplingParams:
    max_new: int = 16
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 0
    stop_tokens: set[int] = field(default_factory=set)


@dataclass
class Request:
    rid: int
    prompt: list[int]
    params: SamplingParams
    generated: list[int] = field(default_factory=list)
    n_fed: int = 0              # tokens whose K/V are already in the cache
    pages: list[int] = field(default_factory=list)
    state: State = State.WAITING
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.params.seed)

    @property
    def n_prompt(self) -> int:
        return len(self.prompt)

    @property
    def done(self) -> bool:
        return self.state == State.FINISHED

    @property
    def needs_logits(self) -> bool:
        return self.state in (State.PREFILL, State.DECODE)


@dataclass
class BatchItem:
    req: Request
    row_start: int
    rows: int
    pos: int                    # n_fed at batch build
    token_ids: list[int]
    prefill: bool
    needs_logits: bool


@dataclass
class Batch:
    items: list[BatchItem]
    total_rows: int

    @property
    def requests(self) -> list[Request]:
        return [it.req for it in self.items]


class PagePool:
    """32-token pages; page table per request."""

    def __init__(self, num_pages: int, page_tokens: int = 32) -> None:
        self.num_pages = num_pages
        self.page_tokens = page_tokens
        self.free = list(range(num_pages))
        self.used: set[int] = set()

    def alloc_page(self) -> int | None:
        if not self.free:
            return None
        p = self.free.pop()
        self.used.add(p)
        return p

    def ensure(self, req: Request, n_tokens: int) -> bool:
        need = (n_tokens + self.page_tokens - 1) // self.page_tokens
        while len(req.pages) < need:
            p = self.alloc_page()
            if p is None:
                return False
            req.pages.append(p)
        return True

    def release(self, req: Request) -> None:
        for p in req.pages:
            if p in self.used:
                self.used.remove(p)
                self.free.append(p)
        req.pages.clear()

    @property
    def free_pages(self) -> int:
        return len(self.free)


class Sampler:
    def sample(self, logits: Sequence[float], req: Request) -> int:
        p = req.params
        x = np.asarray(logits, dtype=np.float64).copy()
        if p.temperature <= 0.0:
            return int(np.argmax(x))
        x /= p.temperature
        if p.repetition_penalty != 1.0:
            for t in set(req.generated):
                if x[t] > 0:
                    x[t] /= p.repetition_penalty
                else:
                    x[t] *= p.repetition_penalty
        if p.top_k > 0 and p.top_k < x.size:
            kth = np.partition(x, -p.top_k)[-p.top_k]
            x[x < kth] = -np.inf
        if p.top_p < 1.0:
            order = np.argsort(-x)
            sorted_x = x[order]
            probs = np.exp(sorted_x - sorted_x[0])
            probs /= probs.sum()
            cum = np.cumsum(probs)
            keep = cum <= p.top_p
            if not keep.any():
                keep[0] = True
            keep[0] = True
            drop = order[~keep]
            x[drop] = -np.inf
        m = np.max(x)
        probs = np.exp(x - m)
        probs /= probs.sum()
        return int(req.rng.choice(x.size, p=probs))


class Scheduler:
    def __init__(self, pool: PagePool, max_rows: int = 32) -> None:
        self.pool = pool
        self.max_rows = max_rows
        self.requests: list[Request] = []

    def add(self, req: Request) -> None:
        self.requests.append(req)

    def finish(self, req: Request) -> None:
        if req.state != State.FINISHED:
            req.state = State.FINISHED
            self.pool.release(req)

    def build_batch(self) -> Batch:
        items: list[BatchItem] = []
        rows = 0
        # keep admission order; prefill chunks first, then decode
        for req in self.requests:
            if req.done:
                continue
            if req.n_fed < req.n_prompt:
                if rows >= self.max_rows:
                    continue
                chunk = min(32, req.n_prompt - req.n_fed, self.max_rows - rows)
                if chunk <= 0:
                    continue
                if not self.pool.ensure(req, req.n_fed + chunk):
                    continue
                last = req.n_fed + chunk == req.n_prompt
                items.append(BatchItem(req, rows, chunk, req.n_fed,
                                       req.prompt[req.n_fed:req.n_fed + chunk],
                                       True, last))
                rows += chunk
                continue
            # decode: feed the last generated token
            if len(req.generated) >= req.params.max_new:
                self.finish(req)
                continue
            if rows >= self.max_rows:
                continue
            if not req.generated:
                continue
            if not self.pool.ensure(req, req.n_fed + 1):
                continue
            items.append(BatchItem(req, rows, 1, req.n_fed,
                                   [req.generated[-1]], False, True))
            rows += 1
        return Batch(items, rows)


class Engine:
    def __init__(self, guest_step: Callable[[Batch], dict[int, Sequence[float]]],
                 num_pages: int = 256, page_tokens: int = 32,
                 max_rows: int = 32, vocab: int = 151936) -> None:
        self.pool = PagePool(num_pages, page_tokens)
        self.sched = Scheduler(self.pool, max_rows)
        self.sampler = Sampler()
        self.guest_step = guest_step
        self.vocab = vocab
        self.next_rid = 0
        self.steps = 0

    def add_request(self, prompt: Sequence[int], params: SamplingParams | None = None) -> Request:
        req = Request(self.next_rid, list(prompt), params or SamplingParams())
        self.next_rid += 1
        self.sched.add(req)
        return req

    def step(self) -> bool:
        batch = self.sched.build_batch()
        if not batch.items:
            # finish any request that reached max_new
            for req in self.sched.requests:
                if not req.done and req.n_fed >= req.n_prompt and len(req.generated) >= req.params.max_new:
                    self.sched.finish(req)
            return False
        logits = self.guest_step(batch)
        for it in batch.items:
            req = it.req
            req.n_fed += it.rows
            if it.needs_logits:
                tok = self.sampler.sample(logits[req.rid], req)
                req.generated.append(tok)
                if tok in req.params.stop_tokens or len(req.generated) >= req.params.max_new:
                    self.sched.finish(req)
                else:
                    req.state = State.DECODE
            elif req.n_fed < req.n_prompt:
                req.state = State.PREFILL
        self.steps += 1
        return True

    def run(self, max_steps: int = 10000) -> None:
        while self.steps < max_steps and self.step():
            pass

    def outputs(self) -> dict[int, list[int]]:
        return {r.rid: list(r.generated) for r in self.sched.requests}
