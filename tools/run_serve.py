#!/usr/bin/env python3
"""Host-side multi-request runtime for the serve guest.

Prefill chunks and decode steps call build/serve.bin.  The serve guest processes
items sequentially (forward_chunk per item) but the host side has full request
scheduling, page tables, KV page pool, and sampling.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import PagePool, SamplingParams, Sampler
from serve_layout import ITEMS, layout, read_header


@dataclass
class Req:
    rid: int
    prompt: list[int]
    params: SamplingParams
    generated: list[int] = field(default_factory=list)
    n_fed: int = 0
    pages: list[int] = field(default_factory=list)
    state: str = "waiting"
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.params.seed)

    @property
    def done(self) -> bool:
        return self.state == "done"


class Serve:
    def __init__(self, model_blob: Path, guest: Path, bridge: Path, header: Path,
                 work: Path, debug: bool = False) -> None:
        self.h = read_header(header)
        self.V = self.h["kVocab"]
        self.debug = debug
        # Single source of truth: tools/serve_layout.py generates the guest's
        # C++ header from the same formulas.
        self.layout = layout(self.h, debug=debug)
        self.cmd_off = self.layout["cmd"]
        self.pt_off = self.layout["pt"]
        self.abi_off = self.layout["abi"]
        self.log_off = self.layout["logits"]
        self.ids_off = self.layout["ids"]
        self.ids_stride = 32 * 4
        self.size = self.layout["end"]
        self.dump_stride = self.layout.get("dump_stride", 0)
        for name in ("sentinel", "snap", "dump", "before"):
            setattr(self, f"{name}_off", self.layout.get(name))
        self.sent_off = self.layout.get("sentinel")
        self.model = bytearray(model_blob.read_bytes())
        if len(self.model) < self.h["kSCRATCH_END"]:
            raise ValueError(
                f"model blob is {len(self.model)} B, header kSCRATCH_END is "
                f"{self.h['kSCRATCH_END']} B")
        if len(self.model) < self.size:
            self.model.extend(bytes(self.size - len(self.model)))
        if debug and not Path(guest).name.endswith("debug.bin"):
            raise ValueError("debug layout requires the YAI_SERVE_DEBUG guest")
        self.guest = guest
        self.bridge = bridge
        self.work = work
        self.work.mkdir(parents=True, exist_ok=True)

    def run(self, items, page_tables: list[list[int]], ids: list[tuple[int, list[int]]],
            wants: list[bool]) -> list[list[float]]:
        if len(items) > ITEMS:
            raise ValueError(f"serve batch has {len(items)} items, max {ITEMS}")
        blob = bytearray(self.model)
        n = len(items)
        struct.pack_into("<I", blob, self.cmd_off, n)
        logits_offs = []
        for i, ((rows, pos, ptab_i, logits_i, ids_i), ptab, (_id_pos, tok), want) in enumerate(
                zip(items, page_tables, ids, wants)):
            if rows == 0 or rows > 32:
                raise ValueError(f"item rows={rows}, expected 1..32")
            if rows > 1 and pos % 32 != 0:
                raise ValueError(
                    f"prefill item rows={rows} pos={pos}: chunks must start "
                    "on a 32-token page boundary")
            base = self.cmd_off + 4 + i * 24
            struct.pack_into("<6I", blob, base, rows, pos, ptab_i, logits_i,
                             int(want), ids_i)
            if len(ptab) > 16:
                raise ValueError(f"page table has {len(ptab)} pages, max 16")
            for j, p in enumerate(ptab):
                struct.pack_into("<H", blob, ptab_i + j * 2, p)
            if len(tok) > 32:
                raise ValueError(f"item has {len(tok)} tokens, max 32")
            for j, t in enumerate(tok):
                struct.pack_into("<I", blob, ids_i + j * 4, t)
            logits_offs.append(logits_i)
        blob_p = self.work / "serve_blob.bin"
        out_p = self.work / "serve_out.bin"
        self.work.mkdir(parents=True, exist_ok=True)
        blob_p.write_bytes(blob)
        r = subprocess.run([str(self.bridge), str(self.guest), str(blob_p), str(out_p)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr or "serve guest failed")
        out = out_p.read_bytes()
        guest_end = struct.unpack_from("<I", out, self.abi_off)[0]
        if guest_end != self.size:
            raise RuntimeError(
                f"serve ABI mismatch: guest scratch end {guest_end}, host "
                f"{self.size}; rebuild {self.guest} from the same header")
        self.last_out = out
        res = []
        for i, want in enumerate(wants):
            if not want:
                res.append([])
                continue
            off = logits_offs[i]
            res.append(list(struct.unpack_from(f"<{self.V}f", out, off)))
        return res


class Runtime:
    def __init__(self, serve: Serve, num_pages: int = 8, max_new: int = 8,
                 temperature: float = 0.0) -> None:
        self.serve = serve
        self.pool = PagePool(num_pages, 32)
        self.sampler = Sampler()
        self.requests: list[Req] = []
        self.next_rid = 0
        self.max_new = max_new
        self.temperature = temperature

    def add(self, prompt: list[int]) -> Req:
        r = Req(self.next_rid, list(prompt), SamplingParams(
            max_new=self.max_new, temperature=self.temperature))
        self.next_rid += 1
        self.requests.append(r)
        return r

    def _item(self, r: Req, rows: int) -> tuple:
        pos = r.n_fed
        slot = self.requests.index(r)
        if slot >= ITEMS:
            raise RuntimeError(f"request slot {slot} exceeds ABI max {ITEMS}")
        if not self.pool.ensure(r, pos + rows):
            raise RuntimeError("page pool exhausted")
        return (rows, pos, self.serve.pt_off + slot * 16,
                self.serve.log_off + slot * self.serve.V * 4,
                self.serve.ids_off + slot * self.serve.ids_stride)

    def prefill(self, r: Req) -> None:
        p = 0
        while p < len(r.prompt):
            r.n_fed = p
            rows = min(32, len(r.prompt) - p)
            last = p + rows == len(r.prompt)
            item = self._item(r, rows)
            out = self.serve.run([item], [r.pages], [(p, r.prompt[p:p + rows])],
                                 [last])
            p += rows
            if last:
                tok = self.sampler.sample(out[0], r)
                r.generated.append(tok)
                r.state = "decode"
        r.n_fed = len(r.prompt)

    def decode(self, reqs: list[Req]) -> None:
        items, pts, ids, wants = [], [], [], []
        for r in reqs:
            item = self._item(r, 1)
            items.append(item)
            pts.append(r.pages)
            ids.append((r.n_fed, [r.generated[-1]]))
            wants.append(True)
        outs = self.serve.run(items, pts, ids, wants)
        for r, logits in zip(reqs, outs):
            r.n_fed += 1
            tok = self.sampler.sample(logits, r)
            r.generated.append(tok)
            if len(r.generated) >= r.params.max_new or tok in r.params.stop_tokens:
                r.state = "done"
                self.pool.release(r)

    def run(self, prompts: list[list[int]]) -> list[list[int]]:
        reqs = [self.add(p) for p in prompts]
        for r in reqs:
            self.prefill(r)
        while any(not r.done for r in reqs):
            active = [r for r in reqs if not r.done]
            self.decode(active)
        return [r.generated for r in reqs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", default="build/full/qwen3_big_data.bin")
    ap.add_argument("--header", default="build/full/model_big.h")
    ap.add_argument("--guest", default="build/serve.bin")
    ap.add_argument("--bridge", default="build/sim_bridge")
    ap.add_argument("--workdir", default="build/serve_work")
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    serve = Serve(Path(a.blob), Path(a.guest), Path(a.bridge), Path(a.header),
                  Path(a.workdir))
    prompts = [[1, 2, 3, 4, 5], list(range(70, 100)), [11, 12, 13]]
    if a.test:
        batch = Runtime(serve, max_new=a.max_new).run(prompts)
        singles = [Runtime(serve, max_new=a.max_new).run([p])[0] for p in prompts]
        ok = batch == singles
        print("batch:", batch)
        print("single:", singles)
        print("ok" if ok else "MISMATCH")
        return 0 if ok else 1
    out = Runtime(serve, max_new=a.max_new).run(prompts)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
