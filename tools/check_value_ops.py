#!/usr/bin/env python3
"""Run tests/guest/value_ops.cpp on the simulator and diff value vs in-place.

Every case stores the value form and the explicit-copy in-place form next to
each other; the two must be bit identical (same instruction, only D != S), and
the arithmetic cases are also checked against numpy.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ROWS, COLS, LANES = 32, 32, 32
ACC_BYTES = ROWS * COLS * 4
SLOT = 0x1000
OUT_OFF = 0x4000
LOG2E = np.uint32(0x3FB8AA3B).view(np.float32)

CASES = [
    "acc add", "acc sub", "acc mul", "acc exp",
    "acc mul byrow", "acc mul bycol", "acc sub byrow",
    "acc mul scalar", "acc add scalar",
    "vec add", "vec sub", "vec mul", "vec exp", "vec mul scalar",
    "vec max", "vec rcp", "vec rsqrt", "acc rcp", "tile dequant",
]
APPROX = {"acc exp", "vec exp", "vec rsqrt"}


def inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.linspace(-2.0, 2.0, ROWS * COLS, dtype=np.float32).reshape(ROWS, COLS)
    b = ((np.arange(ROWS * COLS, dtype=np.float32) % 7.0) - 3.0).reshape(ROWS, COLS) * np.float32(0.5)
    v = np.linspace(-1.0, 1.0, LANES, dtype=np.float32) + np.float32(0.25)
    w = (np.arange(LANES, dtype=np.float32) * np.float32(0.25)) - np.float32(1.0)
    return a, b, v, w


def tile_i8() -> np.ndarray:
    vals = ((np.arange(ROWS * COLS) * 37) % 251) - 125   # spread over int8
    return vals.astype(np.int8).reshape(ROWS, COLS)


def expected(a, b, v, w) -> list[np.ndarray]:
    # Order matches CASES / the guest: col=true means "byrow" (one scalar per row).
    out = [
        (a + b).astype(np.float32),
        (a - b).astype(np.float32),
        (a * b).astype(np.float32),
        np.exp2((a * LOG2E).astype(np.float32)).astype(np.float32),
        (a * v[:, None]).astype(np.float32),    # byrow
        (a * v[None, :]).astype(np.float32),    # bycol
        (a - v[:, None]).astype(np.float32),    # byrow
        (a * np.float32(0.5)).astype(np.float32),
        (a + np.float32(1.5)).astype(np.float32),
        (v + w).astype(np.float32),
        (v - w).astype(np.float32),
        (v * w).astype(np.float32),
        np.exp2((v * LOG2E).astype(np.float32)).astype(np.float32),
        (v * np.float32(0.25)).astype(np.float32),
        np.maximum(v, w).astype(np.float32),
        (np.float32(1.0) / v).astype(np.float32),
        (np.float32(1.0) / np.sqrt(v.astype(np.float32))).astype(np.float32),
        (np.float32(1.0) / a).astype(np.float32),
        (tile_i8().astype(np.float32) * v[:, None]).astype(np.float32),
    ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guest", default=str(ROOT / "build" / "value_ops.bin"))
    ap.add_argument("--bridge", default=str(ROOT / "build" / "sim_bridge"))
    ap.add_argument("--blob", default=str(ROOT / "build" / "value_ops_data.bin"))
    ap.add_argument("--out", default=str(ROOT / "build" / "value_ops_out.bin"))
    ap.add_argument("--tile-off", type=lambda x: int(x, 0), default=0x2200)
    args = ap.parse_args()

    a, b, v, w = inputs()
    size = OUT_OFF + 2 * len(CASES) * 0x1000
    blob = bytearray(size)
    blob[0x0000:0x0000 + ACC_BYTES] = a.tobytes()
    blob[0x1000:0x1000 + ACC_BYTES] = b.tobytes()
    blob[0x2000:0x2000 + LANES * 4] = v.tobytes()
    blob[0x2100:0x2100 + LANES * 4] = w.tobytes()
    tile = tile_i8()
    blob[args.tile_off:args.tile_off + tile.nbytes] = tile.tobytes()
    Path(args.blob).write_bytes(bytes(blob))

    result = subprocess.run([args.bridge, args.guest, args.blob, args.out],
                            capture_output=True, text=True)
    print(result.stderr.strip())
    if result.returncode != 0:
        print("guest failed", file=sys.stderr)
        return 1

    data = np.frombuffer(Path(args.out).read_bytes(), dtype=np.uint8)
    exp = expected(a, b, v, w)
    failures = 0
    for idx, name in enumerate(CASES):
        value_at = OUT_OFF + 2 * idx * SLOT
        copy_at = OUT_OFF + (2 * idx + 1) * SLOT
        is_vec = name.startswith("vec")
        shape = (LANES,) if is_vec else (ROWS, COLS)
        nbytes = LANES * 4 if is_vec else ACC_BYTES
        got_value = data[value_at:value_at + nbytes].view(np.float32).reshape(shape)
        got_copy = data[copy_at:copy_at + nbytes].view(np.float32).reshape(shape)
        same = np.array_equal(got_value.view(np.uint32), got_copy.view(np.uint32))
        want = exp[idx]
        if name in APPROX:
            note = f" numpy max|d|={np.nanmax(np.abs(got_value.astype(np.float64) - want.astype(np.float64))):.3g}"
            numpy_ok = bool(np.allclose(got_value, want, rtol=1e-6, atol=1e-6, equal_nan=True))
        else:
            numpy_ok = np.array_equal(got_value.view(np.uint32), want.view(np.uint32))
            note = " numpy exact" if numpy_ok else " NUMPY-MISMATCH"
        ok = same and numpy_ok
        if not ok:
            failures += 1
        print(f"{'ok  ' if ok else 'FAIL'} {name:14s} value==inplace {same}{note}")

    print(f"{failures} case(s) differ")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
