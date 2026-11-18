# Route B: strict batch QKV/O/MLP (design only)

Status: **not implemented**.  The current serve guest is route A
(host-batched, guest-sequential): `Serve.run` packs up to 8 items, but
`src/serve.cpp` still calls `forward_chunk` once per item.  This document is
the separate route-B design task and is intentionally not mixed into the
route-A cleanup/tests commit.

## Goal

One guest call per scheduler step, with a shared row batch:

- one batch RMSNorm / QKV / O / MLP over the concatenated rows;
- per-request paged attention segments;
- batch K/V scatter into per-request pages;
- per-request lm_head for the requests that need logits;
- no debug buffers in the production guest.

Total rows stay within the existing `kSeq` activation buffers (32 for the
current model).

## ABI

Replace the per-item `ServeItem` loop with a batch descriptor:

```cpp
struct BatchItem {
  uint32_t row_start;   // first activation row
  uint32_t rows;        // contiguous rows for this request
  uint32_t pos;         // absolute position of row_start
  uint32_t ptab;        // blob offset of this request's page table
  uint32_t logits;      // blob offset of this request's logits slot
  uint32_t ids;         // blob offset of this request's token-id slot
  uint32_t needs;       // write logits (last prefill chunk / decode)
};
struct BatchDesc {
  uint32_t n;
  BatchItem items[8];
};
```

The host `ReservingScheduler` already produces exactly this information; the
route-B guest consumes it as a row map.

Keep the generated `serve_layout.py` ABI handshake (cmd / page tables /
logits / ids / end) and extend it with any route-B tables.  Host and guest must
continue to share one layout source.

## Data path

1. **Embedding + norm**: gather token ids per item slot, write `x[0..rows)` and
   `h[0..rows)`; the kernels are already row-wise.
2. **QKV**: one `gemm_nt_group_store` over `h8[0..rows)` for Q, K and V.  Q
   norm/rope/quant epilogue is per row, so it works unchanged.
3. **K/V store**: the current epilogue writes one contiguous page/slot range
   per chunk.  For batch rows, first store K/V to contiguous staging
   `[rows][kv_dim]` + scales, then call `paged_kv_scatter_batch` with the
   per-row table pointer and absolute token position.  This reuses the
   committed batch scatter helper and removes the page-alignment constraint.
4. **Attention**: call `attention_heads_gqa_tokscale_paged` once per request
   segment (`row_start..row_start+rows`) using that request's page table and
   `kv_len = pos + rows`.  Rows of one request must be contiguous in the
   activation buffer.  A later optimization can fuse segments into one kernel.
5. **O / MLP**: one `gemm_nt_group_store` over the full row range, with the
   residual add against `x[0..rows)`.  Row scales stay per row, so no layout
   change is needed.
6. **lm_head**: for each item with `needs`, run the existing per-row
   final-norm + lm_head into `logits[item]`.  Decode batches usually need one
   row per request; prefill batches need only the last row.

## Scheduling

Reuse the route-A `ReservingScheduler`:

- FIFO admission; reserve `ceil((prompt + max_new) / 32)` pages atomically;
- prefill in 32-token chunks (page aligned); decode in one-row items;
- request completion releases pages and admits the next waiting request.

Route B removes the route-A constraint that a prefill chunk must start on a
page boundary, because K/V is scattered per row.

## Milestones and acceptance

- **B0**: `BatchDesc` + row map; host encodes it; guest parses and validates
  (`n <= 8`, `row_start + rows <= 32`, non-overlapping rows).
- **B1**: batch embedding/QKV.  Compare `h8/hs/q8/qs/k/v` per row against
  route A/single.
- **B2**: batch K/V scatter.  Compare every request's physical page K/V and
  scales against single-request runs; cover page 0/1/7 and cross-page prompts.
- **B3**: per-request attention segments.  Compare `o8/os` per row; cover
  different `kv_len` and causal positions in one batch.
- **B4**: batch O/MLP and per-request lm_head.  Compare `x`, `logits[item]`.
- **B5**: end-to-end `make serve-b-test`, `serve-b-test-long`,
  `serve-b-test-pages`; batch tokens == single tokens, page table and K/V
  equal, no debug writes in production.

Add `YAI_SERVE_BATCH_DEBUG` only for stage dumps; production stays clean.

## Risks

- Per-row K/V scatter adds a staging write/read; measure before fusing into
  the GEMM epilogue.
- Attention per request is not cross-request fused; decode batches with many
  requests pay one attention call per request.
- Row ranges must stay contiguous; a scheduler that interleaves requests
  would need an explicit row map in every kernel.
- Numerical comparison should be bit-exact where the same per-row math is
  used; attention segment boundaries must not change each request's K/V or
  causal mask.
