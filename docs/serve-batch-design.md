# Route B: strict batch QKV/O/MLP

Status: **implemented** in `src/serve_batch.cpp` as `build/serve_batch.bin`.
The route-A guest (`src/serve.cpp`) remains the sequential reference.  Both
use the same `ServeCmd` ABI and the generated `serve_layout.h`.

## Implementation notes

- The host ABI is unchanged: an item's `rows` are packed in order, so the
  guest derives `row_start` by summing rows.  The host scheduler therefore
  needed no route-B-specific encoding.
- `ServeItem` carries per-item `ptab`, `logits` and `ids` blobs; the guest
  gathers a contiguous `[total_rows][1]` id vector for one batch embedding.
- Q/K rope uses a gathered per-row cos/sin scratch (`kCos`/`kSin`), because
  rows in one batch have different absolute positions.
- K/V are written to contiguous `w.k8/v8/ks/vs` staging and then scattered
  with `paged_kv_scatter_batch`, one table/token per row.
- Attention is one `attention_heads_gqa_tokscale_paged` call per request
  segment; rows of a request are contiguous.
- O and MLP are one GEMM over all rows with the residual add against the
  batch `x` rows.  lm_head runs once per item that needs logits, into that
  item's logits slot.
- Production has no debug buffers.  `total_rows <= kSeq` is enforced by the
  guest; the host scheduler already caps the batch at 32 rows.

## Verification

`make serve-batch-test`, `make serve-batch-test-long`, and
`make serve-batch-test-pages` all pass on the full model; the short test
produces the same token sequences as route A and as single-request runs.

## Goal

One guest call per scheduler step, with a shared row batch:

- one batch RMSNorm / QKV / O / MLP over the concatenated rows;
- per-request paged attention segments;
- batch K/V scatter into per-request pages;
- per-request lm_head for the requests that need logits;
- no debug buffers in the production guest.

Total rows stay within the existing `kSeq` activation buffers (32 for the
current model).

## ABI (implemented)

Route B reuses the route-A `ServeCmd` unchanged:

```cpp
struct ServeItem {
  uint32_t rows;
  uint32_t pos;
  uint32_t ptab;
  uint32_t logits;
  uint32_t needs;
  uint32_t ids;
};
struct ServeCmd {
  uint32_t n;
  ServeItem items[8];
};
```

The guest derives `row_start[0..n)` by summing `rows` in order; no separate
`BatchDesc` allocation or host encoding is needed.  The generated
`serve_layout.py` ABI handshake is shared by both guests (cmd / page tables /
logits / ids / batch cos/sin / end).

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

Route B's K/V scatter is per row and does not itself require page alignment;
the current host scheduler still uses page-aligned 32-token prefill chunks.

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
- **B5**: end-to-end `make serve-batch-test`, `make serve-batch-test-long`,
  `make serve-batch-test-pages`; batch tokens == single tokens, page table and
  K/V equal, no debug writes in production.  *Implemented and verified on the
  full model.*

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
