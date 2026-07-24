# ADR-0020: Pooled row lifecycle with off-image parking and byte-budgeted capacity

- Status: accepted (opt-in, default off)
- Date: 2026-07-22
- Task: FIT-021
- Related: ADR-0003 (normalized renderer), ADR-0007 (SSPL1 grammar), ADR-0010 (searchable axes),
  ADR-0017/0019 (mask containment), CORE-003 (clipped support), CORE-009 (background rows)

## Context

Every existing topology event (prune/split/relocate/adaptive growth) resizes the
`GaussianField`, rebuilds the optimizer via `_carry_adam_state`, and destroys donor rows that a
later event might have reused. There is no merge operator in the package; the validated
prototype (`deprecated_scripts/fit_janelle_complete_refinement.py`) had to implement count-neutral
merge+teleport *between* fits. Separately, deployment wants a hard per-file byte budget
(Gaussian payload + alpha map ≤ X bytes) rather than a Gaussian-count budget.

## Decision

1. **Fixed-capacity pool.** In pooled mode (`triage_every` set), parameter tensors are
   allocated once at fit entry at `capacity` rows and never resized. Row lifecycle is a `live`
   mask: topology events write rows in place and zero the touched rows' optimizer moments in
   place; the optimizer is never rebuilt. Destruction is deferred to one `subset(live)`
   compaction at the save/export boundary — a "pruned" row is just a parked donor that never
   found a destination.
2. **Off-image parking.** A parked row's mean is moved to a far off-image sentinel
   (`POOL_PARK_COORD`), scales collapsed to the minimum, opacity logit driven far negative.
   CORE-003 tile clipping then assigns it an empty support rectangle: exactly zero render
   contribution, zero gradient, zero tile elements. Parking requires no renderer change and no
   alive-mask branching in hot loops.
3. **Responsibility-gated parking.** Rows are parked only when their maximum normalized
   responsibility over their support is below a threshold. Summed activity (the legacy prune
   criterion) cannot see that a low-mass row is the *sole owner* of some pixels — under the
   normalized compositor such a row fully determines those pixels regardless of its opacity, so
   parking it punches a visible hole. Max responsibility is the correct function-preservation
   gate and falls out of the shared attribution pass at no extra cost.
4. **Byte-budgeted capacity.** `capacity` is derived from `target_file_bytes` through the
   SSPL1 *raw fixed-length* stream layout (bytes/row from the configured bit widths; 11 B at
   defaults with opacity), plus the compressed alpha payload (masked inputs), stream framing,
   and a header allowance. zlib stream coding only ever shrinks the file further and is treated
   as bonus margin, never counted on. The encoded file is asserted against the budget at save.
   `pool_capacity` may override the derivation directly.
5. **Merge is a first-class event op.** The prototype's envelope merge (mutual-nearest pair
   mining; midpoint mean; smallest generalized-eigen envelope of both translated covariances,
   inflated; amplitude-weighted color; union opacity; batch containment restore under
   `mask_contain`) moves into the package (`triage.py`), with the absorbed partner parked into
   the free list instead of being coupled to an immediate teleport.

## Alternatives rejected

- **Epsilon-opacity parking** (keep rows on-image with tiny opacity): not function-preserving
  at solely-owned pixels (normalized compositor), nonzero tile cost, and sigmoid never reaches
  zero.
- **Per-event reallocation with amortized doubling as the primary mechanism**: parameter memory
  is trivial (~36 B/row fp32) and the cap is almost always known (`target_file_bytes`,
  `max_gaussians`), so preallocation removes the rebuild path entirely; doubling remains a
  possible future fallback for a genuinely uncapped mode (deferred, FIT-021 notes).
- **Entropy-coded capacity derivation**: zlib/PNG sizes are content-dependent, so a hard byte
  guarantee would need iteration or slack anyway; raw fixed-length layout gives an exact,
  monotone `bytes(live_n)` and keeps the guarantee trivially checkable.
- **Alpha as a separate side-file**: the SSPL1 container is self-describing framed streams; one
  extra uniform stream keeps old decoders working (they read and ignore it) and keeps the
  budget a single-file statement.

## Consequences

- Pooled mode forbids the legacy independent timers (`split_every`/`relocate_every`/
  `prune_every`), `adaptive_count`, `mask_boundary_add_every`, fit-time QAT/`lambda_rate`,
  best-PSNR checkpoint selection, affine color basis, and non-normalized renderers (v1).
- `field.n` is the *capacity* during a pooled fit; anything rate- or count-bearing must use the
  live count. `fit()` compacts before returning, so downstream consumers see live rows only.
- Parked rows are exempt from `_MaskConstraint.apply` mean projection (otherwise containment
  would drag them back on-image); `apply` gained an `exempt` argument.
- Quantization ranges in the SSPL1 header are computed over encoded rows, so compaction before
  encode is mandatory (park sentinels would destroy the means range).
- Fixed tensor shapes make the pooled path CUDA-graph/`torch.compile` friendly and match the
  PORT-001/002 direction.
