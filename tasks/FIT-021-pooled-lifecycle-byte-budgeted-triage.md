# FIT-021: Pooled row lifecycle, byte-budgeted capacity, and error-triage events

## Context

The fitter's topology operators — sampled-add spawning (FIT-004/017), duplicate/moment splits
(FIT-007), relocation (FIT-004), activity pruning (FIT-002) — each fire on an independent timer,
physically create/destroy rows, and force a full `GaussianField` + optimizer rebuild
(`_carry_adam_state`) on every event. There is no merge operator in the package; the only merge
implementation lives in the standalone prototype
`deprecated_scripts/fit_janelle_complete_refinement.py` (`merge_redundant`: mutual-nearest pair mining,
midpoint + generalized-eigen covariance envelope, count-neutral coupled teleport), which runs
*between* fits because `FitConfig` cannot express it.

Design discussion (2026-07-22, this task's originating thread) converged on:

1. **Pooled lifecycle.** Fix tensor capacity for the whole fit. "Pruning" only *parks* a row
   (making it a teleport/spawn donor with destination assigned later); actual destruction is
   deferred to a single compaction at the save boundary. Parking moves the row off-image, where
   CORE-003 tile clipping makes its render contribution and gradient **exactly zero** at zero
   tile cost — no renderer change and no epsilon-opacity hazard (a tiny-opacity row that is the
   sole owner of a pixel still fully determines that pixel's color under the normalized
   compositor, so opacity-based parking is not function-preserving; support-clipped parking is).
2. **Byte-budgeted capacity.** Capacity is derived from a target encoded-file size (e.g.
   168,000 decimal bytes — the same convention as the existing realtime-gs `.rtgsv` cap; the
   CLI flag `--target-file-kb 168` means exactly that) through the SSPL1 fixed-length grammar
   (11 B/row with opacity at default bit widths) plus a compressed alpha-map stream for masked
   inputs and a header allowance. The budget binds on the *encoded* file; the guarantee is
   `live rows <= capacity` at every instant plus an assert at encode time.
3. **Error-triage event.** One event (`triage_every`) replaces the independent timers in pooled
   mode: a shared responsibility/attribution pass, then park -> merge -> split -> spawn with
   per-op budgets. Merges and parks refill the free list; splits and spawns consume it
   (count-neutral discipline by construction). Park is gated on **max responsibility** over the
   row's support (not summed activity) so parking is function-preserving under the normalized
   renderer.

Negative results this task must not disturb: FIT-017/018 frozen screens (their score fixtures
stay as-is; the responsibility pass here is new plumbing, not a retune), BENCH-012's closed ACPD
selector, COMP-006's marginal-birth negative. No quality/rate claim is made here; this is
mechanism + infrastructure, default off (ADR-0010).

## Goal

Promote the prototype's lifecycle into the package as an opt-in pooled fitting mode:

- `pool.py`: capacity derivation from `target_file_bytes` (SSPL1 raw stream layout + alpha
  payload + header allowance), pooled field preparation (park sentinels, live mask), park/free
  bookkeeping.
- `triage.py`: attribution pass (per-row responsibility mass / error / max responsibility /
  activity + per-pixel weight-sum), park gate, envelope merge (prototype-parity math + mask
  containment batch-restore), major-axis split into free rows, site activation (masked residual
  peak NMS, hole-vs-covered opacity init, tensor-aligned anisotropy).
- `fit.py`: gated triage event with in-place row writes and per-row optimizer-state zeroing (no
  optimizer rebuild), parked-row exemption in `_MaskConstraint.apply`, terminal compaction,
  `pool`/`triage_events` reporting.
- `codec.py`: optional alpha stream (uniform extra framed stream, backward-compatible with old
  decoders), `decode_alpha`, raw stream-size helpers for the capacity math.
- CLI: `--target-file-kb`, `--pool-capacity`, `--triage-every`, per-op counts; budget mode also
  writes the SSPL1 (with alpha when masked) next to the NPZ and reports bytes vs budget.

## Acceptance criteria

1. **Exact-zero parking.** A field with parked rows renders bit-identically to the same field
   with those rows removed, and parked rows receive zero gradient and zero activity. Tested.
2. **Capacity guarantee.** With `target_file_bytes` set, `capacity` is derived so that
   `header_allowance + alpha_payload + framing + live_n * raw_row_bytes <= target_file_bytes`;
   the end-to-end test encodes the compacted field (alpha included when masked) and asserts the
   actual blob is `<= target_file_bytes`. `pool_capacity` overrides derivation.
3. **Count-neutral discipline.** `live_n <= capacity` after every event; splits/spawns consume
   only free rows; merge parks its absorbed partner; park never drops live detail count below
   `prune_keep_min`; background rows (CORE-009) are never parked/merged/split.
4. **No optimizer rebuild.** In pooled mode the optimizer object and parameter tensors are
   created once; triage events zero moment rows in place for touched rows only. Tested by
   object identity across events.
5. **Merge parity + containment.** Envelope-merge math matches the prototype (midpoint mean,
   amplitude-weighted color, union opacity, generalized-eigen envelope * inflation, eigenvalue
   floor); with `mask_contain`, pairs whose certified caps cannot preserve the envelope are
   batch-restored (both rows, partner un-parked). Tested on synthetic pairs.
6. **Mask boundary parity.** Masked pooled fits keep CORE-010/011 semantics: masked loss
   weights band pixels equally with all in-mask pixels, activation sites are restricted to the
   eroded interior, parked rows are exempt from mean projection, and containment refresh runs
   after each triage event. Out-of-mask render stays exactly zero with `support_fade`.
7. **Alpha stream round trip.** `encode(..., alpha=...)` adds one uniform framed stream; the
   pre-change decoder path ignores it (old-blob decode unchanged, new-blob field decode
   identical to alpha-free encode); `decode_alpha` returns the 8-bit-quantized map.
8. **Reproducibility.** Two pooled fits from the same config + seed on CPU produce bitwise
   identical final fields and event histories (deterministic free-list order and selections).
9. **Validation.** Pooled mode rejects incompatible knobs with clear errors: legacy
   `split_every`/`relocate_every`/`prune_every`, `adaptive_count`, `mask_boundary_add_every`,
   fit-time QAT / `lambda_rate`, `checkpoint_policy=best_psnr_final_count`, affine color basis,
   non-normalized renderers, capacity smaller than the initial field.
10. `pytest -q` passes; default behavior of every existing path is unchanged (all new knobs
    default off/None).

## Interfaces touched

`src/structsplat/pool.py` (new), `src/structsplat/triage.py` (new), `fit.py` (gated event +
apply exemption + compaction), `codec.py` (alpha stream + size helpers), `config.py` (knobs +
validation), `cli.py` (flags + budget save), `tests/test_pool_triage.py` (new),
`docs/adr/0020-*`, `docs/architecture.md`, `tasks/INDEX.md`.

## Depends on

FIT-002/004/007/017/018 (operator library + responsibility machinery), CORE-003 (clipped
support = free parking), CORE-009 (background rows), CORE-010/011 + ADR-0017/0019 (mask
containment), COMP-001/ADR-0007 (SSPL1 grammar), ADR-0010 (searchable-axis protocol),
prototype `deprecated_scripts/fit_janelle_complete_refinement.py`.

## Notes / deferred (phase 2 — new tasks, not silent additions)

- Blob-level dispatch (connected high-error components with per-blob budgets) and the
  color-fixable vs geometry error decomposition from the design thread.
- Moment-preserving split reuse (FIT-007 math) inside triage; multi-scale matched-filter site
  scoring (FIT-017 machinery) as searchable alternatives.
- Local render-delta merge acceptance gate (exact crop re-render).
- Benchmark slice: pooled triage vs the best independent-timer configuration at equal budgets
  (difficult-four proxy, then fair regime) before any default/promotion talk.
- Growth-by-doubling for the uncapped mode (capacity is currently required, preallocated once).
- Soft-alpha (boundary-band 8-bit) alpha stream refinement; entropy-aware capacity (zlib is
  bonus margin only in v1).
