# FIT-045 — Residual-budgeted densification allocation

## Context

Residual densification currently selects birth sites from a global per-pixel score
(`refine_site` residual/residual_tensor with `sampled_add_score` variants, or
support/responsibility sites) at event triggers. A global ranking lets one high-error region
absorb an entire event's births while smaller but persistent residual regions receive nothing,
and threshold-triggered spawning confounds method comparisons when arms create different row
counts — FIT-017 and FIT-018 both had to enforce matched budgets to stay interpretable. FIT-030
(design-only) diagnosed the structural version of this — "error is used as a verdict, not a
map" — but is blocked on FIT-028/BENCH-018 and owns the *rate* question. A cheaper mechanism
question is answerable now, inside the current event schedule, with total rate fixed: given a
fixed per-event spawn budget, how should it be *distributed* across residual regions?

## Goal

An opt-in per-event budget allocator for the residual-densification path: partition the frame
into residual regions (fixed tiles or connected residual components), allocate the event budget
`B_r ∝ (E_r + eps)^tau` from per-region residual energy, then run the existing site scoring and
birth machinery within each region's allocation. Screen, at identical total spawned rows per
event across all arms:

- `global` — the shipped global ranking at the same total per-event budget (status-quo control).
- `uniform` — equal candidate count per region (regionalization-only control).
- `residual_proportional` — `tau` in {0.5, 1.0, 2.0}.
- `residual_x_structure` — residual energy weighted by structure-tensor confidence.
- `residual_x_expected_gain` — residual energy weighted by a cheap expected-loss-decrease proxy
  (kernel-matched signed residual from the FIT-017 score axis).

## Non-goals

- Choosing the total spawn rate, capacity, or stopping point (FIT-030's scope; this task feeds
  its allocation component only).
- New birth-site scores (FIT-017's axis) or responsibility normalization (FIT-018's axis);
  within-region selection reuses the existing shipped machinery unchanged.
- Auction/ownership redesign (BENCH-009) and tail policies (FIT-025/031/040).
- Any default change; the allocator ships default-off.

## Acceptance criteria

- [ ] Opt-in allocator in the residual-densification path with region definition (tile size or
      connectivity) and `tau` in `FitConfig`; default off preserves current behavior on a seeded
      regression test.
- [ ] Integer allocation rule is deterministic and documented (largest-remainder or equivalent);
      unit tests cover proportionality at `tau=1`, zero-residual regions, and regions whose
      allocation exceeds their candidate count (surplus redistributes, never silently drops).
- [ ] Total spawned rows per event are asserted identical across arms in the screen harness.
- [ ] Screen on the maintained benchmark images at fixed budgets and seeds: final PSNR/MS-SSIM,
      iterations-to-target, post-event per-region residual dispersion (does allocation flatten
      the error map?), event count, and allocator wall-clock overhead, from logged config + seed.
- [ ] Outcome (positive or negative) recorded as an ARA observation or claim row, and this task's
      Index status updated in the same commit.
- [ ] `./scripts/verify.sh` passes.

## Interfaces touched

`src/structsplat/fit.py` (residual site-selection path around `refine_site`/`sampled_add`),
`src/structsplat/metrics.py` (per-region residual energy only if not already exposed),
`benchmarks/ablation.py` or a driver under `scripts/experiments/`, `tests/`, `tasks/INDEX.md`,
`ara/`.

## Depends on

FIT-017, FIT-018, FIT-030, BENCH-002

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff` and `### Review` blocks using the schema in `tasks/README.md`.
Before a formal result-bearing run, append the prospective `### Protocol review` block from that
document and bind the exact frozen protocol digest.

## Notes

Fairness is the design constraint: with total rows fixed per event, any measured difference is
attributable to *where* capacity went, not *how much* was spent — the confound FIT-017/018 had
to control by hand becomes structural. If no allocation arm beats `global` and `uniform` on the
frozen screen, record the negative result and leave the global ranking as-is; that outcome would
also lower the priority of FIT-030's allocation component relative to its rate component.
