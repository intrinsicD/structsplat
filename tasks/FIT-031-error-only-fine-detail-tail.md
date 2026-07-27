# FIT-031: Error-only fine-detail tail

## Status

Implemented and screened on one exposed development image. The optional tail improved its own
pre-tail protected state, but the existing default run is neither count-, rate-, nor
trajectory-matched; the recipe default remains off. This does not implement FIT-030's continuous
rate-aware allocator.

## Context

The current safe schedule stops with visible high-frequency residual on the masked Janelle
subject. FIT-025 already tested a fixed 512-row, covered-interior high-pass detail reserve and
found that ordinary activation was better at equal count. The requested follow-up is different:
measure the residual after the entire ordinary schedule, estimate its effective spatial demand,
spend half of that estimate on small error-ranked Gaussians, and optimize the resulting field to a
safe fixed point.

## Goal

Add one optional final stage to `scripts/convert.py` that increases fine-detail capacity from the
terminal residual while leaving the default conversion path unchanged.

## Acceptance criteria

- [x] `--fine-detail` is exposed only through the existing `scripts/convert.py` entrypoint and is
      disabled by default.
- [x] The stage estimates complete residual demand as the effective support of foreground
      per-pixel absolute error, `N_eff = ceil((sum e)^2 / sum(e^2))`, and requests
      `ceil(0.5 * N_eff)` rows.
- [x] Candidate rank, orientation, and footprint use only the residual map; the target supplies
      the new row's color, while the mask remains a geometric feasibility/containment constraint.
- [x] Rows are proposed in bounded dynamic batches, retain the existing full Pareto commit gate,
      and are followed by low-learning-rate fixed-topology optimization until a deterministic
      fixed point or a logged step ceiling.
- [x] Config, history, storage telemetry, result rows, and `index.html` expose the estimator,
      requested/accepted rows, convergence reason, and before/after protected metrics.
- [x] Focused tests cover the estimator, residual-only birth geometry, default-off parity, CLI
      wiring, telemetry, finiteness, and mask containment.
- [x] The same masked Janelle C0001 seed-0/max-side-1200 experiment is executed and independently
      audited against the existing default run without promoting a general/default claim.

## Interfaces touched

`src/structsplat/safe_schedule.py` · `src/structsplat/pipeline.py` ·
`src/structsplat/workflows.py` · `scripts/convert.py` (unchanged thin entrypoint) ·
`tests/test_safe_schedule.py` · `tests/test_pipeline.py` ·
`tests/test_pipeline_workflows.py` · `scripts/experiments/audit_fit031_error_tail.py` ·
README/architecture/ADR/ARA evidence records.

## Depends on

FIT-023, FIT-025, CORE-012, ADR-0022, ADR-0025, ADR-0028.

## Notes

- This is a finite post-schedule capacity experiment measured in rows. It does not maintain a
  residual EMA, price rows in bits, remove phase boundaries, or implement `D + lambda R`; those
  remain blocked under FIT-030.
- “Complete residual demand” is an estimator name, not a promise of zero reconstruction error.
  The effective-support formula is logged so the heuristic is falsifiable from the run artifact.
- FIT-025's negative equal-count result remains intact; this task tests a different estimator,
  scale, placement rule, count regime, and terminal position.

## Development result

On masked Janelle C0001, seed 0, max-side 1200, the post-polish estimator returned 14,177
effective sites and requested 7,089 rows. Nine 512-row waves committed (4,608 rows); the next wave
failed at every bisection down to the configured eight-row minimum. The first fixed-topology block
also failed safely, so convergence ended at the deterministic fixed point.

Against the same run's pre-tail state, foreground/boundary PSNR changed
`26.789684/15.003354 -> 27.311923/15.586106 dB`; CVaR99/p99 MSE fell `12.09%/14.82%`, boundary
holes fell `0.4821` percentage points, and interior/outside metrics remained exactly zero. The
existing clean default ended at 10,824 rather than 11,000 rows, so its `26.821604 dB` display PSNR
is context rather than an equal-count or causal comparator. The final 15,608-row field is not
rate-matched and took a different CUDA trajectory. C58 and
`ara/evidence/fit031-error-only-tail-janelle-2026-07-27/` bind the result and its limitations.
