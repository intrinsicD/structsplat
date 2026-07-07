# Task index

Active work stays in `tasks/`; retired completed work lives in `tasks/done/`. Areas: CORE, INIT,
FIT, HIER, BENCH, ABL, FF, GEN, COMP, PORT, MERGE, DOCS. Work items are picked up via the
`task-workflow` skill.

## Active Tasks

| ID | Title | Status | Depends on |
|----|-------|--------|-----------|
| INIT-004 | Flanking vs on-edge placement + threshold study | answered (negative) — retire via INIT-007 | INIT-003 |
| HIER-001 | Progressive pyramid (residual-driven densification) | partial | INIT-002, FIT-001 |
| ABL-001 | Init-strategy x budget sweep (the core experiment + fitness) | partial | INIT-003/004, BENCH-001 |
| ABL-002 | Full stage-combination search | partial | CORE, INIT, FIT, HIER, BENCH |
| FF-001 | Feed-forward init predictor (warm-start) | partial | INIT-003, FIT-001 |
| GEN-001 | Generative 2D Gaussians via SDS distillation (no dataset) | todo | CORE-001, ADR-0006 |
| COMP-001 | Quantization + entropy/VQ codec (rate-distortion) | partial | FIT-001 |
| PORT-001 | CUDA tile rasterizer → IntrinsicEngine RHI pass | partial | CORE-001 |
| CORE-005 | Reference renderer memory bound + C0-continuous support cutoff | partial | CORE-003, CORE-004 |
| ABL-004 | Killer controls + full ABL-001 run + committed evidence | partial | BENCH-002, ABL-003, FIT-004 |
| COMP-003 | Compression-ratio ladder (scale ranges → planes → LSQ → VQ → entropy) | partial | COMP-002, BENCH-002 |
| CORE-007 | Boundary-gated Gaussians | todo | INIT-004, CORE-001 |
| CORE-008 | Hybrid Gaussian + edge primitives | todo | CORE-001, INIT-001, FIT-001 |
| COMP-004 | QAT + entropy-aware fitting | partial | COMP-001, COMP-003, FIT-001 |
| PORT-002 | GPU-native tile index + fused loss/backward | todo | PORT-001, FIT-003 |
| PORT-003 | Avoid atomics in tiled backward | todo | PORT-001 |
| GEN-003 | VSD / multi-particle distillation | todo | GEN-001 |
| INIT-007 | Retire the flanking default (measured answer + ADR) | todo | INIT-004, ABL-006 |
| INIT-008 | Feature-relative scale caps (fix the cap-scaling failure) | todo | ADR-0012, INIT-003 |
| ABL-005 | Fitter-knob influence pass at the fair regime | todo | ADR-0010, FIT-005/006/007, CORE-006 |
| ABL-006 | Successive-halving execution of the remaining confirmation | todo | ABL-004, BENCH-002 |
| FIT-009 | Factor the refine axis into orthogonal sub-axes | todo | FIT-004, FIT-006, FIT-007 |
| FIT-010 | Cheap color-solve schedules (init / final / on-split) | todo | FIT-005 |
| FIT-011 | Split-recovery micro-levers (moment seeding, warmup, scheduled fade) | todo | FIT-004, FIT-007, CORE-005 |
| FIT-012 | Edge-weighted pixel loss (structure-tensor loss weighting) | todo | FIT-001, INIT-001 |
| HIER-003 | Pyramid equal-iteration diagnosis (fix or retire HIER-001) | todo | HIER-001, HIER-002 |
| CORE-009 | DC / background layer under the detail Gaussians | todo | CORE-001, ADR-0003/0006 |

## Retired Done Tasks

| ID | Title | Path |
|----|-------|------|
| CORE-001 | Differentiable reference rasterizer (normalized weighted sum) | `done/CORE-001-reference-rasterizer.md` |
| CORE-002 | RS Gaussian parameterization + conics | `done/CORE-002-rs-gaussian-params.md` |
| CORE-003 | Edge-aware render support window (off-image support + tile waste) | `done/CORE-003-render-support-clamp.md` |
| CORE-004 | Renderer + GaussianField correctness fixes (CUDA N=0, int-cast UB, aliasing, dilation) | `done/CORE-004-renderer-field-correctness.md` |
| CORE-006 | Linear color basis per Gaussian | `done/CORE-006-affine-color-basis.md` |
| INIT-001 | Structure tensor: energy, orientation, flat/edge/corner | `done/INIT-001-structure-tensor.md` |
| INIT-002 | Density field (image + residual) | `done/INIT-002-density-field.md` |
| INIT-003 | Anisotropic blue-noise sampling (WSE + metric) | `done/INIT-003-anisotropic-pds.md` |
| INIT-005 | Init-math robustness, flanking unification, WSE test coverage | `done/INIT-005-init-robustness.md` |
| INIT-006 | Init-time performance (quadtree, spacing, run-lengths, pair discovery) | `done/INIT-006-init-performance.md` |
| FIT-001 | Adam fitter (L1+SSIM), PSNR history, iters-to-target | `done/FIT-001-optimizer.md` |
| FIT-002 | Fitter correctness (split colors, opacity pruning, history pairing) | `done/FIT-002-fitter-correctness.md` |
| FIT-003 | Fit-loop speed (device-side targets, SSIM hygiene, fused SSIM) | `done/FIT-003-fit-loop-speed.md` |
| FIT-004 | Densification & convergence upgrades (fp-growth, relocation, NMS) | `done/FIT-004-densification-upgrades.md` |
| FIT-005 | Exact / alternating color solve | `done/FIT-005-exact-color-solve.md` |
| FIT-006 | Frequency-violation densification | `done/FIT-006-frequency-violation-densification.md` |
| FIT-007 | Moment-preserving split / clone | `done/FIT-007-moment-preserving-split.md` |
| FIT-008 | Self-adaptive Gaussian count | `done/FIT-008-self-adaptive-gaussian-count.md` |
| HIER-002 | Pyramid bookkeeping (iteration accounting, budgets, schedules) | `done/HIER-002-pyramid-bookkeeping.md` |
| BENCH-001 | Metric protocol (PSNR/MS-SSIM/LPIPS + iters-to-target) | `done/BENCH-001-metrics.md` |
| BENCH-002 | Benchmark harness experimental-validity fixes (equal budgets, resumable sweeps, seed-aware comparisons) | `done/BENCH-002-harness-validity.md` |
| BENCH-003 | Benchmark script consolidation + documentation | `done/BENCH-003-benchmark-consolidation.md` |
| BENCH-004 | Sweep-cost controls (plateau exit, multi-target tables, proxy regime) | `done/BENCH-004-sweep-cost-controls.md` |
| ABL-003 | Bisect the undiagnosed −0.794 dB flagship regression | `done/ABL-003-regression-bisect.md` |
| MERGE-001 | Integrate Claude core optimizations and Codex stage search into main | `done/MERGE-001-claude-codex-main.md` |
| COMP-002 | Codec / metrics / CLI correctness and protocol fixes | `done/COMP-002-codec-correctness.md` |
| DOCS-001 | Docs-sync backfill (stale status, missing ADRs, ara scaffold) | `done/DOCS-001-docs-sync-backfill.md` |

Retired tasks remain valid dependency IDs. They describe completed reference/correctness work; the
performance and scale follow-ups stay active under PORT/FIT/INIT/BENCH/ABL tasks.

## Suggested order (from the 2026-07-03 repo review)

CORE-004/FIT-002/HIER-002/COMP-002/INIT-005/BENCH-002/ABL-003 fix confirmed bugs and
science-gating ambiguities. FIT-003 removed fit-loop metric overhead and added the optional fused
SSIM backend, and FIT-004 added the densification/relocation controls needed for the experiment.
Next run ABL-004 (the actual experiment, with evidence committed). After that, the 2026-07 SOTA
review suggests the most pragmatic improvement order:

1. FIT-005 exact/alternating color solve — completed 2026-07-06; keep default off, stage-search
   axis available as `color_solve=every10`.
2. FIT-006 frequency-violation densification — completed 2026-07-06; keep default off,
   stage-search refine axis available as `freq_violation`.
3. FIT-007 moment-preserving split — completed 2026-07-06; keep default off,
   stage-search refine axis available as `moment_preserving`.
4. CORE-006 affine color basis — completed 2026-07-06; keep default `constant`, stage-search
   axis available as `color_basis=affine`.
5. FIT-008 self-adaptive Gaussian count — completed 2026-07-06; keep default fixed-N, but
   `--adaptive-count` now exposes target/max-N/stall-controlled growth and selected-N metadata.
6. FF-001 feed-forward teacher-student warm start, now able to consume FIT-008 selected-N metadata.
7. COMP-004 for compression-aware fitting once RD baselines are stable.
8. PORT-002/PORT-003 if tiled CUDA remains strategically important after quality work.
9. GEN-003 after GEN-001 has a debuggable SDS baseline.

## Suggested order (from the 2026-07-07 benchmark review)

The 2026-07-04..07 evidence answered the flanking question negatively (INIT-004; retirement is
INIT-007) and shifted the frontier from init strategies to fitter knobs and sweep economics:

1. BENCH-004 sweep-cost controls — completed 2026-07-07. Use the 512/750 proxy screen for cheap
   screening, keep early exit opt-in, and promote only from full fair-regime evidence.
2. ABL-005 fitter-knob influence pass — the +0.26 dB `charbonnier`/`variance`/`opacity` bundle
   and the FIT-005/006/007/CORE-006 candidates, isolated at the fair regime. Highest expected
   dB-per-GPU-hour in the queue.
3. ABL-006 successive-halving confirmation — finishes ABL-004's remaining cells at half to
   two-thirds cost; feeds INIT-007's default flip (ADR-0013).
4. FIT-009 refine-axis factoring, then FIT-010/FIT-011 — convergence-rate work targeting the
   measured split dip; `residual_tensor x moment_preserving` is the first inexpressible
   combination to test.
5. INIT-008 feature-relative caps and FIT-012 edge-weighted loss — quality levers with clear
   accept/park criteria.
6. HIER-003 pyramid diagnosis and CORE-009 background layer — the two low-frequency-coverage
   investigations; either may produce a default or an honest retirement.
7. FF-001 (multi-image teacher training; the 2026-07-07 equal-N smoke is a measured negative for
   the tiny checkpoint) and COMP-004 (lambda sweep) continue in their existing task files.
