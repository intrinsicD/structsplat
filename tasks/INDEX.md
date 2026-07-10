# Task index

Active work stays in `tasks/`; retired completed work lives in `tasks/done/`. Areas: CORE, INIT,
FIT, HIER, BENCH, ABL, FF, GEN, COMP, PORT, MERGE, DOCS. Work items are picked up via the
`task-workflow` skill.

## Active Tasks

| ID | Title | Status | Depends on |
|----|-------|--------|-----------|
| ABL-001 | Init-strategy x budget sweep (the core experiment + fitness) | partial | INIT-003/004, BENCH-001 |
| ABL-002 | Full stage-combination search | partial | CORE, INIT, FIT, HIER, BENCH |
| GEN-001 | Generative 2D Gaussians via SDS distillation (no dataset) | todo | CORE-001, ADR-0006 |
| COMP-001 | Quantization + entropy/VQ codec (rate-distortion) | partial | FIT-001 |
| PORT-001 | CUDA tile rasterizer → IntrinsicEngine RHI pass | partial | CORE-001 |
| ABL-004 | Killer controls + full ABL-001 run + committed evidence | partial | BENCH-002, ABL-003, FIT-004 |
| COMP-003 | Compression-ratio ladder (scale ranges → planes → LSQ → VQ → entropy) | partial | COMP-002, BENCH-002 |
| CORE-007 | Boundary-gated Gaussians | todo | INIT-004, CORE-001 |
| CORE-008 | Hybrid Gaussian + edge primitives | todo | CORE-001, INIT-001, FIT-001 |
| PORT-002 | GPU-native tile index + fused loss/backward | todo | PORT-001, FIT-003 |
| PORT-003 | Avoid atomics in tiled backward | todo | PORT-001 |
| GEN-003 | VSD / multi-particle distillation | todo | GEN-001 |
| ABL-005 | Fitter-knob influence pass at the fair regime | partial — CUDA-native fair shard started; color-solve and broader Kodak cells pending | ADR-0010, FIT-005/006/007, CORE-006 |
| BENCH-005 | Native external-reference pipelines and paired central metrics | partial — native GI++/Image-GS/GaussianImage adapters, official proxy lanes, and terminal/checkpoint pairing complete | BENCH-001/002/003, ABL-004 |
| FIT-013 | Geometry-consistent Sobel regularization | partial — quality candidate validated on COCO proxy and Kodak4; speed blocks promotion | FIT-005/006/007, ABL-004 |
| FIT-014 | Generation-density covariance filtering | implemented/screened — exact and weaker cohort filters lose the COCO4 proxy; default off | FIT-004, CORE-002, COMP-002, BENCH-002 |
| FIT-015 | Same-final-count best-PSNR checkpoint selection | implemented/confirmed — +0.4884 dB pooled across 72 Kodak trajectories; opt-in for sparse/moderate-density long fits | FIT-001/002, ABL-004, BENCH-005 |
| FIT-016 | Coarse-to-full loss-target curriculum | implemented/screened — rejected by 500-step guard (−0.1645 dB selected PSNR); default off | FIT-015, HIER-003/004, ABL-004, BENCH-002 |

## Retired Done Tasks

| ID | Title | Path |
|----|-------|------|
| CORE-001 | Differentiable reference rasterizer (normalized weighted sum) | `done/CORE-001-reference-rasterizer.md` |
| CORE-002 | RS Gaussian parameterization + conics | `done/CORE-002-rs-gaussian-params.md` |
| CORE-003 | Edge-aware render support window (off-image support + tile waste) | `done/CORE-003-render-support-clamp.md` |
| CORE-004 | Renderer + GaussianField correctness fixes (CUDA N=0, int-cast UB, aliasing, dilation) | `done/CORE-004-renderer-field-correctness.md` |
| CORE-005 | Reference renderer memory bound + C0-continuous support cutoff | `done/CORE-005-renderer-memory-continuity.md` |
| CORE-006 | Linear color basis per Gaussian | `done/CORE-006-affine-color-basis.md` |
| CORE-009 | DC / background layer under the detail Gaussians | `done/CORE-009-dc-background-layer.md` |
| INIT-001 | Structure tensor: energy, orientation, flat/edge/corner | `done/INIT-001-structure-tensor.md` |
| INIT-002 | Density field (image + residual) | `done/INIT-002-density-field.md` |
| INIT-003 | Anisotropic blue-noise sampling (WSE + metric) | `done/INIT-003-anisotropic-pds.md` |
| INIT-004 | Flanking vs on-edge placement + threshold study | `done/INIT-004-flanking-vs-onedge.md` |
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
| FIT-009 | Factor the refine axis into orthogonal sub-axes | `done/FIT-009-factor-refine-axis.md` |
| FIT-010 | Cheap color-solve schedules | `done/FIT-010-color-solve-schedules.md` |
| FIT-011 | Split-recovery micro-levers (moment seeding, warmup, scheduled fade) | `done/FIT-011-split-recovery-microlevers.md` |
| FIT-012 | Edge-weighted pixel loss (structure-tensor loss weighting) | `done/FIT-012-edge-weighted-loss.md` |
| FF-001 | Feed-forward init predictor (warm-start) | `done/FF-001-feedforward-predictor.md` |
| HIER-001 | Progressive pyramid (residual-driven densification) | `done/HIER-001-progressive-pyramid.md` |
| HIER-002 | Pyramid bookkeeping (iteration accounting, budgets, schedules) | `done/HIER-002-pyramid-bookkeeping.md` |
| HIER-003 | Pyramid equal-iteration diagnosis (fix or retire HIER-001) | `done/HIER-003-pyramid-iteration-accounting.md` |
| HIER-004 | Pyramid convergence repair and promotion decision | `done/HIER-004-pyramid-convergence-repair.md` |
| BENCH-001 | Metric protocol (PSNR/MS-SSIM/LPIPS + iters-to-target) | `done/BENCH-001-metrics.md` |
| BENCH-002 | Benchmark harness experimental-validity fixes (equal budgets, resumable sweeps, seed-aware comparisons) | `done/BENCH-002-harness-validity.md` |
| BENCH-003 | Benchmark script consolidation + documentation | `done/BENCH-003-benchmark-consolidation.md` |
| BENCH-004 | Sweep-cost controls (plateau exit, multi-target tables, proxy regime) | `done/BENCH-004-sweep-cost-controls.md` |
| ABL-003 | Bisect the undiagnosed −0.794 dB flagship regression | `done/ABL-003-regression-bisect.md` |
| ABL-006 | Successive-halving execution of the remaining confirmation | `done/ABL-006-successive-halving-confirmation.md` |
| INIT-007 | Retire the flanking default (measured answer + ADR) | `done/INIT-007-retire-flanking-default.md` |
| INIT-008 | Feature-relative scale caps (fix the cap-scaling failure) | `done/INIT-008-feature-relative-scale-caps.md` |
| MERGE-001 | Integrate Claude core optimizations and Codex stage search into main | `done/MERGE-001-claude-codex-main.md` |
| COMP-002 | Codec / metrics / CLI correctness and protocol fixes | `done/COMP-002-codec-correctness.md` |
| COMP-004 | QAT + entropy-aware fitting | `done/COMP-004-entropy-aware-fitting.md` |
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
6. FF-001 feed-forward teacher-student warm start — completed 2026-07-07. Tensor-prior inputs make
   the tiny predictor useful versus random scratch on a held-out slice, but it still loses to the
   hand `quadtree_wse` prior; keep `strategy=feedforward` experimental.
7. COMP-004 compression-aware fitting — completed 2026-07-07. Fit-time STE QAT and lambda sweeps
   are available, but post-fit QAT remains the stronger default on the local RD slice.
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
3. ABL-006 successive-halving confirmation — completed 2026-07-07; fed INIT-007's default flip
   (ADR-0013), also completed 2026-07-07.
4. FIT-009 refine-axis factoring — completed 2026-07-07. The new
   `residual_tensor x moment_preserving` combination is now expressible but did not win the
   difficult-four fair slice; keep it as searchable, not default.
5. FIT-010 color-solve schedules — completed 2026-07-07. `on_split` helped split recovery but
   failed the final-PSNR promotion rule; keep `every<N>` as the quality arm and `on_split`
   searchable.
6. FIT-011 split-recovery micro-levers — completed 2026-07-07. State seeding and row tempering did
   not improve split recovery, and scheduled fade missed the AUC promotion rule; keep all three
   searchable, default off.
7. INIT-008 feature-relative caps — completed 2026-07-07. `feature_rel` repairs most of the old
   resolution-scaled cap failure but fails the no-loss promotion rule (48 paired cells, mean
   dPSNR -0.3733 vs uncapped); keep it searchable and default off.
8. FIT-012 edge-weighted loss — completed 2026-07-07. `loss_weight=tensor` is PSNR-neutral in the
   difficult-four aggregate (+0.0061 dB over 16 pairs) and helps `aniso_onedge`, but it hurts
   `quadtree_wse` and loses AUC on average; keep it searchable and default off.
9. HIER-003 pyramid diagnosis — completed 2026-07-07. The current two-level pyramid is a
   final-quality positive (+1.0000 dB mean PSNR over the difficult-four 2k/5k slice), but it loses
   AUC in every pair (-1.3540 mean), so HIER-004 owns convergence repair.
10. HIER-004 pyramid convergence repair — completed 2026-07-07. Explicit per-level schedules are
   available; 150/1350 repairs the AUC loss while preserving final PSNR on the difficult-four
   slice (+0.0601 dB vs the 750/750 pyramid control and +0.0011 AUC vs single). Keep `single` as
   shipped default pending larger confirmation; use 150/1350 as the pyramid quality candidate.
11. CORE-009 background layer — completed 2026-07-07. A counted frozen-geometry background
    Gaussian layer is a strong low-budget quality candidate (`frac0.05_grid8`: +1.0152 dB mean
    PSNR over 24 pairs), but it loses AUC at 5000 rows; keep it searchable and default off.
12. FF-001 — completed 2026-07-07 with multi-image teacher training and tensor-prior input
   ablation.
13. COMP-004 — completed 2026-07-07. Fit-time STE QAT helps low/mid-bit direct-encode quality,
   but the existing post-fit QAT control is still marginally better overall; keep the new knobs
   searchable, not default.
