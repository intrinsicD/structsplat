# Task index

Active work stays in `tasks/`; retired completed work lives in `tasks/done/`. Areas: CORE, INIT,
FIT, HIER, BENCH, ABL, FF, GEN, COMP, PORT, MERGE, DOCS. Work items are picked up via the
`task-workflow` skill.

The table below is the current outcome authority. Executed tasks whose protocol text is bound by
an artifact may intentionally retain a pre-execution `Status` section so its frozen hash remains
replay-valid; do not rewrite those task bytes merely to duplicate this table.

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
| CORE-007 | Segmentation-free responsibility boundary flux | design-only — BENCH-007 did not authorize the old structure/compression path; any reopening needs a new prior-art-controlled question | INIT-004, CORE-001, BENCH-007 |
| CORE-008 | Hybrid Gaussian + frequency-bearing primitive control | design-only — WIPES remains the direct primitive baseline and BENCH-007 supplies no promotion signal | CORE-001, INIT-001, FIT-001, BENCH-007 |
| INIT-009 | Progressive WSE survivor ordering | implemented/confirmed — 32/32 uniform Euclidean prefix wins with identical terminal sets; opt-in for compatibility | INIT-003/005/006, BENCH-002 |
| PORT-002 | GPU-native tile index + fused loss/backward | locally correctness-validated, performance unmeasured — in-extension CUB binning, staged tiled kernels, opt-in exact ellipse cull; preregistered profile, fused loss, and graphs open | PORT-001, FIT-003 |
| PORT-003 | Avoid atomics in tiled backward | locally gradient-validated, performance unmeasured — warp-reduced tiled backward atomics; preregistered timings pending | PORT-001 |
| PORT-004 | Exact-backward block reduction | implemented/screened — large N=2048 microprofile gains, but all-grid direction and independent CV guards fail; benchmark-only, not CLI/default | PORT-001/003, ADR-0011, BENCH-002 |
| GEN-003 | VSD / multi-particle distillation | todo | GEN-001 |
| ABL-005 | Fitter-knob influence pass at the fair regime | partial — CUDA-native fair shard started; color-solve and broader Kodak cells pending | ADR-0010, FIT-005/006/007, CORE-006 |
| BENCH-005 | Native external-reference pipelines and paired central metrics | partial — GI++/Image-GS/GaussianImage adapters plus bounded native GaussianImage/AIR evidence complete; full-resolution actual-RD and remaining 2026 methods open | BENCH-001/002/003, ABL-004 |
| FIT-013 | Geometry-consistent Sobel regularization | partial — quality candidate validated on COCO proxy and Kodak4; speed blocks promotion | FIT-005/006/007, ABL-004 |
| FIT-014 | Generation-density covariance filtering | implemented/screened — exact and weaker cohort filters lose the COCO4 proxy; default off | FIT-004, CORE-002, COMP-002, BENCH-002 |
| FIT-015 | Same-final-count best-PSNR checkpoint selection | implemented/confirmed — +0.4884 dB pooled across 72 Kodak trajectories; opt-in for sparse/moderate-density long fits | FIT-001/002, ABL-004, BENCH-005 |
| FIT-016 | Coarse-to-full loss-target curriculum | implemented/screened — rejected by 500-step guard (−0.1645 dB selected PSNR); default off | FIT-015, HIER-003/004, ABL-004, BENCH-002 |
| FIT-017 | Kernel-matched signed-residual densification | implemented/screened — wider scores improve immediate PSNR but lose post-20/post-100; default legacy score retained | FIT-004, FIT-009, BENCH-002, ABL-004 |
| FIT-018 | Responsibility-normalized error-density densification | implemented/screened — SAD alpha-0.7 transfer rejected (−0.0198 dB post-20, 4/8 wins); opt-in control only | FIT-004, FIT-009, FIT-017, BENCH-002, ADR-0010 |
| FIT-019 | Opacity-split gauge-equivalence audit | implemented/screened — exact commutation confirmed, recovery utility rejected; no production quotient/lineage state | FIT-007, FIT-009, FIT-018, BENCH-002 |
| FIT-020 | Ranked deduplication response spectroscopy | implemented/screened — signal strong, but bend prediction and screening rejected; close lineage, no production selector | FIT-007, FIT-009, FIT-018/019, BENCH-002 |
| BENCH-006 | Fixed-storage all-method convergence lane | completed as a 320-cell high-rate local-policy diagnostic; superseded for compression decisions by BENCH-007 | BENCH-001/002/003/004, COMP-002, ABL-004 |
| BENCH-007 | Actual-rate structure phase diagram | completed negative — Stage-1 gate failed; Stage 2 prohibited and not run | BENCH-001/002/003/004/006, COMP-001/002/004, INIT-003/009 |
| BENCH-008 | Common/native causal bridge | not authorized — BENCH-007 found no promotable renderer/objective interaction | BENCH-005/007, CORE-001/003/005, COMP-002 |
| BENCH-009 | Rate/DOF-priced residual tangent-space auction | completed negative/unavailable — exact Stage-1/recovery ledgers complete; causal calibration and projector validity failed; no method/expressiveness claim | CORE-001/006, FIT-005/017/020, BENCH-002, COMP-006 |
| BENCH-011 | Nested residual-extension spent-data diagnostic | completed negative — corrected v2 exactly reproduces BENCH-009 bases and all 96 rows; all four calibration strata fail; close without retuning | BENCH-009 |
| BENCH-012 | Spatial-connectivity policy value for finite reallocation | completed unavailable — topology core passed, but first preflight cell had only 2/4 required untruncated equal-work actions; no selector or recovery outcome scored | BENCH-009/011, FIT-004/017/020, COMP-006 |
| COMP-007 | Gauge-free log-Euclidean covariance codec | completed negative — v4 audit replayed all 12,096 streams; log-SPD failed 7/8 frozen gates and confirmation stayed sealed | COMP-006, BENCH-012 |
| COMP-008 | Mean-conditioned entropy-oracle killing test | completed inconclusive — both fixed tuples survived the necessary-condition lower-bound screen; this authorized COMP-009 only, not an actual compression claim | BENCH-016, COMP-006/007 |
| COMP-009 | Exact SSP2E actual-coder development test | completed negative — both tuples failed actual-rate and spatial-attribution gates; frozen decision `ABANDON_FIXED_SSP2E_V1`, confirmation sealed | COMP-008 |
| COMP-010 | SSP2E captured-replay relocation repair | completed provenance GO — captured-source replay confirms COMP-009's frozen negative decision; no new rate, quality, convergence, or performance evidence | COMP-009 |
| COMP-011 | Complete-stream RGB VQ/RVQ | terminal invalid/no-decision — candidate-byte replay passed, but one original decision-relevant LPIPS constituent is inside the frozen replay exclusion band; no SSP2V/SSP2L confirmation | COMP-008/009/010 |
| COMP-012 | Exact-byte RGB coordinate RDO with unchanged SSP2F | repaired draft/pre-data — objective/search/oracle/operator/paired-adapter cores implemented and hostile-tested; lifecycle orchestration, runtime/transcript caps, frozen gates, and preflight remain before any target access | COMP-011 provenance audit, COMP-009 SSP2F |
| BENCH-013 | Local-linear reproducing compositor | completed negative — affine reproduction passed, but signed local leverage/ringing killed 82/108 forward cells | COMP-007, CORE-006 |
| BENCH-014 | Explicit affine carrier | completed negative — strong synthetic quality/rank/cost, but complete-byte and terminal-convergence gates failed | BENCH-013, CORE-006 |
| BENCH-015 | Decoder-synchronized affine lift | completed negative — equal-byte smooth quality passed; no-harm, convergence, and cold-decode gates failed; Stage 1/local successor prohibited | BENCH-014, COMP-007 |
| BENCH-016 | Pinned native SAD frontier screen | completed negative development screen — valid 144-row v6 matrix; 0.5-bpp gates passed, 2.0-bpp quality/LPIPS guards failed; decision `abandon SAD reuse` | BENCH-005/007/015, FIT-018 |
| COMP-005 | Decoder-synchronized structural geometry | not authorized — tensor structure failed the actual-rate gate and layout bytes were not established as the binding loss | BENCH-007, COMP-001/002/003/004, INIT-001/003/009 |
| COMP-006 | Marginal cold-stream rate--distortion attribution | completed negative — exact replay matched; birth lost `-1.0714 dB` to the strongest actual-byte control and confirmation remains sealed | COMP-001/002/003/004, BENCH-002/007, FIT-004/017 |
| CORE-010 | Mask-contained fitting for alpha-masked inputs | implemented (opt-in, default off) — SDF hard containment (mean projection + dynamic caps, ADR-0017) + out-of-mask coverage penalty + masked loss; exact zero outside with support_fade; directional-cap follow-up landed as CORE-011; five-arm cost benchmark + pyramid support deferred | CORE-003/005, INIT-002/003, FIT-012, ADR-0012/0017 |
| CORE-011 | Boundary coverage for mask-contained fitting | implemented (opt-in, default off) — certified anisotropic tangent caps (station-ball SDF certificate, ADR-0019) + boundary-band under-coverage hinge + boundary tangent densification + CLI band PSNR; exact-zero-outside guarantee preserved; committed multi-arm benchmark deferred (shared with CORE-010) | CORE-010, ADR-0017/0019, CORE-003/005, FIT-012 |
| PORT-005 | Batch encode throughput (multi-process across images) | implemented (CPU-validated) — `batch-fit` CLI with worker pool, device round-robin, resumable metrics.jsonl, failure isolation; GPU scaling row pending | FIT-001, CORE-001, ADR-0011, PORT-001 |
| FIT-021 | Pooled row lifecycle, byte-budgeted capacity, error-triage events | implemented (opt-in, default off, ADR-0020) — fixed-capacity pool with off-image parking + free list, capacity from `target_file_bytes` via SSPL1 raw layout + in-container alpha stream, one triage event (responsibility-gated park → envelope merge → split → spawn) with in-place optimizer rows; CPU end-to-end + budget/determinism tests; benchmark slice and blob-dispatch/color-fixable phase 2 open | FIT-002/004/007/017/018, CORE-003/009/010/011, COMP-001, ADR-0007/0010/0017/0019/0020 |
| FIT-022 | Coverage-matching regularizer (feature-targeted fit-time Gaussian blue noise) | implemented (opt-in, default off) — mass-neutral `mean_inside (S−c)²` on the compositor's raw weight sum with detached opacities, tensor/tensor_boundary/error_blend targets, cosine decay; gradient/transport/compose tests green; the audit's preregistered fixed-N killing experiment (incl. additive-differential arm) remains the required screen before any promotion | 2026-07-23 ideation audit, CORE-010/011, FIT-012/021, ADR-0003/0010/0020 |

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
| DOCS-002 | Publication visual diagnostics | `done/DOCS-002-publication-visual-diagnostics.md` |

Retired tasks remain valid dependency IDs. They describe completed reference/correctness work; the
performance and scale follow-ups stay active under PORT/FIT/INIT/BENCH/ABL tasks.

## Current priority after the 2026-07-15 recovery and marginal-RD assays

FIT-018--020 completed the recovery branch requested after BENCH-007, and COMP-006 completed its
next exact-rate screen. Responsibility-density transfer, opacity-gauge quotient allocation, the
frozen response bend, and one-extra-standard-row at a complete-stream byte cap all failed their
promotion gates. Their exact grouping, response logging, and cold-stream selection machinery
remain benchmark oracles, not production features. The queue is therefore:

1. Preserve the FIT-018--020 and COMP-006 negative results; do not retune their fixtures, splits,
   alphas, bends, horizons, caps, actions, bit box, or exposed held-out targets.
2. Preserve BENCH-009's completed v3 ledgers and failed causal decision. Do not reinterpret its
   negative incremental projector rows, late carrier recovery, objective mismatch, or provisional
   packet bytes as expressiveness, convergence, performance, or compression evidence.
3. Preserve BENCH-011 v1 as an audited invalid run and corrected v2 as the valid negative. V2
   reproduces BENCH-009's exact base seeds/ranks and all `96` rows, but all four calibration strata
   fail. Close the local-linear extension formulation without retuning or disjoint-data spending.
4. Preserve BENCH-012 as an unavailable preflight, not a topology result. Do not relax its
   equal-work/untruncated candidate filter, move the exposed targets, shrink its inherited action,
   or reinterpret the 2/4 feasibility failure as quality, convergence, or allocation evidence.
5. Preserve PORT-004's opt-in exact-backward block-reduction prototype as benchmark-only. It
   repeats large `N=2048` representative gains, but fails the governing all-grid direction rule
   and an independent run's `5%` stability guard; do not open end-to-end or default promotion.
6. Preserve COMP-007 v2/v3 as pre-scoring unavailable artifacts and v4 as the valid negative. Do
   not retune the log-SPD chart or expose the odd Kodak IDs; its `0.3426%` zstd median movement is
   below the frozen effect and seven of eight gates fail.
7. Preserve BENCH-013--015 as the closed first-order-reproduction lineage. The local-linear
   compositor, transmitted affine carrier, and decoder-synchronized lift each failed their frozen
   gate; do not run Stage 1 or retune a local successor on these artifacts.
8. Explore existing-grammar attribute/rate co-design—per-group QAT, real context/range coding, and
   `R + lambda D`—as a separate task that retains COMP-006's exact-byte gate.
9. Continue BENCH-005 native-authentic coverage only as external-validity infrastructure; it cannot
   retroactively promote a failed local mechanism.

## Dated priority after the 2026-07-14 actual-rate decision

BENCH-007 is complete. Tensor-WSE showed a bounded 0.5-bpp gain over the strongest local gradient
control, but the effect vanished at 1.0 bpp, reached only `-4.5417%` BD-rate versus the required
`-10%`, cost `1.4752x`, and exceeded the texture guard. The preregistered gate failed, so Stage 2,
BENCH-008, and COMP-005 are not authorized. The queue is therefore:

1. Preserve the BENCH-007 negative result and do not tune the eight Stage-1 images or consume
   untouched DIV2K validation as a rescue set.
2. Close the current tensor-WSE compression-paper claim. F5--F9 and the actual-rate harness are
   reusable infrastructure, not held-out method evidence.
3. If method research continues, begin a fresh `structsplat-research-ideation` pass around a
   materially different question, null, hard compute/texture guards, and a disjoint development
   screen. The low-rate edge/bleed signal may motivate that search but cannot preselect a rescue
   formulation on the failed pilot.
4. Continue BENCH-005's native-authentic coverage only as an independent external-validity or
   benchmark lane; it cannot retroactively promote tensor-WSE.
5. Keep CORE-007/008 design-only until a new question and their direct Contour-Aware 2DGS/WIPES
   controls justify implementation.

The dated priority sections below are retained as execution history, not as the current queue.

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
