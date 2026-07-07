# Claims

## C01: StructSplat is primarily a controllable stage-search harness

StructSplat's current strength is not a final codec claim; it is that initialization, renderer,
fitting, densification, pyramid, and codec choices are exposed as testable stages with JSON/CSV
evidence. See `benchmarks/stage_search.py`, ADR-0010, and ABL-002.

## C02: Feature-aware scale caps are searchable controls, not a current default candidate

Feature-adaptive caps reduced scale outliers and improved the small COCO screens, but the
fair-density difficult-four protocol rejected both the old resolution-scaled cap and the INIT-008
feature-relative repair as defaults. `feature_rel` improved strongly over the old absolute cap, but
still averaged -0.3733 dB PSNR versus matching uncapped rows over 48 paired cells and lost badly at
budget 2000. Keep scale caps as stage-search controls; do not present them as a default candidate
without a new task and stronger evidence. See ADR-0012 and
`ara/evidence/init008-feature-relative-scale-caps-2026-07-07/`.

## C03: Exact CUDA is a semantic accelerator, while gsplat is a comparator

`renderer=cuda` and `renderer=cuda_additive` implement StructSplat's own equations and can be used
for speed without changing the method. `renderer=gsplat` remains useful for comparison, but its
alpha/sum semantics are not equivalent to the normalized reference. See ADR-0011 and trace node N15.

## C04: Current cross-repo evidence is metric-split

The cross-repo matrices are matched executable policy analogues under StructSplat's harness, not
native external repository benchmarks. The 2026-07-03 COCO matrix showed the best-searched
StructSplat policy row leading PSNR while GaussianImage/Instant-GI-style analogue rows remained
competitive or better on some MS-SSIM/LPIPS slices. A 2026-07-04 held-out Kodak4 matrix added a
`structsplat_shipped_defaults` row; current and shipped-default StructSplat rows both beat the
GaussianImage/Image-GS analogue rows at max-side 160 / 80 iterations, while Instant-GI was not run
without its external script. Stronger claims still need the full ABL-001/ABL-004 sweep, native
external-repo runs where possible, and a declared target metric. See trace node N17 and
`ara/evidence/abl004-kodak4-cross-repo-2026-07-04/`.

## C05: The flanking hypothesis is retired; structured placement is the claim that survives

Across the 2026-07-04 8-image exact-CUDA screen
(`ara/evidence/abl004-stage-screen-8img-cuda-2026-07-04/`), the 2026-07-05 fair density-control
comparison (`ara/evidence/fair-density-control-difficult4-2026-07-05/`), and the completed ABL-006
successive-halving confirmation (`ara/evidence/abl006-complete-2026-07-07/`),
`aniso_flanking` never leads a decision-grade slice and is eliminated at stage 1. ABL-006's
3-seed finalist confirmation gives a budget-specific answer: `aniso_onedge` has the higher
2000-Gaussian mean PSNR but not a significant paired lead over `quadtree_wse`; `quadtree_wse` is
the clear 5000-Gaussian PSNR winner (+0.0930 dB, 95% CI [+0.0168, +0.1700]); and `quadtree_wse`
has a small non-significant 10000-Gaussian PSNR lead while `aniso_onedge` has higher MS-SSIM.
The claim that survives is structure-tensor-driven, density-aware placement, with `quadtree_wse`
as the high-budget PSNR default candidate and `aniso_onedge` as the low-budget/MS-SSIM alternative.
Default retirement is INIT-007 (ADR-0013).

## C06: Structure-tensor loss weighting is searchable, not a default

FIT-012 added `loss_weight=tensor`, which weights only the pixel-loss term by normalized
structure-tensor energy while leaving SSIM and all reported metrics unweighted. The fair-regime
difficult-four slice at budgets 2000/5000 produced 16 paired tensor-vs-none cells: mean PSNR was
effectively neutral (+0.0061 dB), edge MAE improved in 10/16 pairs, but mean AUC fell by -0.0107
and the result split by strategy. Tensor weighting helped `aniso_onedge` (+0.2661 dB mean PSNR)
and hurt `quadtree_wse` (-0.2538 dB mean PSNR). Keep it available as a stage-search axis and
default it off unless a future task proves a narrower promoted recipe. See
`ara/evidence/fit012-edge-weighted-loss-2026-07-07/`.

## C07: Pyramid fitting has a local quality candidate, not a shipped default yet

HIER-003 overturned the stale claim that the image pyramid simply loses final quality, but exposed
an AUC loss for the old 750/750 two-level schedule. HIER-004 repaired that local tradeoff with
explicit per-level iteration schedules: `level_iters=[150, 1350]` on the 0.35/0.65 budget split
beats the 750/750 pyramid control by +0.0601 dB mean PSNR and is AUC-neutral versus single-stage
(+0.0011 mean AUC) on the difficult-four 2k/5k slice. Keep `pyramid=single` as the shipped default
until larger multi-seed confirmation; use 150/1350 as the pyramid quality candidate. See
`ara/evidence/hier003-pyramid-diagnosis-2026-07-07/` and
`ara/evidence/hier004-pyramid-convergence-repair-2026-07-07/`.

## C08: The counted background layer is a low-budget candidate, not a default

CORE-009 added a stage-searchable frozen-geometry Gaussian background layer whose rows count
against `num_gaussians` and whose colors remain learnable. On the difficult-four fair-regime slice
at budgets 1000/2000/5000, `background=frac0.05_grid8` averaged +1.0152 dB PSNR and +0.01412
MS-SSIM over 24 paired cells versus `background=off`, winning PSNR in 22/24 pairs. The effect is
budget-dependent: +1.8768 dB at 1000 rows, +1.1183 dB at 2000 rows, and only +0.0504 dB at 5000
rows, where AUC lost in every pair. Keep the layer searchable and default off. Do not claim a true
additive background/compositing layer until a new ADR and larger confirmation justify changing
renderer semantics. See `ara/evidence/core009-background-layer-2026-07-07/`.

## C09: Feed-forward warm starts need tensor priors and are not default-ready

FF-001 established the learned warm-start path, but the current tiny CNN is not better than the
hand tensor-prior initializer. On the 2026-07-07 held-out Kodak slice, the image-only learned
checkpoint reached only 21.0514 dB (`kodim19`, 512 Gaussians, 200 refinement iterations), while the
image+structure-tensor checkpoint beat random scratch on final PSNR (23.4686 vs 22.9056 dB) and
reached 22 dB faster (69 iterations / 0.128815 s vs 108 iterations / 0.166847 s). The same
tensor-prior checkpoint still lost clearly to `quadtree_wse` (25.3249 dB, AUC 24.1797). Keep
`strategy=feedforward` as an experimental warm-start path; do not promote learned initialization
without a larger architecture or predict-optimize-distill evidence. See
`ara/evidence/ff001-multimage-tensor-ablation-2026-07-07/`.
