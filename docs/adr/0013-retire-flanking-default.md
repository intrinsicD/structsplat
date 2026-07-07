# ADR-0013: Retire flanking as the shipped initialization default

## Context

ADR-0009 shipped `aniso_flanking` as the production initialization default because the original
StructSplat hypothesis expected flanking to improve low-budget edge quality. The larger benchmark
program now answers that hypothesis negatively:

- `ara/evidence/abl004-stage-screen-8img-cuda-2026-07-04/`: `aniso_onedge` won the 2000-Gaussian
  screen slice and `quadtree_wse`/`quadtree_hybrid` led at 5000.
- `ara/evidence/fair-density-control-difficult4-2026-07-05/`: flanking was the weakest
  StructSplat row in the fair density-control comparison.
- `ara/evidence/abl006-complete-2026-07-07/`: ABL-006 completed 728/728 staged cells on Kodak-24
  + COCO4 with exact CUDA, max-side 768, 1500 iterations, and 3-seed finalist confirmation.

ABL-006 gives the decision-grade result. `aniso_flanking`, `quadtree_hybrid`, `iso_blue_noise`, and
Floyd-Steinberg were eliminated at stage 1 by the frozen paired-CI rule. The finalists were
`quadtree_wse` and `aniso_onedge`.

## Decision

Change the shipped initialization default from `aniso_flanking` to `quadtree_wse`.

The default is not budget-conditional. `quadtree_wse` is the significant budget-5000 PSNR winner
against `aniso_onedge` (+0.0930 dB, 95% CI [+0.0168, +0.1700]) and has the higher budget-10000
mean PSNR (+0.0357 dB, 95% CI [-0.0041, +0.0778]). `aniso_onedge` has the higher budget-2000 mean
PSNR, but the paired PSNR CI overlaps zero, and it has the better budget-10000 MS-SSIM. Therefore
`aniso_onedge` stays documented as the low-budget/MS-SSIM alternative rather than becoming the
single shipped default.

`aniso_flanking` remains available as an explicit strategy and stage-search axis value. It is now a
control arm, not the thesis/default.

## Consequences

+ `InitConfig()` and `structsplat fit` now default to `strategy="quadtree_wse"`.
+ Feed-forward tensor-prior fallback defaults now use the shipped init default unless explicitly
  overridden.
+ Stage-search influence mode now measures one-factor deltas around `quadtree_wse`; pre-ADR-0013
  influence runs are not directly comparable because the baseline row changed.
+ `flank_offset_frac` is strategy-aware when omitted: `aniso_flanking` gets the historical 0.5
  offset, while the shipped `quadtree_wse` path gets the measured 0.0 offset. Explicit offsets
  remain supported for ablations.
- A future budget/metric-aware policy may beat any single default, but that is a separate selector
  task. This ADR deliberately chooses one conservative shipped default.

## Links

Supersedes ADR-0009 only for the initialization strategy default. Depends on ABL-006 and BENCH-002.
Feeds INIT-008/FIT-009 only as the new default baseline, not as evidence that those axes are
settled.
