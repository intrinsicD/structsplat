# Claims

## C01: StructSplat is primarily a controllable stage-search harness

StructSplat's current strength is not a final codec claim; it is that initialization, renderer,
fitting, densification, pyramid, and codec choices are exposed as testable stages with JSON/CSV
evidence. See `benchmarks/stage_search.py`, ADR-0010, and ABL-002.

## C02: Feature-aware scale caps are a strong default candidate, not a settled law

Feature-adaptive caps reduced scale outliers and improved the small COCO screens, but caps can
trade broad low-frequency support against detail recovery. They should stay in stage-search until
ABL-004 confirms them across larger image sets, budgets, and seeds. See H02 and ADR-0012.

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
