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
comparison (`ara/evidence/fair-density-control-difficult4-2026-07-05/`), and the first ABL-004
confirmation shards, `aniso_flanking` never leads a decision-grade slice: `aniso_onedge` wins at
budget 2000 (+0.24 dB paired, 7/8 wins), `quadtree_wse`/`quadtree_hybrid` lead at >=5000, and
flanking is the weakest StructSplat row in the fair comparison. Its only observed niche is
tiny-budget/short-fit cells, and only bundled with unrelated fitter knobs
(`ara/evidence/merge001-coco-cuda-confirmation-2026-07-06/`, codex top1). The claim that
survives: structure-tensor-driven, density-aware placement (on-edge, quadtree-WSE) beats matched
unstructured and residual-growth analogues in every fair PSNR slice. Default retirement is
INIT-007 (ADR-0013); the remaining confirmation runs as successive halving under ABL-006.
