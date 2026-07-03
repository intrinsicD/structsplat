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

The current matched matrix shows StructSplat leading PSNR, while GaussianImage and Instant-GI-style
policies remain competitive or better on MS-SSIM/LPIPS slices. Stronger claims need ABL-004-sized
evidence and a declared target metric. See trace node N17.
