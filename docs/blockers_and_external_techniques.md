# Remaining Blockers and External Techniques

Analysis date: 2026-07-02

Inputs:

- `results/quadtree_init_compare/summary.md`
- `results/optimization_followup/summary.md`
- `results/stage_search_coco4_after_update/best_vs_previous.md`
- Local repos: `GaussianImage`, `GaussianImage_plus`, `image-gs`, `Instant-GI`
- Literature: GaussianImage, GaussianImage++, Image-GS, Instant-GI, 3DGS, 3DGS surveys and densification papers.

## Current Signal

The latest cap/quadtree run shows that scale control is a real improvement, not just a visual cleanup:

- `quadtree_wse_feature_cap12`: 24.6528 PSNR, 0.95769 MS-SSIM, 1.360 s total.
- `stage_top1_feature_cap12`: 24.6423 PSNR, 0.95736 MS-SSIM, 1.269 s total.
- `stage_top1` uncapped: 24.5490 PSNR, 0.95824 MS-SSIM, 2.293 s total, with mean final max scale 72.740 px.
- `current_baseline`: 24.3378 PSNR, 0.95590 MS-SSIM, 2.062 s total.

So the feature cap improved mean PSNR by about +0.31 dB against the current baseline and about +0.10 dB against the previous stage-search best while cutting total time by roughly 0.9 to 1.0 s versus uncapped `stage_top1` on the 8-image COCO subset.

The remaining issue is that the best candidate changes by image and metric. `quadtree_wse_feature_cap12` wins PSNR on 3/8 images, `stage_top1_feature_cap12` wins on 2/8, uncapped `stage_top1` wins on 1/8, and uncapped `quadtree_hybrid_agg_variance` wins on 2/8. For MS-SSIM, the winners are even more scattered. This means a fixed initializer/cap policy is probably leaving performance on the table.

## Main Blockers

1. Fixed policy rather than adaptive density control

StructSplat is still mostly being evaluated as a fixed-budget initializer followed by fitting. The fitter has residual-add and prune support, but the current winning cap/quadtree comparison did not tune those paths as the main method. GaussianImage++, Image-GS, Instant-GI, and 3DGS literature all point in the same direction: allocation has to change during fitting, not only before it.

2. Residual-add is too simple

Current `residual_add` adds Gaussians at top residual pixels with a uniform base scale and zero rotation. That is closer to a useful baseline than a strong densifier. Image-GS uses residual maps but initializes new colors from the residual itself. GaussianImage++ adds points from high-error pixels under a bounded max count. 3DGS variants use gradient/error/opacity/scale criteria to decide split, clone, or prune. StructSplat should use residual plus local tensor/scale/cap information when adding new points.

3. Scale cap solved overshoot but not detail recovery

The cap removes long spiky Gaussians and reduces runtime, but it can also reduce the broad support that previously masked missing detail. The next step is to pair capped large Gaussians with targeted small residual Gaussians. Otherwise the cap is acting only as a clamp, not as a capacity reallocation mechanism.

4. Renderer is still the production bottleneck

GaussianImage, GaussianImage++, Image-GS, and Instant-GI all rely on CUDA splatting paths. StructSplat's renderer is a reference PyTorch implementation. The Python tile prototype matched numerically but was 10x to 12x slower than the current reference path in the small benchmark, so the viable speed path is a compiled tile renderer, not more Python tiling.

5. No image-adaptive budget policy

Instant-GI explicitly targets the fixed-count weakness by deriving the number of Gaussians from image complexity. StructSplat currently uses fixed budgets in the benchmarks. COCO results show some images barely benefit while others gain much more, which suggests entropy/detail maps should control either the budget or the growth schedule.

6. Initial attributes are underpowered

Quadtree aggregate variants often have stronger init PSNR, but that does not consistently translate into best final PSNR. Instant-GI predicts position, scale, rotation, opacity, and color from local features plus Delaunay/ellipse geometry. StructSplat currently uses tensor priors and sampled or aggregate colors, but it does not estimate local affine/ellipse attributes from a neighborhood graph.

7. Objective mismatch remains

The best PSNR candidate is not always the best MS-SSIM candidate. Uncapped `stage_top1` still has slightly higher aggregate MS-SSIM than capped top candidates. Image-GS reports PSNR, SSIM, MS-SSIM, LPIPS, and FLIP; StructSplat can compute some of these, but the search has mostly optimized PSNR/MS-SSIM. We need a target metric choice before optimizing visual quality claims.

8. Compression path is not yet competitive

GaussianImage uses vector quantization and quantization-aware training; GaussianImage++ adds attribute-separated learnable scalar quantizers and QAT. 3DGS compression surveys emphasize pruning plus scalar/vector quantization and entropy coding. StructSplat has basic quantization/QAT support, but density control and codec decisions are not yet jointly optimized.

## Techniques Worth Borrowing

### From GaussianImage

- CUDA sum rasterization and tile projection path.
- Fixed, simple representation as the sanity baseline.
- QAT and residual vector quantization for color/attributes.
- Adan optimizer as a candidate for longer high-budget fits.

### From GaussianImage++

- Distortion-driven densification: allocate new primitives where current reconstruction error is high.
- Context-aware low-pass/covariance bounds to avoid badly conditioned or wasteful large primitives.
- Prune invalid/non-positive-definite covariance entries.
- Attribute-separated scalar quantization and QAT.

### From Image-GS

- Gradient/saliency initialization plus error-guided progressive optimization.
- Residual-color initialization for newly added Gaussians.
- Smooth LOD stack from progressive additions.
- Wider metric set: PSNR, SSIM, MS-SSIM, LPIPS, FLIP.
- Compiled CUDA rendering with top-k normalization and optional tiled/no-tile modes.

### From Instant-GI

- Position Probability Map followed by dithering/discretization.
- Dynamic point count based on image entropy/detail.
- Delaunay/ellipse geometry for scale and rotation estimates.
- Learned or amortized initializer for batches of similar images.
- The quadtree baseline plus Delaunay post-process is a useful non-learned variant.

### From 3D Gaussian Splatting Literature

- Interleaved optimization and density control rather than one-shot initialization.
- Clone/split/prune cycles based on positional gradients, opacity/activity, scale, and screen-space footprint.
- Improved density-control criteria: pixel-error-driven densification, homodirectional gradients, long-axis splitting, recovery-aware/adaptive pruning, and dynamic thresholds.
- Anti-aliasing and scale-aware filters: Mip-Splatting-style constraints map naturally to 2D scale caps and render-time dilation/filtering.
- Hierarchical anchors/LOD: Scaffold-GS and Octree-GS suggest that quadtree/octree structure should not only initialize points; it can also organize levels and runtime selection.

## Prioritized Next Experiments

1. Tuned capped residual densification

Start at 512 capped Gaussians, then add 128 or 256 residual Gaussians in 2 to 4 waves. New Gaussians should inherit local tensor orientation, feature cap, and either target color or residual color depending on renderer. Compare against fixed 512 and fixed 640/768 from scratch.

2. Better split criterion

Replace simple top residual at Gaussian centers with a combined score:

`score = residual_under_support * activity * scale_factor * feature_energy`

Then split long/high-error Gaussians along their major axis or add small residual children near support error maxima. This is the 2D analog of 3DGS split/clone/densify logic.

3. Image-adaptive budget/growth router

Compute cheap image statistics from the tensor/quadtree pass: entropy, high-energy mass, edge/corner fraction, residual concentration after a short fit. Use those to choose between `stage_top1_feature_cap12`, `quadtree_wse_feature_cap12`, and `quadtree_hybrid_agg_feature_cap12`, and to decide whether to grow by 0, 128, or 256 points.

4. Geometry-derived quadtree attributes

Extend quadtree initialization with a Delaunay/ellipse or local second-moment pass. Use aggregate color, local covariance, and feature cap together. This tests the Instant-GI geometric post-process without requiring a learned network.

5. Compiled renderer spike

Do not optimize the Python tile prototype further. Build a minimal CUDA/Triton/C++ extension or adapt `gsplat` for the current normalized/additive renderer. The goal is exact or near-exact parity on the 8-image benchmark, then measure fit-loop time.

### 2026-07-02 Follow-up: Exact CUDA Renderer

StructSplat now has an owned CUDA extension for the exact clipped-support normalized and additive
equations. `renderer=cuda` matches normalized reference output with mean absolute difference
`4e-8` and max difference `5.4e-7` on the 512-Gaussian COCO parity smoke; `renderer=cuda_additive`
matches additive with mean absolute difference `1.2e-7` and max difference `2.9e-6`. On the
four-image COCO 80-iteration fit loop, exact CUDA reduced mean normalized wall time from `1.181s`
to `0.181s` (`6.51x`) while preserving PSNR/MS-SSIM. The local conda `libstdc++` still requires
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` for CUDA extension loading. See ARA evidence
`cuda-exact-renderer-2026-07-02` / trace node `N15`.

6. Metric-aligned search

Run the top candidates with LPIPS/FLIP enabled on a smaller image set. Decide whether the project optimizes PSNR, MS-SSIM, or a multi-metric score. Current capped candidates improve PSNR/runtime but do not dominate MS-SSIM.

7. Codec-aware pass after density control

Once adaptive growth/prune is stable, rerun rate-distortion with attribute-separated quantization. Compression before density control will mainly encode the current allocation mistakes more cheaply.

## Recommended Immediate Direction

The best next implementation is not a new initializer alone. It is capped quadtree/stage initialization plus a tuned residual densification loop that keeps the scale cap and adds detail where the cap exposes residual error. That directly addresses the current observed artifacts and aligns with the strongest repeated idea from GaussianImage++, Image-GS, Instant-GI, and 3DGS densification work.
