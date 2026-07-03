# StructSplat Comparison Against Related 2D Gaussian Image Repos

This note compares the local `structsplat` approach against local copies of:

- `/home/alex/Documents/GaussianImage`
- `/home/alex/Documents/GaussianImage_plus`
- `/home/alex/Documents/image-gs`
- `/home/alex/Documents/Instant-GI`

The comparison is based on the local README files and implementation paths inspected on
2026-07-02. The initial pass only test-ran `structsplat` (`33 passed in 108.55s` at commit state
`pytest-2026-07-02`). Later ARA entries add current exact-CUDA and cross-repo matrix evidence; keep
the dates/commit states separate when citing numbers. The other repos were not benchmarked through
their native end-to-end pipelines because they require CUDA extension builds, datasets, and in some
cases pretrained checkpoints. Treat runtime and quality claims for those repos as their
repo/paper-code intent unless a specific ARA evidence ID says otherwise.

## High-Level Positioning

| Project | Main bet | Best comparison axis |
|---|---|---|
| StructSplat | Better deterministic initialization from structure tensor + anisotropic blue-noise sampling + edge flanking | Low-budget quality, convergence speed, interpretable ablations |
| GaussianImage | Simple fixed-count 2D Gaussian representation with very fast CUDA rendering and codec path | Published baseline for per-image fitting, rendering FPS, and rate-distortion |
| GaussianImage++ | Boost GaussianImage with direct covariance, adaptive point growth, pruning, and improved quantization | Practical improvement over GaussianImage under variable budgets |
| Image-GS | Content-adaptive gradient/saliency init plus error-guided progressive optimization and LOD | Closest conceptual neighbor for adaptive placement and progressive hierarchy |
| Instant-GI | Learned initializer predicts a coarse Gaussian representation, then lightly fine-tunes | Fast amortized initialization across many images from a learned distribution |

## What StructSplat Actually Contributes

`structsplat` is not primarily a faster renderer today. Its current contribution is a controlled
research harness around initialization:

- It computes a structure tensor once and uses it consistently for density, orientation, and
  flat/edge/corner classification.
- It samples positions with density-adaptive Weighted Sample Elimination, optionally in an
  anisotropic Mahalanobis metric.
- It explicitly flanks edge Gaussians rather than placing them on the discontinuity.
- It supports a coarse-to-fine pyramid where later levels are driven by residual structure.
- It has focused ablation and stage-search harnesses for deciding whether the extra structure
  actually helps.

This makes StructSplat strong as a hypothesis-testing framework. It now has an exact CUDA extension
for its own normalized/additive equations, but the tiled production CUDA/Vulkan path and native
end-to-end codec comparisons are still open.

## Fair Repo-by-Repo Comparison

### GaussianImage

GaussianImage is the clean baseline. Its local code initializes Gaussian positions randomly in
normalized coordinates, optimizes a fixed number of points, and offers Cholesky and scale-rotation
parameterizations backed by a CUDA `gsplat` renderer. The compression path uses quantization-aware
training, half-precision positions, vector-quantized colors, low-bit covariance/scale parameters,
and optional entropy coding analysis.

Compared with GaussianImage, StructSplat changes the front of the pipeline, not the overall idea
of overfitting a single image with 2D Gaussians. The fair question is: with the same renderer,
budget, loss, and iteration limit, does tensor/blue-noise/flanking initialization reach the same
PSNR faster or reach better PSNR at low budgets? StructSplat is designed exactly to answer that.

GaussianImage should still be expected to win on maturity, CUDA throughput, published codec
numbers, and simple reproducibility. StructSplat should only claim advantage if its ablation shows
the initializer improves convergence or low-budget quality under matched conditions.

### GaussianImage++

GaussianImage++ is closer to a practical improvement of GaussianImage. In the local code, it uses
direct 2D covariance parameters, an image-size/point-count scale lower bound (`SLV_init`), pruning
of non-positive-definite covariance entries, and adaptive addition of points at high-error pixels
until `max_num_points` is reached. Its quantization path adds LSQ-style control for positions,
covariance, and colors.

StructSplat and GaussianImage++ both attack poor capacity allocation, but from opposite ends:

- StructSplat tries to start from a good structured distribution before fitting.
- GaussianImage++ starts simpler, then repairs allocation during fitting with residual growth and
  pruning.

GaussianImage++ is more directly useful if the goal is "make GaussianImage better with a stronger
training loop and codec." StructSplat is cleaner if the goal is "is there a geometry/sampling
principle that gives a better first layout?" The strongest fair experiment would combine them:
use StructSplat initialization as the initial point set, then run GaussianImage++ growth/pruning.

### Image-GS

Image-GS is the closest conceptual peer. It supports gradient, saliency, and random position
initialization; error-guided progressive optimization; a natural LOD stack; configurable bit
precision; top-k normalized CUDA rendering; texture stack compression; and rendering at new
resolutions.

The key difference is placement quality. Image-GS uses weighted random sampling from gradient or
saliency maps, then adds more Gaussians from residual-error probability. StructSplat instead uses
blue-noise elimination under a density and anisotropy field, so it is explicitly trying to avoid
clumps while respecting local orientation. StructSplat also distinguishes on-edge versus flanking
edge placement, which Image-GS leaves to optimization.

Image-GS is more mature for applications: it has CUDA kernels, richer metrics/logging, texture
stack support, post-optimization rendering, and a clear compression interface. StructSplat is more
methodically isolated for testing one initialization hypothesis. If StructSplat wins anywhere, it
should be in low-budget or early-iteration regimes where placement regularity matters most.

### Instant-GI

Instant-GI changes the problem: it amortizes initialization over a dataset. Its network predicts a
position field, discretizes it through dithering, constructs geometry through Delaunay/ellipse
processing, predicts Gaussian attributes, and then runs GaussianImage-style fine-tuning. It also
has quadtree and random initializers as baselines. The learned path can dynamically choose the
number of Gaussians produced by the position field.

StructSplat is non-learned and per-image. That is a strength when no training corpus or checkpoint
is available, and a weakness when many images come from the same distribution. Instant-GI should
win on time-to-good-initialization if its pretrained checkpoint generalizes to the target images.
StructSplat is easier to audit, easier to port, and less vulnerable to domain shift.

The fairest interpretation is that Instant-GI is an amortized learned initializer, while
StructSplat is a deterministic analytic initializer. They can be complementary: StructSplat's
tensor density or blue-noise targets could be used as training supervision or as a fallback when
Instant-GI's checkpoint is unavailable.

## Comparison Matrix

| Axis | StructSplat | GaussianImage | GaussianImage++ | Image-GS | Instant-GI |
|---|---|---|---|---|---|
| Initialization | Structure tensor, density-adaptive WSE, anisotropic metric, edge flanking | Random fixed-count | Random plus adaptive residual growth/pruning | Gradient/saliency/random, then residual additions | Learned position field plus dithering/Delaunay/ellipse attributes |
| Orientation/scale prior | Tensor-driven orientation and axis ratio | Learned from random start | Direct covariance with scale lower bound | Learned scale/rotation; initial scale fixed | Network predicts scale/rotation from local geometry/features |
| Budget policy | Fixed budget, optional pyramid allocation | Fixed count | Starts at `num_points`, grows to `max_num_points` | Starts at fraction, progressively adds to fixed total | Point count emerges from predicted/dithered position field, or fixed random baseline |
| Renderer maturity | PyTorch reference; CUDA/Vulkan planned | CUDA `gsplat` sum renderer | CUDA `gsplat` plus modified covariance/raster path | Custom CUDA tile/no-tile renderer with top-k norm | GaussianImage-style CUDA renderer |
| Compression path | Basic post-fit quantization + Morton/zlib + STE QAT | Mature QAT/VQ/entropy-analysis path | Stronger LSQ-style quantization controls | Bit-precision control by parameter group | Not the main local code path |
| LOD/progressive behavior | Explicit prefix-oriented pyramid | No natural LOD in baseline | Adaptive count, not explicitly prefix-LOD | Explicit progressive optimization/LOD stack | Coarse net init then fine-tune, not primarily LOD |
| External dependency | Minimal for reference; metrics optional | CUDA extensions and datasets | CUDA extensions and datasets | CUDA extensions, fused SSIM, optional saliency model | CUDA extensions, pretrained init checkpoint, training data for custom init net |
| Scientific auditability | High: tests, ADRs, ablation harnesses | Medium: compact paper code | Medium: practical but more entangled | Medium-high: configurable and documented | Medium: learned components add training-data dependence |

## Suggested Fair Benchmark

To compare fairly, do not compare README headline numbers. Use one harness and record:

- Same image set: Kodak 24 plus a small DIV2K or COCO subset.
- Same output resolution and color space.
- Same Gaussian budgets: for example 2k, 5k, 10k, 20k.
- Same stopping modes: fixed iterations, fixed wall-clock, and time-to-target PSNR.
- Same metrics: PSNR, SSIM, MS-SSIM, LPIPS/FLIP if available, init time, fit time, render FPS,
  actual bpp after each repo's codec.
- Same seeds where stochastic sampling is used.
- Separate representation results from codec results, because the repos define bpp differently.

The most informative first test is:

1. Disable progressive growth in Image-GS/GaussianImage++ and compare only initial layouts at fixed
   count and fixed iterations.
2. Then enable each repo's native growth/progressive mode and compare end-to-end practical quality.
3. Finally test hybrids, especially StructSplat init plus GaussianImage++ or Image-GS training.

## Bottom Line

StructSplat is best understood as an initialization research contribution, not yet as a complete
replacement for the mature CUDA codecs. GaussianImage is the baseline to beat, GaussianImage++ is
the strongest practical GaussianImage-style training/codec improvement, Image-GS is the closest
peer for progressive content-adaptive placement, and Instant-GI is the learned amortized alternative.

The decisive claim for StructSplat should be narrow: under matched renderer/loss/budget settings,
does structure-tensor anisotropic blue-noise flanking improve low-budget quality or convergence
speed? The local repo is set up to test that claim, but it still needs full cross-repo benchmarks
before making stronger performance claims.
