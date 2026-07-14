# StructSplat and the 2D Gaussian image frontier

**Updated:** 2026-07-14

**Scope:** primary papers/official pages plus provenance-checked local executions.
**Detailed audits:** `ara/evidence/storage-budget-168k-sota-audit-2026-07-13.md` and
`ara/evidence/bench007-stage1-killing-pilot-2026-07-14/run.md`.

## Bottom line

StructSplat is not established as a state-of-the-art image codec. The completed 168 KiB experiment
is high-rate local-policy evidence:

- 40 methods × four COCO training images × two seeds = 320 completed cells;
- 71.68–81.15 analytical bpp at the prepared resolutions;
- roughly 22 bpp for the actual SSPL1 streams;
- about 17.99 bpp for the corresponding lossless target PNGs;
- mostly local paper-inspired controls rather than native external executions.

BENCH-007 has now answered the narrower development question negatively. Tensor-metric blue-noise
placement beats the strongest local gradient control at 0.5 bpp but ties it at 1.0 bpp, misses the
frozen BD-rate magnitude, costs 47.5% more, and violates the texture guard. Stage 2 was not
authorized. The strongest current output is the controlled actual-rate benchmark and bounded
negative result, not a positive tensor-WSE compression method.

## Current primary-source map

These results do not form one leaderboard: datasets, rate definitions, renderers, optimization
horizons, learned priors, and hardware differ.

| Method | What it establishes | Consequence for StructSplat |
|---|---|---|
| [GaussianImage](https://arxiv.org/abs/2403.08551) (ECCV 2024) | Foundational per-image 2D Gaussian representation and quantization path. | Baseline renderer/optimizer/codec family; native fixed-horizon results are tradeoffs, not common-method ablations. |
| [Image-GS](https://arxiv.org/abs/2407.01866) (SIGGRAPH 2025) | Gradient-informed allocation, error-guided progressive optimization, top-K normalized rendering, random access, and LOD. | Broad content-adaptive progressive allocation and normalized-mixture territory is occupied. |
| [GaussianImage++](https://ojs.aaai.org/index.php/AAAI/article/view/37572) (AAAI 2026) | Distortion-driven growth, covariance filtering, and attribute-separated learned quantization. | Local residual/covariance transplants do not reproduce the native method; full native RD is required. |
| [Structure-Guided Allocation](https://arxiv.org/abs/2512.24018) (2025) | SLIC/Sobel allocation, geometry consistency, and structure-adaptive covariance precision; reports -43.44% Kodak and -29.91% DIV2K BD-rate versus GSImage. | Closest handcrafted direct baseline. Broad structure-aware allocation/orientation/precision novelty is occupied. |
| [Soft Anisotropic Diagrams](https://arxiv.org/abs/2604.21984) (SIGGRAPH 2026) | Top-K normalized anisotropic ownership with independent reach/temperature and removal-delta pruning. Its reported BPP is a parameter proxy, not entropy-coded bytes. | Strong representation-level threat and a causal bridge target; never compare its proxy BPP with SSPL1 actual BPP as if identical. |
| [SGI](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_SGI_Structured_2D_Gaussians_for_Efficient_and_Compact_Large_Image_CVPR_2026_paper.pdf) (CVPR 2026) | Seed-organized neural Gaussians, context entropy coding, and multiscale fitting on 27–76 MP images. | Defines the high-resolution structured/entropy-coded frontier absent from the 160-pixel study. |
| [AIR](https://arxiv.org/abs/2605.20820) (2026) | Learned stage-wise residual prediction and adaptive quantization with feed-forward encoding. | Defines the amortized encoder frontier. The local four-image max-side-256 run is only environment evidence. |
| [P-GSVC](https://arxiv.org/abs/2603.10551) (MMSys 2026) | Joint base/enhancement Gaussian layers for quality and resolution scalability. | Progressive WSE order is an engineering mechanism unless preserved and optimized in an embedded stream. |
| [CGVQ](https://arxiv.org/abs/2607.05667) (2026) | Cluster-guided codebooks; reports about 20% lower bpp at similar quality versus its baseline. | Generic clustered VQ is occupied; a new codec needs a structure-conditioned rate argument. |
| [Contour-Aware 2DGS](https://arxiv.org/abs/2512.23255) (2025/2026) | Segmentation-region-constrained rasterization reduces cross-boundary mixing at low counts. | Direct threat to CORE-007's old broad gate. Remaining lane is segmentation-free flux with every boundary bit counted. |
| [WIPES](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_WIPES_Wavelet-based_Visual_Primitives_ICCV_2025_paper.html) (ICCV 2025) | Localized wavelet visual primitives provide a frequency-bearing alternative to Gaussians. | Required direct control for CORE-008; Gaussian-only comparisons cannot identify the best primitive family. |
| [Instant-GI](https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Instant_GaussianImage_A_Generalizable_and_Self-Adaptive_Image_Representation_via_2D_ICCV_2025_paper.html) and later learned samplers | Learned adaptive Gaussian fields with little or no per-image fitting. | Learned initialization is crowded; StructSplat's viable distinction is training-free interpretability with measured RD or robustness value. |

For ordinary image compression, the outside-class frontier also matters. A practical learned codec
at CVPR 2026 reports subjective bitrate gains against AV1/AV2/VVC/ECM/JPEG-AI while decoding large
images on-device
([primary source](https://openaccess.thecvf.com/content/CVPR2026/html/Tatwawadi_What_Matters_in_Practical_Learned_Image_Compression_CVPR_2026_paper.html)).
A Gaussian method is not overall compression SOTA merely because it leads other Gaussian methods.

## What the local native evidence says

Native results are intentionally kept separate from common-renderer mechanism controls:

| Native lane | Bounded result | Missing for a strong claim |
|---|---|---|
| Image-GS official environment, fixed N, 500 steps, COCO4 at max-side 160 | StructSplat has higher paired final PSNR/proxy-MS-SSIM; start-count and timing semantics differ. | Native resolution, rate curve, packed stream, broader held-out data. |
| Image-GS `siggraph25`, 5k, one seed | Metric tradeoff: Image-GS has higher proxy MS-SSIM; StructSplat checkpoint has higher PSNR and better LPIPS. | Multi-seed, native-resolution, actual-rate confirmation. |
| GaussianImage official environment, 5k | Speed/quality tradeoff; neither method is a uniform winner. | Released 50k + 50k QAT Kodak protocol and a self-contained rate definition. |
| GaussianImage official environment, N=5,376, 10k | 35.657 dB, 6.392 s, about 4,412 FPS on the four small targets; -13.142 dB but +8.288 s versus the historical StructSplat row. | This is float representation evidence, not codec RD. |
| AIR checkpoint, four inputs at max-side 256 | 25.254 dB, 37.0 ms inference, native-reported quantized 4.328 bpp. | Same resolution/denominator, central stream audit, paper-protocol dataset. |

No native evidence above supports a global implementation ranking.

## The two comparison lanes

### Common-mechanism lane

Use one StructSplat representation, renderer, fitter, codec, candidate search, and actual byte
definition. Change exactly one allocation mechanism. This lane answers causal questions and must
label transplants `local_<mechanism>_control`.

BENCH-007 completed:

- tensor/on-edge WSE;
- shipped quadtree-WSE;
- SLIC/Sobel structure classes;
- Image-GS-style gradient sampling;
- uniform Euclidean WSE;
- random placement.

The eight-image DIV2K-train killing pilot completed 288 fits and 1,152 validated streams. The local
gradient arm was the strongest direct control and the promotion gate failed. DIV2K validation and
Kodak confirmation remain unrun because the gate denied Stage 2.

### Native-authentic lane

Execute official code at its intended renderer, optimizer, checkpoint, resolution, rate definition,
and horizon. Record repository/build/environment provenance and centrally rescore decoded pixels.
Keep these rate columns distinct:

- self-contained codec bytes/bpp;
- analytical or parameter bpp;
- checkpoint bytes;
- null when no complete stream exists.

BENCH-008 is not authorized by the negative Stage-1 result. A future field/renderer interaction
study would need a new question and cannot replace native-authentic rows.

## Claim boundary

Currently defensible:

> StructSplat is an interpretable, training-free structural-prior and causal-experimentation
> substrate for normalized local image representations.

Not established:

- state-of-the-art image compression;
- broad novelty for structure-aware allocation or orientation;
- native superiority over named external methods;
- progressive-codec novelty from WSE ordering alone;
- boundary or hybrid-primitive novelty without Contour-Aware 2DGS/WIPES controls;
- actual-rate superiority from Gaussian count, float payload, analytical BPP, or checkpoint size.

The current tensor-WSE compression claim is closed. Any expansion requires a materially new
mechanism, null, disjoint development screen, and later held-out confirmation; the failed pilot and
untouched Stage-2 set cannot be used for post-hoc rescue.
