# 2D Gaussian image fields: state of the art in quality, speed, convergence, and storage

**Review date and literature cutoff:** 2026-08-04

**Repository:** StructSplat

**Evidence class:** primary-source literature synthesis; no new native execution

**Status:** producer-authored and awaiting independent scientific review

## Executive answer

There is no single method that simultaneously maximizes reconstruction quality, encoding speed,
rendering speed, and reduction relative to an original image file. The apparent winners change when
the representation, decoder dependency, image scale, rate definition, and timing boundary change.
The strongest conclusions supported by the reviewed primary sources are:

1. **GaussianImage remains the foundational high-throughput additive formulation.** Its small
   eight-parameter RGB Gaussians and order-independent rasterizer establish a useful warm-rendering
   baseline, with roughly 1,700--2,100 author-reported FPS on its Kodak/DIV2K protocol. Its published
   compression path mixes fixed quantization, residual vector quantization, and optional bits-back
   accounting; the paper's rate should not be read as a universally available, independently audited
   cold-decodable file format.
2. **Image-GS is the clearest strict-Gaussian choice when an editable, random-access, naturally
   progressive field matters.** Top-K normalized rendering, content-adaptive initialization, and
   error-guided growth give strong low-parameter quality and natural levels of detail. Its reported
   rate is raw FP16 parameter payload without entropy coding, not a complete image-codec rate.
3. **SGI is the strongest complete high-resolution Gaussian coding package found in this review.**
   It transmits a structured seed representation, networks, hash/context state, GPCC-coded positions,
   and arithmetic-coded attributes, and reports the most convincing joint gains in file size,
   convergence, and high-resolution quality. Its fitting still takes tens to hundreds of minutes,
   so it is a storage/scale leader rather than an interactive encoder.
4. **Structure-Guided Allocation is the strongest direct low-rate improvement reported within the
   GaussianImage codec lineage.** It reports Kodak/DIV2K BD-rate reductions of 43.44%/29.91% against
   GSImage. However, its fixed/adaptive attribute accounting and inherited color-codebook path have
   not been demonstrated here as a self-contained, byte-audited file package.
5. **AIR is the clearest amortized latency leader among strict 2D Gaussian image predictors.** It
   reports 160--300 ms image-to-field inference with stage-wise residual prediction and no per-image
   test-time fitting. Its 3.28-bpp adaptive-quantization result is parameter accounting rather than a
   demonstrated entropy-coded stream, and its large shared predictor is not included in that rate.
6. **For pure representation quality and fitting speed, the strongest threat is not Gaussian.**
   Soft Anisotropic Diagrams (SAD) reports materially higher quality than Image-GS at matched packed
   parameter budgets and 46.00 dB on Kodak in 2.2 seconds, but uses soft anisotropic ownership sites
   rather than Gaussian kernels and explicitly reports a parameter proxy without entropy coding.
   WIPES similarly shows that frequency-bearing local primitives can improve high-frequency fidelity.
7. **No reviewed paper establishes the best compression ratio relative to the supplied original
   image file.** Most papers report bits per pixel, Gaussian count, raw/quantized parameter payload,
   or reduction against another Gaussian representation. SmartSplat and GaussianVision explicitly
   normalize to raw RGB rather than the source PNG/JPEG; several other methods omit headers,
   codebooks, checkpoints, or entropy streams. Literal original-file compression ratio is therefore
   unknown for nearly the entire literature and must be measured, not inferred.
8. **2D Gaussian fields are not established as the overall still-image compression state of the
   art.** Their compelling advantages are fast repeated rendering, continuous coordinates, sparse
   spatial queries, editability, LOD/progressive access, and direct downstream use. Mature raster and
   learned codecs remain the required outside-class rate-distortion controls.

The best research direction is therefore a **hybrid Pareto design**, not an attempt to copy one
paper: choose field semantics under a controlled renderer factorial; combine a strong initializer,
regional allocate/merge, staged fitting, and fused kernels; then price every decision through an
actual cold-decodable codec. Add a learned predictor only when its model cost and training
break-even are acceptable, and add SGI-style seed structure only if it beats a complete direct
codec after all bytes and query costs are counted.

## 1. Review question and boundary

### 1.1 Operational question

Given an input image, which methods produce an explicit or explicit-queryable two-dimensional
field of anisotropic kernels and lie on a useful Pareto frontier for:

- reconstruction quality: PSNR, MS-SSIM/SSIM, and LPIPS;
- convergence: iterations, wall-clock image-to-field time, and time to target quality;
- performance: cold decode, first render, warm rerender, throughput, memory, and random access;
- storage: actual self-contained bits per pixel and complete package bytes; and
- reduction relative to the exact original image file supplied to the encoder?

The review uses **strict Gaussian** for a field whose reconstructed image is produced from 2D
Gaussian kernels. Methods that predict Gaussian locations but rasterize with a 3DGS formulation are
marked as boundary cases. Non-Gaussian ownership diagrams, wavelet-bearing splats, and hybrid
implicit/explicit splats are included only as direct controls because they reveal which limitations
come from the Gaussian primitive itself.

### 1.2 Included evidence

The search covered peer-reviewed papers, official proceedings, arXiv/OpenReview manuscripts, and
official project or code pages available by 2026-08-04. Query families combined:

- `2D Gaussian image representation`, `GaussianImage`, `Gaussian image compression`;
- `image Gaussian splatting adaptive initialization allocation densification`;
- `2D Gaussian quantization entropy coding progressive random access`;
- `amortized feed-forward Gaussian image reconstruction`;
- `large ultra-high-resolution image Gaussian field`; and
- forward/backward citation searches from GaussianImage, Image-GS, SGI, AIR, and SAD.

The last targeted date sweep covered 2026-07-01 through 2026-08-04. The newest directly relevant
source found was Gaussian Texture Compression (submitted 2026-07-30), an adjacent shared-geometry texture
stack rather than an ordinary single-image codec. Application papers whose goal is enhancement,
dehazing, super-resolution, segmentation, or 3D reconstruction were excluded unless they supplied a
transferable image-to-field mechanism or system result.

### 1.3 Evidence hierarchy

| Level | Meaning | What it can support |
|---|---|---|
| A | Complete encoded package, every per-image dependency counted, cold decode demonstrated | Actual rate, package size, decode, and rate-distortion claims within the paper protocol |
| B | Quantized/packed parameter accounting, but no complete entropy stream or omitted shared state | Representation-rate comparison only; not a file-codec claim |
| C | Float parameters or Gaussian-count budget | Fitting quality, convergence, rendering, and memory only |
| D | Abstract/metadata, adjacent primitive, downstream system, or donor mechanism | Research direction and required control only |

Author-reported numbers below remain author-reported. Different datasets, image scales, hardware,
losses, renderer semantics, and timing boundaries are not pooled into a synthetic ranking. No row is
a repository-native reproduction. Existing StructSplat analogues and external native execution are
separate work owned by BENCH-005/BENCH-007 and are not upgraded by this review.

## 2. The representations are not interchangeable

### 2.1 Additive Gaussian fields

GaussianImage and much of its lineage use an order-independent additive image:

\[
  \hat I(p)=\sum_{i=1}^{N} c_i\,G_i(p),\qquad
  G_i(p)=\exp\left[-\tfrac12(p-\mu_i)^\top\Sigma_i^{-1}(p-\mu_i)\right].
\]

Color may be premultiplied by a learned weight/opacity. This formulation is simple, fully
differentiable, and fast, but overlapping splats add energy and a finite field needs to learn both
coverage and amplitude.

### 2.2 Normalized mixture fields

Image-GS and StructSplat-like representations normalize local contributions:

\[
  \hat I(p)=
  \frac{\sum_{i\in\mathcal A(p)} c_i w_i(p)}
       {\sum_{i\in\mathcal A(p)} w_i(p)+\epsilon}.
\]

Normalization gives constant-color reproduction and local ownership-like behavior, but the active
set, denominator, background policy, and truncation become part of the semantics. Image-GS uses a
top-K local mixture; SAD generalizes ownership further with anisotropic distances plus independent
radius and temperature.

### 2.3 Structured implicit Gaussians

SGI does not transmit every Gaussian independently. It transmits seeds and compact networks that
generate local Gaussians. This trades direct primitive access for spatial regularity and much better
entropy structure. It is still queryable as a Gaussian representation, but rate, cold decode, and
random access must include the generative state.

### 2.4 Richer and hybrid fields

WIPES modulates a Gaussian envelope with a sinusoidal carrier; SAD replaces kernel accumulation with
a soft anisotropic diagram; PA-G2DS adds learned implicit coefficients to pixel-aligned generalized
splats; GTC shares Gaussian geometry across texture maps and mip levels. These methods are not strict
drop-in Gaussian codecs. They are important because a Gaussian-only comparison cannot determine
whether the best primitive, ownership rule, or shared structure has already moved outside the class.

## 3. Rate and timing definitions

### 3.1 Complete rate

For image dimensions \(H\times W\), the primary rate is:

\[
  R_{\mathrm{complete}} = \frac{8B_{\mathrm{package}}}{HW}\quad\text{bits/pixel},
\]

where `B_package` includes, as applicable:

- dimensions, colorspace, transfer function, alpha, and format/version headers;
- primitive count, positions, covariance/scale/rotation, color, opacity, and background;
- ordering, tile indexes, LOD/progressive tables, masks, and segmentation state;
- quantizer scales/offsets, codebooks, entropy tables, and termination/padding;
- per-image neural weights, hash grids, latent seeds, and context-model state; and
- every side file needed for a fresh process to reproduce the decoded raster.

A global pretrained model may be treated as a deployed decoder dependency only if its exact version
and size are reported separately. The per-image rate must then be accompanied by total installation
size and an amortization/break-even analysis. Hiding a 1--4 GB predictor behind a few-byte Gaussian
field is not a complete system comparison.

### 3.2 Original-file compression ratio

For the exact supplied source file:

\[
  CR_{\mathrm{file}}=\frac{B_{\mathrm{source\ file}}}{B_{\mathrm{package}}}.
\]

`CR_file > 1` means the Gaussian package is smaller. The corresponding byte saving is
`1 - 1/CR_file`. This is a secondary metric because the numerator depends on whether the source is
PNG, JPEG, AVIF, TIFF, camera raw, includes alpha/ICC/EXIF metadata, or was already lossily encoded.
Two identical decoded pixel arrays can therefore have very different `CR_file` values.

The scientifically comparable primary result is a rate-distortion curve against a standardized
decoded target, using complete bpp. Literal file ratio is useful for the user's storage workflow,
but must be reported separately for:

1. the exact supplied original file;
2. a canonical lossless PNG of the evaluated pixels; and
3. raw RGB bytes, if desired only as an engineering reference.

### 3.3 Four clocks, not one

The literature often calls different quantities “speed.” A useful report separates:

- **offline training:** cost to learn a shared predictor or codebook;
- **encode/conversion:** input pixels to a finalized field or bitstream;
- **cold decode/first render:** fresh process and serialized package to pixels; and
- **warm render/query:** an already decoded field to a frame, tile, crop, or LOD.

Iterations are not a portable convergence metric: kernels, resolutions, losses, and update work
differ. Wall-clock time-to-quality with hardware, peak memory, and preprocessing included is the
minimum fair convergence evidence.

## 4. Primary-source landscape

### 4.1 Master taxonomy

| Method | Venue/year | Class | Main mechanism | Evidence level and rate semantics |
|---|---|---|---|---|
| [GaussianImage](https://arxiv.org/abs/2403.08551) | ECCV 2024 | Strict additive Gaussian | Eight parameters per splat; Cholesky or rotation-scale covariance; additive CUDA rasterizer; QAT/RVQ codec | B: quantized/codebook and optional bits-back accounting; released cold stream is not established by this review |
| [Image-GS](https://arxiv.org/abs/2407.01866) | SIGGRAPH 2025 | Strict normalized Gaussian | Gradient/uniform initialization, sampled colors, top-K normalization, error-guided growth, natural LOD | B/C: FP16 payload bpp, explicitly without entropy coding |
| [Large Images are Gaussians (LIG)](https://arxiv.org/abs/2502.09039) | AAAI 2025 | Strict additive Gaussian | Two-stage Laplacian-of-Gaussian coarse/residual fitting at million-splat scale | C: float representation; multi-GB memory, not a codec |
| [EigenGS](https://openaccess.thecvf.com/content/CVPR2025/html/Tai_EigenGS_Representation_From_Eigenspace_to_Gaussian_Image_Space_CVPR_2025_paper.html) | CVPR 2025 | Strict Gaussian with dataset prior | PCA/eigenspace field initialization followed by per-image refinement | C: basis/checkpoint and amortized storage excluded |
| [Instant-GI](https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Instant_GaussianImage_A_Generalizable_and_Self-Adaptive_Image_Representation_via_2D_ICCV_2025_paper.html) | ICCV 2025 | Strict Gaussian with learned prior | ConvNeXt U-Net predicts density/parameters; error-diffusion sampling controls count; short refinement | C: count/float field; 3--4 GB predictor plus runtime memory is separate |
| [WIPES](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_WIPES_Wavelet-based_Visual_Primitives_ICCV_2025_paper.html) | ICCV 2025 | Adjacent richer primitive | Morlet-like carrier inside an anisotropic Gaussian envelope | C/D: no coded storage; extra frequency parameters |
| [GaussianImage++](https://ojs.aaai.org/index.php/AAAI/article/view/37572) | AAAI 2026 | Strict additive Gaussian | Distortion-driven densification, content-aware filtering, learned per-attribute quantization | B: fixed learned quantization accounting; no demonstrated entropy package |
| [SmartSplat](https://arxiv.org/abs/2512.20377) | AAAI 2026 | Strict Gaussian | Gradient/color-variance sampling, exclusion, scale/color initialization, UHR tiling | C: Gaussian count derived from an assumed seven bytes/splat and raw RGB ratio; trained fields are float |
| [Structure-Guided Allocation](https://arxiv.org/abs/2512.24018) | 2025 preprint | Strict additive Gaussian | SLIC/Sobel structural classes, geometry consistency, adaptive covariance precision | B: fixed/adaptive attribute accounting plus inherited color RVQ; no byte-audited package here |
| [Contour-Aware 2DGS](https://arxiv.org/abs/2512.23255) | ICCE 2026 | Strict Gaussian with external regions | Segmentation-region-constrained rasterization and warmup | C/D: segmentation/mask cost excluded; no codec |
| [Fast-2DGS](https://openaccess.thecvf.com/content/WACV2026W/WVAQ/html/Wang_Fast_2DGS_Efficient_Image_Representation_with_Deep_Gaussian_Prior_WACVW_2026_paper.html) | WACV workshop 2026 | Strict Gaussian with learned prior | Conditional heatmap sampler and dense attribute regressor; optional refinement | C: count/float field; predictor and 19-hour training excluded |
| [EllipssianNet](https://openaccess.thecvf.com/content/WACV2026/html/Kim_EllipssianNet_Image-guided_Sampling_of_2D_Gaussians_for_Gaussian_Splatting_WACV_2026_paper.html) | WACV 2026 | Gaussian initializer boundary | Voronoi-synthetic center/covariance predictor; colors sampled from image; 3DGS-style fitting | C/D: initializer quality only; not a 2D image codec |
| [SGI](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_SGI_Structured_2D_Gaussians_for_Efficient_and_Compact_Large_Image_CVPR_2026_paper.html) | CVPR 2026 | Structured implicit Gaussian | Seeds generate local Gaussians; coarse-to-fine fitting; hash/context model; GPCC and arithmetic coding | A: clearest complete coded package in the reviewed strict-Gaussian set |
| [GaussianVision](https://openaccess.thecvf.com/content/CVPR2026/html/Omri_GaussianVision_Vision-Language_Alignment_from_Compressed_Image_Representations_using_2D_Gaussian_CVPR_2026_paper.html) | CVPR 2026 | Gaussian system/downstream | Structured initialization, luminance pruning, batched CUDA fitting, Gaussian-native vision encoder | C/D: 3--23.5x against raw pixels, not supplied files or a codec |
| [P-GSVC](https://arxiv.org/abs/2603.10551) | MMSys 2026 | Strict Gaussian progressive system | Joint base/enhancement layers with intermediate and final supervision | B/C: image experiments use counts; video reports a stream but is a different protocol |
| [SAD](https://arxiv.org/abs/2604.21984) | SIGGRAPH 2026 | Adjacent anisotropic ownership | Top-K soft diagram, independent radius/temperature, removal-delta pruning | B/D: 128-bit packed-site proxy plus header, explicitly no entropy coding |
| [AIR](https://arxiv.org/abs/2605.20820) | 2026 preprint | Strict Gaussian with amortized encoder | Stage-wise residual predictor, stage control, Predict-Optimize-Distill, adaptive quantization | B/C: 160--300 ms field prediction; no complete entropy stream and shared model excluded |
| [CGVQ](https://arxiv.org/abs/2607.05667) | SIGGRAPH poster 2026 | Strict Gaussian quantization | Cluster appearance/anisotropy, cluster-specific position coding, scale/rotation UQ, color RVQ | B: partial bits-back/index accounting; codebook/header/cold package not audited |
| [EA-GI](https://www.sciencedirect.com/science/article/pii/S0165168426000356) | Signal Processing 2026 | Strict Gaussian allocation | Quadtree partition into equal hybrid entropy regions and adaptive thresholding | C: “entropy” drives allocation; it is not an entropy codec |
| [LocoADC](https://arxiv.org/abs/2607.17896) | ACM MM 2026 | Strict Gaussian allocation | Region-wise coherent-distortion densification and similarity/color-consistent merging | C: matched-count representation improvement; no codec |
| [Gaussian Texture Compression](https://arxiv.org/abs/2607.27943) | 2026 preprint | Adjacent multi-map Gaussian | Shared geometry across SVBRDF maps/mips, residual hierarchy, pruning, fixed quantization | B/D: texture-stack bpp/channel; not a single RGB image or demonstrated entropy stream |
| [PA-G2DS](https://doi.org/10.1109/TCSVT.2026.3687408) | TCSVT 2026 | Adjacent hybrid splatting codec | Pixel-aligned generalized splats plus implicit coefficients; sequential and random-access decoding | D in this review: abstract/metadata verified; no exact table promoted without accessible primary full text |

This table distinguishes “produces a field” from “produces a portable compressed file.” The first
is common; the second remains rare.

## 5. Reconstruction quality and field efficiency

### 5.1 Foundational direct fitting

GaussianImage uses eight scalar parameters per RGB Gaussian: two means, three covariance degrees of
freedom, and three weighted-color values. On its V100/50,000-step Table 1 protocol it reports:

| Dataset | PSNR | MS-SSIM | Optimization | Render | Gaussians / parameters |
|---|---:|---:|---:|---:|---:|
| Kodak | 44.08 dB | 0.9985 | 106.59 s | 2,092.17 FPS | 70k / 560k |
| DIV2K x2 | 39.53 dB | 0.9975 | 120.76 s | 1,737.60 FPS | 70k / 560k |

At 30k Kodak Gaussians, its Cholesky and rotation-scale variants report 38.57/38.83 dB in
91.06/98.55 seconds. These values establish the basic quality-throughput trade-off, but the large
iteration count leaves substantial room for better initialization and topology schedules.

Image-GS changes both placement and rendering. On 45 roughly 2K evaluation images it reports:

| Raw FP16 parameter rate | PSNR | MS-SSIM | LPIPS |
|---:|---:|---:|---:|
| 0.122 bpp | 29.20 +/- 4.57 | 0.924 +/- 0.042 | 0.173 +/- 0.082 |
| 0.366 bpp | 32.99 +/- 4.49 | 0.966 +/- 0.020 | 0.083 +/- 0.057 |

The paper reports reaching 95% of final quality before 400 steps and 99% before 2,000, with typical
runs of 3,000--4,000 steps. On an A6000 at 2K resolution, 10k Gaussians for 1,000 steps take 18.74
seconds and render in 3.7 ms; 50k take 26.32 seconds and render in 4.5 ms. Its principal limitation
is natural-image noise and fine texture: the stylized-image benchmark is a favorable domain and its
natural CLIC results are weaker.

### 5.2 Quality at equal Gaussian count

Several methods improve the field without claiming complete compression:

- **LocoADC** is the clearest 2026 matched-count refinement mechanism. Its official-code
  reproduction at 120k steps reports GaussianImage gains from 32.58 to 35.60 dB on Kodak at 10k,
  34.64 to 37.66 dB on DIV2K at 30k, and 32.50 to 35.43 dB on CLIC at 30k. Applied to
  GaussianImage++, it reports smaller but positive gains (35.41 to 36.15 dB on Kodak and 38.30 to
  39.09 dB on DIV2K). Runtime increases, so this is a quality/count mechanism, not a speed winner.
- **Structure-Guided Allocation** reports 45.40 dB at 70k Kodak Gaussians versus GaussianImage's
  44.08 dB, and 41.29 versus 39.53 dB on DIV2K x2. At 30k Kodak Gaussians it reports 40.51 versus
  38.57 dB, but optimization rises from 91.06 to 131.94 seconds.
- **GaussianImage++** reports 35.41 dB at 10k Kodak Gaussians versus GaussianImage's 32.48 dB, and
  33.75 versus 31.45 dB at 50k DIV2K Gaussians. The improvement comes with longer fitting: 356.71
  versus 192.61 seconds in the 50k DIV2K row.
- **WIPES**, an adjacent carrier-bearing primitive, reports 45.87 dB on Kodak and 40.32 dB on
  DIV2K with Cholesky parameterization, versus GaussianImage's 44.06/39.53 dB in the same table,
  while retaining 1,778.75/1,830 FPS. This is evidence that high-frequency capacity, not only
  Gaussian allocation, controls the upper frontier.

Contour-Aware 2DGS supplies a different boundary control using ground-truth segmentation regions.
On DAVIS with 1,250 Gaussians, its full contour/warmup/no-clamp configuration reports
27.31 PSNR and 23.41 edge-focused PSNR versus 27.10/22.73 for the baseline. With 7,500 Gaussians,
the full configuration reports 33.44/29.18 versus 33.41/30.33: ordinary PSNR is neutral while the
edge metric is worse. The paper itself concludes that the mechanism is most useful when primitives
are scarce. Because the region masks are supplied and their storage is not counted, this is a
boundary-rasterization control rather than rate-distortion evidence.

The fair inference is not that one allocation method is globally superior. LocoADC and
Structure-Guided occupy direct regional allocation/merge controls; GaussianImage++ occupies
distortion-driven growth; WIPES tests whether more expressive atoms buy more than another topology
policy. A new method must hold renderer, count or complete bytes, optimizer work, and primitive
degrees of freedom fixed enough to isolate the mechanism.

### 5.3 Ultra-high-resolution quality

LIG demonstrates that additive Gaussian fields can fit extremely large images at high fidelity if
storage and fitting cost are allowed to grow:

| Dataset/regime | Gaussians | PSNR | Optimization | GPU memory | Render |
|---|---:|---:|---:|---:|---:|
| STimage 9K | 35m / 45m / 55m | 37.47 / 39.82 / 42.19 | up to about 4,778 s reported at 45m | 16.67 / 17.75 / 20.26 GB | 20.19 / 17.89 / 15.72 FPS |
| FGF2 4K | 10m / 12m / 14m | 51.81 / 53.90 / 56.05 | about 1,926 s at 14m | 4.21 / 4.26 / 4.39 GB | 74.38 / 63.62 / 58.09 FPS |
| DIV-HR 2K | 0.5m / 0.7m / 0.9m | 44.89 / 49.07 / 52.22 | paper protocol | about 1.01--1.05 GB | 542 / 491 / 442 FPS |

This is a fidelity/scale result, not a compression result. SGI attacks exactly this redundancy by
generating local Gaussian groups from coded seeds. SmartSplat attacks initialization and memory
failure on even larger images, but its “compression ratio” is a count budget derived from raw RGB,
not a produced file.

SmartSplat reports the following representative DIV8K average PSNR values under its analytical
raw-RGB/count ratios: 33.26 dB at CR=20, 29.65 at 50, 27.49 at 100, 25.75 at 200, 23.82 at 500,
and 22.66 at 1,000. On DIV16K it reports 34.34 dB at CR=50 down to 24.72 dB at CR=3,000. The
critical accounting detail is that it assumes seven bytes per Gaussian from another method and sets
the count as `3HW / (7*CR)` while optimizing float fields. These rows demonstrate robust UHR
sampling, not literal compression of the reported 53.56/235.52 MB source PNGs.

## 6. Convergence and image-to-field latency

### 6.1 Handcrafted and adaptive initialization

Image-GS shows that a cheap gradient/uniform mixture, sampled color, inverse scale initialization,
and error-guided growth can move most quality gain into the first hundreds of iterations. EA-GI and
Structure-Guided partition image structure before sampling; SmartSplat uses local gradient/color
variance plus exclusion; EllipssianNet learns center/covariance maps from synthetic Voronoi data.

EllipssianNet's author-reported final PSNR illustrates a modest but broad initialization gain:

| Dataset | EllipssianNet | Isotropic EGS | Color-guided | Random |
|---|---:|---:|---:|---:|
| Kodak | 27.54 | 27.17 | 25.81 | 25.46 |
| DIV2K | 25.84 | 25.62 | 24.16 | 23.60 |
| Image-GS dataset | 24.80 | 23.95 | 22.63 | 22.12 |

It predicts a patch in roughly 135--139 ms, but the downstream representation uses a 3DGS-style
14-parameter primitive and 5,000-step refinement. It is an initializer control, not a complete
field-conversion winner.

### 6.2 Domain-level amortization

EigenGS initializes a new image from a PCA basis learned over a domain, then refines the field. On
FFHQ with 20k Gaussians, GaussianImage reports 10.4/21.8/29.4/39.2/40.1 dB after
100/500/1k/5k/10k iterations. EigenGS with 300 components starts at 28.0 dB and reports
34.4/36.4/37.5/40.7/41.8 dB over the same checkpoints, with similar final runtime. The early gain is
large, but the eigenspace and its domain restriction are part of the system cost.

Instant-GI generalizes this idea with a learned predictor and adaptive count. On three DIV2K
examples at 2/10/20 seconds it reports 46.68/49.05/49.58, 37.41/41.51/42.49, and
36.79/39.72/40.51 dB; the paper's random baselines are lower at every checkpoint. Its 50k-iteration
upper-bound table reports 42.92 dB on Kodak with 342.86k parameters and 42.80 dB on DIV2K x2 with
615.05k, versus 41.44/40.26 dB for GaussianImage at matched counts. The roughly 3--4 GB network and
about 400 MB test memory must be accounted separately.

Fast-2DGS reports a 4.29 ms predictor, followed by a 5,000-step refinement of roughly 10 seconds. At
50k Kodak Gaussians its paper table reports 43.13 dB in 10 seconds, versus 39.36/39.78 dB for
GaussianImage Cholesky/rotation-scale in 13/14 seconds, 39.04 dB for Image-GS in 28 seconds, and
41.41 dB for Instant-GI in 10 seconds. This comparison is compelling for initialization, but the
29 MB predictor and about 19 hours of staged training are outside the per-image figures.

### 6.3 Fully feed-forward prediction

AIR removes per-image test-time fitting. Its author-reported Kodak and DIV2K rows are:

| Dataset | Predicted Gaussians | PSNR | MS-SSIM | Image-to-field latency |
|---|---:|---:|---:|---:|
| Kodak | 28k / 37k / 52k | 30.93 / 31.47 / 32.17 | 0.978 / 0.982 / 0.985 | 157 / 159 / 162 ms |
| DIV2K | 46k / 61k / 86k | 30.76 / 31.30 / 32.02 | 0.984 / 0.987 / 0.989 | 292 / 294 / 300 ms |

At its DIV2K adaptive-quantized setting AIR reports 30.23 dB, 0.979 MS-SSIM, LPIPS 0.214, 3.28
bpp, and 300 ms. GaussianImage and Image-GS rows in that table report 26.42 dB at 3.51 bpp/4,446 ms
and 30.17 dB at 6.04 bpp/3,091 ms. The latency conclusion is strong; the compression conclusion is
not, because the source describes adaptive uniform quantization and leaves an advanced entropy
model to future work.

### 6.4 What “fast convergence” should mean

The learned methods shift work rather than eliminate it. Their correct system comparison reports:

\[
T_{\mathrm{effective}}(M)=T_{\mathrm{train}}/M+T_{\mathrm{predict}}+T_{\mathrm{refine}}+
T_{\mathrm{encode}},
\]

for an expected deployment volume \(M\). It must also report model download/storage, domain shift,
and a training-free baseline. For one-off images, Image-GS-like initialization or a direct
structure-aware initializer can dominate total cost; at large scale, AIR-like amortization can be
the only path to subsecond conversion.

## 7. Rendering, decoding, memory, and query performance

### 7.1 Warm rendering

Representative author-reported warm-rendering results include:

| Method/protocol | Warm render result | Boundary |
|---|---:|---|
| GaussianImage, Kodak 70k | 2,092 FPS | V100, paper image protocol |
| GaussianImage, DIV2K x2 70k | 1,738 FPS | V100 |
| Image-GS, 2K 10k/50k | 3.7/4.5 ms (about 270/222 FPS) | A6000, top-K normalized field |
| LIG, 9K 35m--55m | 20.19--15.72 FPS | multi-million field |
| LIG, 4K 10m--14m | 74.38--58.09 FPS | multi-million field |
| SGI, 2K 50k | about 88 ms total reported inference, about 80 ms arithmetic-coding component | cold/package-oriented path, not comparable to warm raster only |
| SAD, cached 512/1024/2048 square | 0.015/0.034/0.132 ms | cached ownership; non-Gaussian |
| SAD, full refresh 512/1024/2048 square | 1.100/4.614/26.539 ms | ownership recomputation included |

These numbers are not one renderer benchmark. Gaussian count, resolution, active-set semantics,
hardware, cache state, and whether entropy decode is included differ. A production evaluation needs
the same cold/warm/query boundary for every method.

### 7.2 Batched fitting as a separate frontier

GaussianVision is primarily a systems and downstream-use paper. It reports up to 90.3x batched
fitting speedup over its baseline at batch 4,096 and about 97% GPU utilization. Its H100 profile for
4,000 Gaussians and 2,000 iterations rises from 7 seconds at batch 1 to 327 seconds at batch 4,096,
turning a large dataset into a throughput problem rather than thousands of sequential fits.
Structured initialization also improves its reported 3,000-step quality from 28.24 to 35.25 dB at
4,900 Gaussians.

This is transferable to offline corpora but not evidence for low single-image latency. It also
reports 3--23.52x compression relative to raw pixel tensors for downstream transport, not an actual
file format or ratio against PNG/JPEG originals.

### 7.3 Random access and progressive access

Image-GS has the cleanest direct random-access/LOD story: Gaussians can be spatially selected and its
growth order yields natural prefixes. P-GSVC strengthens progressive quality by jointly optimizing
base and enhancement layers. On Kodak at 5k/7k/9k Gaussian prefixes it reports 28.5/29.4/30.2 dB
versus 26.6/27.3/28.1 dB for sequential LIG layers; on DIV-HR at 50k/70k/90k it reports
30.2/31.3/32.1 versus 27.8/28.7/29.5 dB. The image tables do not establish complete stream bytes.

For a true embedded codec, every prefix must be independently decodable and the order/index cost
must be transmitted. For random access, report cold tile latency, bytes fetched, overfetch, cache
state, and quality against full decode. A globally arithmetic-coded stream may compress well but
destroy cheap spatial access unless it is packetized.

## 8. Compression and complete-byte evidence

### 8.1 GaussianImage codec lineage

GaussianImage quantizes positions in FP16, covariances to six bits, and color with two-stage residual
vector quantization; it also discusses optional partial bits-back coding. On DIV2K it reports:

| Operating point | Rate | PSNR | MS-SSIM | Encode throughput | Decode throughput |
|---|---:|---:|---:|---:|---:|
| Low | 0.3221 bpp | 25.6631 | 0.9154 | 0.00411 FPS | 1,970.76 FPS |
| High | 0.6417 bpp | 27.5656 | 0.9483 | 0.00473 FPS | 1,980.54 FPS |

The encode throughputs correspond to roughly 243 and 211 seconds per image. The paper is an
important codec starting point, but optional bits-back recovery and implementation-specific
serialization make complete cold-package verification necessary before treating those rates as
deployable bytes.

GaussianImage++ changes quantization to learned 12-bit position, 10-bit covariance, and 6-bit color
accounting and reports:

| Dataset | Rate range | PSNR range | MS-SSIM range | Encode time | Decode |
|---|---:|---:|---:|---:|---:|
| Kodak | 0.15--1.08 bpp | 25.3--31.1 | 0.834--0.961 | 338--347 s | 1,839--1,666 FPS |
| DIV2K | 0.25--0.92 bpp | 27.6--31.4 | 0.910--0.966 | 488--576 s | 440--748 FPS |

These rows improve on the paper's GaussianImage baseline but remain behind learned codec controls,
especially at high rate. The method does not describe a complete entropy-coded container sufficient
to promote the fixed-bit accounting to Level A.

Structure-Guided Allocation reports the strongest direct improvement over this lineage:

| DIV2K point | Rate | PSNR | MS-SSIM | Encode | Decode |
|---|---:|---:|---:|---:|---:|
| Structure-Guided low | 0.3033 bpp | 26.52 | 0.9126 | about 160 s | 1,540.87 FPS |
| GaussianImage low | 0.3221 bpp | 25.66 | 0.9154 | about 243 s | 1,970.76 FPS |
| Structure-Guided high | 0.6039 bpp | 28.74 | 0.9492 | about 145 s | 1,519.39 FPS |
| GaussianImage high | 0.6417 bpp | 27.57 | 0.9483 | about 211 s | 1,980.54 FPS |

It reports 43.44% Kodak and 29.91% DIV2K x2 BD-rate reduction against GSImage and only 0.0004 bpp
of structural metadata. The appropriate conclusion is “best reported RD improvement inside this
accounting lineage,” not “best complete image file.”

CGVQ further clusters appearance/anisotropy before coding and reports roughly 20% less bpp at
similar fidelity and a 1.68-dB gain over GaussianImage at the same 15k splats. Its K=1/4/8/16,
20k-splat table improves PSNR from 29.33 to 30.71/30.90/31.18 dB while encode throughput falls from
0.0291 to 0.00592/0.00358/0.00190 FPS and decode falls from 133.3 to 74.6/53.5/33.3 FPS. The
poster-sized source does not establish an audited header/codebook/index package, and its partial
bits-back ancestry retains the same implementation caveat.

### 8.2 SGI: the clearest complete high-resolution package

SGI's key contribution is not merely fewer Gaussians. Each seed generates a local group (the paper
uses ten) through lightweight MLPs; a multiscale schedule makes this structured representation
optimizable; then positions use geometry coding and seed attributes use a learned context model and
arithmetic coding. The transmitted package includes the generative state rather than pretending the
generated Gaussians are free.

Representative author-reported results are:

| Dataset | Method/regime | PSNR | SSIM | LPIPS | Fit | Package |
|---|---|---:|---:|---:|---:|---:|
| FGF2 | GaussianImage | 27.30 | 0.9457 | 0.1342 | 322.17 min | 23.37 MB |
| FGF2 | SGI low / high | 31.24 / 36.27 | 0.9863 / 0.9961 | 0.0731 / 0.0162 | 48.43 / 97.75 min | 16.33 / 41.74 MB |
| ICB | GaussianImage | 31.09 | 0.9330 | 0.1462 | 282.61 min | 23.37 MB |
| ICB | SGI low / high | 35.27 / 39.09 | 0.9853 / 0.9949 | 0.0575 / 0.0122 | 44.75 / 86.11 min | 12.30 / 32.15 MB |
| STimage | SGI low / high | 33.96 / 38.72 | 0.9743 / 0.9935 | 0.1196 / 0.0208 | 103.43 / 136.26 min | 10.05 / 22.03 MB |
| DIV2K | SGI low / medium / high | 28.69 / 37.72 / 44.03 | paper table | paper table | 3.31 / 6.80 / 15.64 min | 0.33 / 1.82 / 5.40 MB |

The high-rate SGI packages can exceed the fixed GaussianImage package, while delivering much higher
quality; thus “up to 7.5x” is not a universal per-row ratio. The important result is a broad
quality-size-time frontier on 27--76 MP imagery with an actual entropy path. The remaining drawbacks
are long fitting, more complicated cold decode, and possible tension between global context coding
and fine-grained random access.

### 8.3 Why the remaining “compression” figures are not file ratios

| Method | Reported storage language | Missing for original-file CR |
|---|---|---|
| Image-GS | FP16 bpp | entropy stream, headers/indexes; source-file denominator |
| AIR | adaptive quantization bpp | entropy coder, container, predictor cost; source-file denominator |
| SAD | packed 128 bits/site plus header | entropy coding and source-file denominator; also not Gaussian |
| SmartSplat | assumed seven bytes/G and raw-RGB CR | actual produced stream; PNG/JPEG denominator |
| GaussianVision | 3--23.52x versus raw pixels | portable image package and original-file denominator |
| LIG/LocoADC/EA-GI | count/float-field quality | all serialization and complete bytes |
| P-GSVC images | prefix Gaussian counts | image bitstream bytes and dependencies |
| GTC | fixed-quantized bpp/channel across a texture stack | entropy stream and ordinary single-image denominator |

Consequently, this review cannot truthfully name a winner for reduction against the original image
file. SGI supplies the strongest numerator evidence (`B_package`); the denominator needed for
`CR_file` is still absent from its reported tables.

### 8.4 Outside-class image codecs remain ahead on ordinary rate-distortion

GaussianImage's own DIV2K table makes the boundary concrete. Its low/high rows report
25.66/27.57 dB at 0.3221/0.6417 bpp. In the same table, JPEG2000 reports 27.28/30.93 dB at
0.2394/0.5993 bpp, Ballé17 reports 27.72/30.78 dB at 0.2271/0.4987 bpp, and Ballé18 reports
28.75/32.24 dB at 0.2533/0.5415 bpp. These baselines are not today's codec frontier, yet they
already dominate the GaussianImage points in PSNR/rate. GaussianImage++ improves its Gaussian
baseline but likewise remains behind learned-codec rows in its paper.

The appropriate current deployment control is therefore not only another splat representation. A
practical learned-codec study at [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Tatwawadi_What_Matters_in_Practical_Learned_Image_Compression_CVPR_2026_paper.html)
compares against modern standards and mobile decoding constraints. Its task and subjective protocol
are not directly pooled here, but it represents the production frontier that a claim of “best image
compression” would have to address. Gaussian fields may still win a joint objective that prices
warm rerendering, editability, sparse queries, and LOD; ordinary RD alone is not their demonstrated
advantage.

## 9. Adjacent methods that set the upper control

### 9.1 Soft Anisotropic Diagrams

SAD makes ownership explicit rather than summing Gaussian amplitudes. On the 45-image Image-GS
benchmark, it reports:

| Packed parameter rate | SAD PSNR | Image-GS PSNR |
|---:|---:|---:|
| 0.2 bpp | 33.87 | 31.32 |
| 0.3 bpp | 35.72 | 32.79 |
| 0.4 bpp | 36.97 | 33.80 |
| 0.5 bpp | 37.86 | 34.57 |

On Kodak at 50k sites/about 16 parameter bpp it reports 46.00 dB, SSIM 0.9871, LPIPS 0.0032,
and 2.2-second fitting, versus its re-evaluated Image-GS row at 36.90 dB/28 seconds. The paper is
explicit that rate is packed parameter storage without entropy coding. SAD therefore leads an
adjacent representation-quality/speed frontier, not the complete codec frontier. Its independent
radius, temperature, top-K ownership, and removal-delta pruning are mandatory causal controls for
claims about normalized Gaussian mixtures.

### 9.2 Wavelet-bearing splats

WIPES shows that localized oscillatory carriers can represent high-frequency detail more efficiently
than positive Gaussian kernels. The cost is extra frequency/phase state, more complex optimization,
and no demonstrated codec. It is the correct primitive control before concluding that a new
allocation or densification mechanism solves Gaussian blur/detail limitations.

### 9.3 Hybrid explicit/implicit codecs

PA-G2DS describes pixel-aligned generalized splats with implicit coefficients, millisecond-class
decoding, and both sequential and random-access modes. Because the accessible primary boundary in
this review was publication metadata/abstract rather than a verifiable full quantitative table, no
exact RD claim is imported. It nevertheless marks an important design direction: keep spatially
addressable explicit support while moving appearance capacity into a compact learned function.

GTC applies a related sharing idea to material texture stacks. At about 0.02/0.05/0.10/0.20 bpp per
channel it reports 36.42/39.44/41.19/42.21 dB versus Image-GS at
32.29/34.64/37.15/39.21 dB. Its high point reports 43.11 dB at 0.299 bpp/channel, compared with
Image-GS 40.77 at 0.400 and ASTC 41.10 at 0.407. The advantage depends on sharing geometry across
multiple correlated maps and mip levels; it is a donor for hierarchy/shared structure, not evidence
for an ordinary RGB-image file.

## 10. Pareto frontiers by operating regime

The table gives the most defensible answer to “which method is best?” without collapsing incompatible
protocols.

| Operating regime | Pareto-relevant method(s) | Why | Critical caveat |
|---|---|---|---|
| Simple additive field and very fast warm render | GaussianImage | Minimal eight-parameter splats; roughly 1.7--2.1k FPS in paper protocol | Slow per-image optimization; codec stream boundary is not fully portable |
| Editable/random-access strict Gaussian at low raw parameter rate | Image-GS | Top-K normalized mixture, adaptive growth, natural LOD, strong early convergence | FP16 payload, not entropy-coded rate; stylized-domain advantage |
| Direct GaussianImage-lineage low-rate RD | Structure-Guided Allocation | Largest reported BD-rate reduction versus GSImage; quality and encode-time gains | Level-B accounting, not independently byte-audited package |
| Complete high-resolution Gaussian storage | SGI | Actual structured/context-coded package; strong 27--76 MP size-quality-time evidence | Tens to hundreds of minutes fitting; more complex cold query |
| Highest strict-Gaussian fidelity when storage is loose | LIG, with LocoADC as allocation control | Multi-million-splat high-PSNR fitting; LocoADC improves equal count | Multi-GB memory and long optimization; not compression |
| Ultra-high-resolution training-free initialization | SmartSplat | Avoids failures and gives strong UHR fields with 1k-step option | “CR” is raw-RGB/count proxy, no stream |
| Subsecond image-to-Gaussian field | AIR | 160--300 ms fully feed-forward stage-wise prediction | Shared model and complete codec bytes omitted |
| Learned warm start plus refinement | Fast-2DGS / Instant-GI / EigenGS | High early quality when a trained domain prior is available | Offline training/model size/domain shift; not standalone codecs |
| Progressive Gaussian prefixes | P-GSVC; Image-GS | Joint base/enhancement supervision or natural growth order | Complete image prefix-stream bytes not reported |
| Batched corpus conversion/downstream use | GaussianVision | Up to 90.3x batched fitting speedup and Gaussian-native VLM substrate | Throughput rather than single-image latency; raw-pixel “compression” |
| Representation quality/speed, primitive unrestricted | SAD; WIPES | SAD dominates Image-GS packed-rate quality; WIPES improves frequency detail | Neither is a strict Gaussian complete codec |
| Correlated texture-map stacks | GTC | Geometry sharing across maps/mips | Not an ordinary single RGB image |

### 10.1 Multi-objective selection rule

No scalar ranking should be chosen without deployment weights. A practical candidate is
non-dominated only if no other candidate is at least as good in all required dimensions and better
in one:

\[
(D,\ B_{\mathrm{complete}},\ T_{\mathrm{encode}},\ T_{\mathrm{cold}},\ T_{\mathrm{warm}},
\ M_{\mathrm{peak}},\ Q_{\mathrm{access}}).
\]

Use distortion `D` rather than PSNR alone, retain perceptual and structural metrics separately, and
treat random-access/LOD correctness as constraints when required. A storage winner that takes hours
to encode and a latency winner that depends on a multi-gigabyte predictor can both be Pareto-optimal.

## 11. A fair experiment for original-file compression

### 11.1 Dataset manifest

For every image, preserve:

- SHA-256 and byte size of the exact supplied file;
- format, dimensions, bit depth, alpha, colorspace/ICC, orientation, and metadata policy;
- SHA-256 of the canonical decoded target pixels;
- canonical lossless PNG size; and
- whether the source is already lossy.

Evaluate Kodak, CLIC, and DIV2K or another standard set without reusing development images. Add an
ultra-high-resolution set only as a separate stratum; pixel count changes the Gaussian scaling
problem and cannot be averaged away.

### 11.2 Complete codec manifest

Each produced package must be cold-decoded in an empty temporary directory with network disabled.
The decoder receives only the declared shared runtime/model and the package. Record:

- exact package bytes and a per-section byte ledger;
- shared model/runtime bytes and version digest;
- decoded-pixel digest, PSNR, MS-SSIM, LPIPS, and alpha/color validity;
- encode, cold-decode, first-render, warm-render, tile-query, and LOD times;
- peak CPU/GPU memory and hardware/software identity; and
- failures, timeouts, and unsupported features.

For learned models, publish two rates: per-image stream bpp and system-amortized bpp at declared
corpus sizes. For methods with codebooks trained per image, the codebooks are per-image bytes. For
methods with segmentation masks or tile indexes, the masks/indexes are bytes. For bits-back, report
gross bytes, realized recovered bytes, required initial seed/message, and actual wall-clock decode.

### 11.3 Required outputs

The comparison should include:

1. complete-bpp RD curves and BD-rate against at least PNG/lossless, JPEG, WebP/AVIF or JPEG XL,
   and a practical learned-codec control appropriate to the quality range;
2. `CR_file` against both the exact supplied source and canonical PNG;
3. encode-time-to-quality and cold-decode-time-to-quality frontiers;
4. warm render/query curves versus resolution and primitive count;
5. count-to-bytes residuals showing why Gaussian count is not rate; and
6. a Pareto table with confidence intervals over images, not only pooled means.

A lossy source requires special care. If a JPEG is the supplied file, evaluate against the pixels
decoded from that JPEG and report whether the Gaussian package is smaller than the JPEG at the
measured reconstruction loss. Do not compare a lossy Gaussian rendition of a PNG against the byte
size of a separately quality-matched JPEG without stating the two different targets.

## 12. Mechanism-level synthesis

### 12.1 Allocation and topology

The literature consistently shows that uniform random placement wastes capacity. The strongest
families are:

- image-gradient/uniform mixtures and residual growth (Image-GS);
- structure/region partition followed by local budgets (Structure-Guided, EA-GI);
- local variance, exclusion, and robust sampled color (SmartSplat);
- distortion-driven densification and filtering (GaussianImage++);
- coherent-distortion births plus redundancy-releasing merges (LocoADC); and
- explicit removal-delta pruning (SAD).

The most promising composition is **proposal by cheap structure, decision by measured marginal
distortion per complete byte, and capacity recovery by safe merge/removal**. Structure should not
directly dictate the final topology when residual evidence disagrees.

### 12.2 Covariance and field semantics

Covariance parameterization changes stability and speed but is not the central quality bottleneck.
The larger choice is additive accumulation versus normalized/ownership behavior. This affects
constant reproduction, boundary bleed, opacity meaning, support truncation, gradients, and codec
state. The choice must precede initializer and codec optimization; otherwise a placement technique
can appear to win because it compensates for one renderer's failure mode.

### 12.3 Optimization schedule

Progressive coarse-to-fine fitting is common in LIG, SGI, P-GSVC, Image-GS, and AIR. The useful
unifying principle is not “use a pyramid”; it is:

1. fit the globally useful low-frequency/base component;
2. activate geometry and appearance degrees of freedom in stages;
3. allocate residual capacity only where the current field cannot explain the target;
4. periodically remove/merge redundant capacity; and
5. optimize the quantized or cold-decoded field before finalizing the stream.

Iteration counts should be replaced by measured work or wall-clock checkpoints because a topology
event, full raster pass, color solve, and predictor pass have very different cost.

### 12.4 Amortization

EigenGS, Instant-GI, Fast-2DGS, EllipssianNet, and AIR show a progression from domain bases to learned
samplers to full field prediction. The remaining opportunity is an **elastic target-rate predictor**:
one model should emit a nested field or stream across a byte ladder, rather than train separate
fixed-count models. Its rate controller must operate on the actual codec, not Gaussian count.

### 12.5 Codec design

The central compression problem is unstructured per-splat overhead. Promising coding axes are:

- delta-coded or geometry-coded positions after a stable spatial order;
- gauge-free or structure-relative covariance coordinates;
- shared/clustered color and shape codebooks;
- context models conditioned on decoded neighbors;
- structured seeds that generate several correlated Gaussians;
- embedded ordering for progressive and spatial packet access; and
- QAT/straight-through fitting against the exact decoder reconstruction.

SGI shows the benefit of structured seeds, but the direct codec must be established first. A neural
grammar can reduce rate while increasing cold-decode latency, package complexity, and loss of local
editability. The correct necessity test compares both at equal complete bytes and query semantics.

### 12.6 Systems

The main performance opportunities are fused forward/backward rasterization, exact tile culling,
batched fitting, sparse residual sampling with an unbiased objective, cached target statistics, and
separating field construction from warm query. GaussianVision shows that corpus throughput is a
different optimization target from one-image conversion. SAD shows the value of caching ownership.
Both suggest reporting the entire conversion and query pipeline rather than isolated kernel FPS.

## 13. Recommended architecture by deployment profile

### 13.1 Interactive editable field

Use an Image-GS-like normalized/top-K field or a controlled additive alternative, deterministic
structure-plus-gradient initialization, sampled/solved colors, residual allocate/merge, and a simple
direct codec with spatial packets. Optimize for subsecond-to-seconds conversion, cold tile query,
and edit stability. Avoid a heavy predictor unless many images share the deployment.

### 13.2 Maximum high-resolution storage efficiency

Use an SGI-like multiscale structured representation, but compare it against a strong direct
context-coded field first. Include the generative networks and context state; expose coarse seeds as
an independently decodable base layer; packetize spatial contexts if random access matters. Expect
offline fitting rather than interactive encode unless amortization is added.

### 13.3 Large-scale ingestion

Use AIR-like predict-optimize-distill with a single elastic model, GaussianVision-style batch
kernels, and a short quantization-aware correction pass. Publish offline training cost, model size,
domain robustness, and break-even volume. The product metric is images/hour at a complete-byte and
quality target, not predictor latency alone.

### 13.4 Maximum fidelity at loose rate

Use multiscale residual fitting with LocoADC-style allocation/merge and test a WIPES-like carrier or
SAD-like independent sharpness/ownership control. If richer atoms win after their extra parameters
are coded, the best representation is no longer a pure Gaussian field; that is a valid result.

## 14. Implications for StructSplat's task graph

This review does not authorize a default or scientific claim. It sharpens the existing evidence
gates:

| Transferable mechanism | Repository authority | Required comparison |
|---|---|---|
| Additive vs normalized/top-K/alpha semantics | CORE-013 -> BENCH-020 | Hold field and work fixed; include constant, boundary, texture, and downstream validity |
| Structure/gradient/region/UHR initialization | INIT-010 | Uniform/random, Image-GS gradient, direct SLIC/Sobel, and current deterministic priors |
| Stage-wise parameter activation | FIT-044 | Full-field activation control at equal work |
| Regional allocate/merge | FIT-045 | LocoADC-style direct control plus fixed-N/global/uniform arms |
| Appearance solve | FIT-046 | Conditional color solve at matched work and codec semantics |
| Sparse residual fitting | FIT-047 | Unbiased full-objective control and end-to-end timing |
| Coarse-to-fine topology order | FIT-048 | Full-N/single-scale control and exact work accounting |
| Loss/objective choice | FIT-049 | Isolate after field semantics are frozen |
| Convergence composition | BENCH-021 | Compose only predeclared component winners |
| Direct complete codec | COMP-013 | Exact stream bytes, cold decode/query, target-rate control, strict versioning |
| End-to-end acceleration | PORT-006 | Full conversion parity and representative timing, not kernel-only FPS |
| Byte-priced topology | FIT-030 | Select/stop using the chosen codec's realized bytes |
| Need for seed-generated structure | BENCH-025 | Direct codec versus SGI-like grammar at equal complete bytes and query cost |
| Conditional seed codec | COMP-014 | Implement only after a positive necessity verdict |
| Amortized predictor | FF-002 -> FF-003 -> BENCH-023 | Complete-byte elastic ladder, model/training break-even, held-out latency/quality |
| Richer primitive control | CORE-008 | WIPES/SAD-controlled test only after a residual failure justifies it |

The sequence matters: selecting a learned initializer before renderer semantics, or selecting a
topology rule before a complete codec, optimizes a proxy that may reverse under the final system.

## 15. Research claims occupied by prior art

The review rejects the following broad novelty framings:

- structure-aware Gaussian allocation, orientation, or precision;
- residual/progressive Gaussian densification;
- a normalized local Gaussian mixture as a category;
- learned instant Gaussian initialization or feed-forward image-to-Gaussian prediction;
- region/boundary-gated splatting using an external segmentation;
- clustered Gaussian attribute quantization; and
- seed-structured neural Gaussians for high-resolution compression.

A defensible contribution must instead establish a narrower relationship with actual evidence, for
example: a renderer-conditioned phase transition in which a deterministic tensor metric improves
complete-byte RD; a direct-versus-seed codec necessity result; an elastic amortized predictor whose
complete stream remains random-access; or an explicit causal result showing when Gaussian kernels
lose to ownership/frequency-bearing primitives.

## 16. Screened application branches not ranked

The search also found 2D Gaussian methods for super-resolution, low-light enhancement, dehazing,
editable layered images, and Gaussian-domain vision. They are not ranked above because their target
is not faithful reconstruction of the supplied image under a storage/latency budget. In particular,
[LL-GaussianImage](https://arxiv.org/abs/2601.15772) and
[Dehaze-GaussianImage](https://arxiv.org/abs/2606.16163) optimize enhanced or dehazed outputs;
[GaussianSR](https://ojs.aaai.org/index.php/AAAI/article/view/32369) predicts a super-resolved target;
[MiraGe](https://arxiv.org/abs/2410.01521) emphasizes semantic editability. Their losses and
reference pixels make their PSNR, field size, or runtime incommensurate with image-as-field codecs.
They may still be useful later for Gaussian-domain editing or task-aware transmission.

The review also excludes ordinary 3D/2D surface Gaussian splatting for novel-view synthesis. Those
methods contribute kernels, pruning, quantization, and renderer optimizations, but their fields are
multi-view scene representations with visibility and camera geometry, not 2D image fields. Such
mechanisms should enter only as controlled donors.

## 17. Limitations and confidence

- The field is moving quickly; 2026 preprints may change before archival publication.
- Several papers reuse names such as bpp, compression, encoding, and FPS for materially different
  quantities. This review follows the methods sections and table boundaries rather than the label.
- Exact numbers are author-reported and may use different code revisions from public repositories.
- PA-G2DS is included at abstract/metadata level only; no inaccessible quantitative table is
  reconstructed from secondary sources.
- The literature provides almost no paired original-file byte ledger, so the user's requested
  `CR_file` frontier remains an open benchmark outcome, not a reviewable published fact.
- This producer-authored synthesis requires independent scientific review before its Pareto and
  novelty conclusions are treated as accepted repository authority.

## 18. Bottom line

For a **portable high-resolution compressed Gaussian package**, start from SGI. For a **simple,
fast-rendering additive field**, start from GaussianImage. For an **editable, low-parameter,
random-access field**, start from Image-GS. For **subsecond amortized conversion**, start from AIR.
For the **strongest non-Gaussian representation control**, use SAD and WIPES. For **direct regional
allocation and low-rate GaussianImage-lineage controls**, use Structure-Guided Allocation and
LocoADC.

Do not select among them by Gaussian count or claimed compression ratio. Build complete streams,
cold-decode them, measure quality and four separate clocks, then compute both complete bpp and
`original_file_bytes / package_bytes`. Until that experiment exists, no method can honestly be
called the state of the art for compression ratio relative to the original image file.
