# StructSplat publication-readiness research and visual-evidence audit

**Research prompt:** `ara/prompts/publication-ready-research-and-figures-2026-07-14.md`

**Executed:** 2026-07-14

**Literature cutoff:** 2026-07-14

**Repository state inspected:** live working tree at execution time

**Assumed contribution bar:** a top graphics/vision conference paper; no target venue was given.

## Post-audit execution update

The audit's recommended substrate, direct controls, Stage-0 validation, Stage-0b calibration, and
Stage-1 killing pilot were subsequently implemented and executed on 2026-07-14. BENCH-007 completed
288/288 fits and 1,152/1,152 latest validated candidates, but the preregistered gate failed:
tensor-WSE gained at 0.5 bpp, tied the strongest gradient control at 1.0 bpp, achieved only
`-4.5417%` BD-rate versus the required `-10%`, cost `1.4752x`, and exceeded the texture guard.
Stage 2 was not authorized. The exact proposed claim below is therefore closed rather than merely
blocked; see `ara/evidence/bench007-stage1-killing-pilot-2026-07-14/run.md`. The remainder of this
document preserves the pre-execution research audit and rationale.

## Executive verdict

StructSplat is **not publication-ready**. The newly implemented figures make the method inspectable,
but they do not close the scientific blocker: no experiment currently establishes that the specific
tensor-metric/WSE mechanism improves held-out rate-distortion against the direct SLIC/Sobel prior at
equal self-contained bytes. BENCH-007 is preregistered but its target-rate harness, direct control,
Stage-1 killing pilot, and held-out Stage-2 evidence do not yet exist.

The narrowest defensible statement today is:

> StructSplat is an interpretable, training-free structure-tensor/WSE initialization and causal
> experimentation substrate for a normalized local Gaussian representation. Its existing evidence
> supports budget-dependent structured-placement effects inside its own high-rate harness, but does
> not establish actual-rate compression superiority, broad novelty for structure-aware allocation,
> or native superiority over current 2D image representations.

The potentially publishable claim, **only if BENCH-007 passes**, is narrower:

> At sparse self-contained rate, a unit-area structure-tensor metric plus weighted sample
> elimination improves rate-distortion over direct SLIC/Sobel, gradient, uniform-WSE, and random
> allocation under a common renderer/fitter/codec, and the gain is explained by edge-band coverage
> rather than extra search, capacity, or transmitted side information.

The strongest prior-art threat is
[Structure-Guided Allocation](https://arxiv.org/abs/2512.24018), which already couples SLIC/Sobel
structure to Gaussian allocation, orientation regularization, and adaptive covariance precision.
The strongest representation-level threat is
[Soft Anisotropic Diagrams](https://arxiv.org/abs/2604.21984), which replaces Gaussian overlap with
explicit top-K anisotropic ownership. The outside-class compression bar is much higher than the
within-Gaussian frontier; a recent practical learned codec evaluates subjective quality, modern
conventional codecs, on-device latency, and large images
([CVPR 2026 primary source](https://openaccess.thecvf.com/content/CVPR2026/html/Tatwawadi_What_Matters_in_Practical_Learned_Image_Compression_CVPR_2026_paper.html)).

## 1. How the research was performed

The saved prompt was applied to the live code, tasks, ARA record, benchmark outputs, and primary
sources. The search covered exact method names, functional descriptions, mathematical mechanisms,
official proceedings/project pages, and official repositories where located. Recipient-field
search was supplemented with anisotropic sampling, structure-tensor visualization, adaptive
approximation, minimum-description-length, predictive coding, phase-diagram, and experimental-design
mechanisms.

Primary sources inspected included:

- [GaussianImage (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01421.pdf)
  and its [official code](https://github.com/Xinjie-Q/GaussianImage);
- [Image-GS](https://arxiv.org/abs/2407.01866);
- [GaussianImage++ (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/37572);
- [Structure-Guided Allocation](https://arxiv.org/abs/2512.24018);
- [Soft Anisotropic Diagrams](https://arxiv.org/abs/2604.21984), its
  [project page](https://luckyiyi.github.io/SAD/index.html), and
  [official code](https://github.com/LuckyIYI/SAD);
- [SGI (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_SGI_Structured_2D_Gaussians_for_Efficient_and_Compact_Large_Image_CVPR_2026_paper.pdf);
- [P-GSVC](https://arxiv.org/abs/2603.10551);
- [CGVQ](https://arxiv.org/abs/2607.05667), released shortly before this cutoff;
- [Contour Information Aware 2DGS](https://arxiv.org/abs/2512.23255);
- [WIPES (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_WIPES_Wavelet-based_Visual_Primitives_ICCV_2025_paper.html);
- [Instant GaussianImage (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Instant_GaussianImage_A_Generalizable_and_Self-Adaptive_Image_Representation_via_2D_ICCV_2025_paper.html);
- [Anisotropic Blue Noise Sampling](https://doi.org/10.1145/1882261.1866189); and
- Yuksel's [Weighted Sample Elimination paper/project page](https://www.cemyuksel.com/research/sampleelimination/).

The July 13 repository audits were treated as leads, not authority. Every live-code or task-status
claim below was rechecked. Cross-paper numbers remain contextual because datasets, rate definitions,
renderers, horizons, and hardware differ.

## 2. Live frontier map

| Layer | Primitive / mechanism in the repository | Evidence now | Unresolved failure |
|---|---|---|---|
| Signal analysis | Two-scale structure tensor; energy, coherence, flat/edge/corner labels, gradient/tangent directions | Unit tests and numerical synthetic checks | Scale sensitivity and luma/RGB choice have no paper-grade sensitivity map |
| Allocation | Density PMF plus exact-N WSE; optional unit-area anisotropic metric; quadtree-WSE shipped default | ABL-006 supports a budget-dependent local PSNR effect and rejects flanking | No actual-rate comparison to SLIC/Sobel, gradient, uniform-WSE, and random |
| Primitive | RS Gaussian with tangent-aligned `sx`, across-edge `sy`, sampled color | Correctness tests and new anatomy figures | Closest work already uses anisotropic Gaussians; irreducible value is unproved |
| Renderer | Sorting-free normalized weighted sum with clipped rectangular support; exact CUDA oracle parity | Forward/backward tests and native-reference separation | Ownership/overlap mechanism was not previously visible; external formulations differ |
| Optimization | Adam family, structured densification, optional controls, same-count checkpoint policy | Many positive and negative local studies | Large search history makes a clean paper narrative and untouched confirmation essential |
| Codec | Self-contained SSPL1 stream with actual byte measurement and cold decode | Codec tests and high-rate storage diagnostics | No target-rate RDO, equal candidate search, robust BD-rate, or held-out phase diagram |
| Measurement | PSNR, MS-SSIM, LPIPS, convergence, timing, provenance-aware native lanes | Strong local audit discipline | Main claim lacks edge-band/texture-band/bleed diagnostics and image-cluster inference |
| Communication | README/docs/ARA record; new deterministic tensor/field/ownership panels | Explanatory bundle implemented by DOCS-002 | No paper manuscript, bibliography, algorithm box, empirical result figures, or supplement |

### Repository/profile conflicts found

- `ara/PAPER.md` is an ARA manifest, not a manuscript. No `.tex`, `.bib`, notebook, or durable
  paper source was found.
- `benchmarks/rate_distortion.py` sweeps budgets and bit mixes, but does not implement BENCH-007's
  byte-target RDO, equal candidate ladder, missing-point semantics, monotone envelopes, BD-rate,
  cluster bootstrap, direct structural controls, or held-out manifests.
- No SLIC allocator or direct Structure-Guided Allocation-style common-renderer control was found.
- Existing plots visualize benchmark scores/reconstructions, but there was no code for structure
  tensors, tensor metrics, Gaussian ellipses, or normalized ownership before DOCS-002.

## 3. Domain-neutral functional signature

A dense bounded vector signal is approximated by a finite unordered set of local anisotropic
kernels. A pre-analysis maps local differential evidence into where kernels are permitted to spend
capacity, how their exclusion neighborhoods deform, and how their principal axes begin. Kernel
colors and geometry are then optimized through a local-support forward operator whose normalized
denominator couples every overlapping kernel. The encoder must divide a finite transmitted-byte,
search-time, and rendering budget between discrete membership/count choices and continuous
attributes. The decoder observes only the stream, not the source-derived tensor. The main failure
modes are missed sparse coverage, cross-interface mixing, texture starvation, overlap
ill-conditioning, unstable discrete site choices, hidden side information, and overcomplete regimes
that erase the initialization effect.

Local variables are position, covariance, color, opacity, and responsibility; global coupling
comes from normalization, total count/bytes, codec ranges/codebooks, and shared optimization.
Rotation is modulo pi and becomes unidentified for isotropic kernels. Boundary clipping breaks
translation symmetry. CUDA atomics make independent trajectories nondeterministic. Actual byte
rate, not count or float payload, is the resource that must be conserved in a compression claim.

## 4. Fixation anti-library

The following must not be presented as transformational or broadly novel:

- structure-aware Gaussian allocation or orientation;
- gradient-weighted initialization, residual densification, or dynamic count;
- generic learned initialization or an image-conditioned sampler;
- generic QAT, scalar/VQ quantization, clustered codebooks, or adaptive precision;
- progressive Gaussian layers or a WSE prefix by itself;
- segmentation-gated rendering or a generic boundary loss;
- a Gaussian-plus-wavelet hybrid without WIPES as a direct primitive control;
- another loss, scale cap, optimizer, low-pass schedule, or checkpoint heuristic;
- “interpretable,” “training-free,” or “blue noise” as a substitute for actual held-out value.

## 5. Adversarial prior-art audit

| Prior work | Problem overlap | Representation overlap | Mechanism overlap | Prediction/evidence overlap | Threat to StructSplat |
|---|---|---|---|---|---|
| [GaussianImage](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01421.pdf) | Single-image representation and codec | Per-image 2D colored Gaussians | Accumulated rasterization, QAT/VQ | Kodak/DIV2K representation and compression | Owns the base problem and primitive family |
| [Image-GS](https://arxiv.org/abs/2407.01866) | Content-adaptive explicit image representation | Anisotropic colored Gaussians and normalized local mixture | Gradient allocation, residual growth, top-K locality, LOD | Visual point distributions, error maps, rate/quality and speed | Occupies adaptive allocation, normalized mixing, and progressive growth |
| [Structure-Guided Allocation](https://arxiv.org/abs/2512.24018) | Low-rate structural allocation | 2D Gaussians | SLIC/Sobel structural classes, orientation regularization, adaptive covariance bits | Initial/optimized distributions, RD curves, BD-rate, spatial bit maps | Closest direct threat; broad structure claim is unavailable |
| [GaussianImage++](https://ojs.aaai.org/index.php/AAAI/article/view/37572) | Compact Gaussian representation/compression | Same base family | Distortion growth, context covariance filters, separated learned quantizers | RD/quality/runtime evidence | Occupies obvious densification/filter/quantizer combinations |
| [SAD](https://arxiv.org/abs/2604.21984) | Differentiable local image representation | Anisotropic sites with explicit partition of unity | Per-site reach/temperature, top-K ownership, removal-delta pruning | Strong fitting/speed/quality claim plus released multi-backend code | Strongest representation/formulation threat; motivates ownership diagnostics |
| [SGI](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_SGI_Structured_2D_Gaussians_for_Efficient_and_Compact_Large_Image_CVPR_2026_paper.pdf) | Compact high-resolution representation | Seed-organized neural Gaussians | Multiscale regions, context entropy, coarse-to-fine fit | Megapixel scaling and compression/speed evidence | Occupies broad “structured Gaussians” and high-resolution story |
| [P-GSVC](https://arxiv.org/abs/2603.10551) | Scalable image/video Gaussian coding | Base plus enhancement Gaussian layers | Joint multilevel optimization | Prefix quality/resolution and training-stability figures | Occupies progressive/layered coding claim |
| [CGVQ](https://arxiv.org/abs/2607.05667) | Gaussian parameter compression | Clustered Gaussian attribute groups | Appearance/anisotropy-conditioned codebooks | Kodak RD and runtime tradeoff | Occupies generic structure-like clustered VQ |
| [Contour-aware 2DGS](https://arxiv.org/abs/2512.23255) | Sparse boundary preservation | Region-gated Gaussians | Segmentation-constrained rasterization and edge-band metric | Synthetic/real zooms, edge-focused PSNR | Direct threat to broad boundary gating; external masks are its weakness |
| [WIPES](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_WIPES_Wavelet-based_Visual_Primitives_ICCV_2025_paper.html) | Explicit local image fitting | Frequency-bearing wavelet primitives | Local frequency coefficients rather than Gaussian-only color | Quality/speed and visual comparisons | A Gaussian-only paper cannot claim the best primitive grammar |
| [Instant-GI](https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Instant_GaussianImage_A_Generalizable_and_Self-Adaptive_Image_Representation_via_2D_ICCV_2025_paper.html) | Fast adaptive Gaussian generation | Predicted Gaussian fields | Learned probability map, Floyd–Steinberg placement, Delaunay/ellipse features | Kodak/DIV2K and training-time evidence | Occupies learned adaptive initialization; training-free remains a distinction, not value |

### Irreducible remainder

After subtracting known structure-aware allocation, Gaussian orientation, adaptive quantization,
progressive growth, and generic blue-noise sampling, the candidate remainder is:

> A source-derived, unit-area tensor metric deforms the *repulsion geometry* of exact-N WSE while
> the same tensor separately controls density and tangent initialization; the scientific question is
> whether that particular repulsion relationship provides actual-rate, edge-localized value beyond
> direct structure allocation under a shared normalized renderer.

This is best classified as **known components, possibly new relationship (N2/N2-T)** until the
direct-control result exists. Apparent novelty confidence is **25–45%** under this search. The range
is low because anisotropic blue noise is old, structure-aware Gaussian allocation is current, and a
recipient-field paper may use the same relation under different terminology. I did not find the
exact structure-tensor-to-WSE metric relationship in the searched 2D image-representation sources,
but this is not an absolute novelty claim.

### Transformation tests

- **A+B test:** “structure tensor + WSE + Gaussians” is plainly A+B+C; therefore the broad method is
  N1/N2, not N3.
- **Subtraction test:** the only surviving delta is tensor control of the sample-exclusion metric
  and its testable relation to sparse coverage.
- **Grammar test:** the current method does not change the representation grammar; it selects a
  point set inside the existing Gaussian grammar.
- **Prediction test:** it predicts a sparse-rate edge-band gain coupled to better coverage/bleed,
  disappearing in an overcomplete regime.
- **Necessity test:** anisotropic repulsion is necessary only if equal-density isotropic WSE and
  direct structure allocation fail in the predicted strata. This is unproved.
- **Compression test:** the idea compresses to a precise mechanism, but not to a new primitive.

## 6. Claim-evidence matrix

| Claim or requirement | Status | Publication impact | What closes it |
|---|---|---|---|
| Tensor, density, WSE, RS, and normalized renderer are implemented as documented | **Supported now** | Foundation only | Existing tests plus code references |
| Tensor/WSE anatomy is reproducibly visible on a real image | **Supported now** | Communication gap closed | DOCS-002 bundle and tests |
| Structured placement matters at some local count/horizon slices | **Supported now, bounded** | Context, not main compression claim | ABL-006 with its stated high-rate scope |
| Flanking is the preferred edge mechanism | **Refuted** | Must appear as negative result/ablation | Preserve ABL-006 result; do not resurrect claim |
| SSPL1 can produce and cold-decode a self-contained stream | **Supported now** | Necessary but insufficient | Existing codec tests; BENCH-007 must use it |
| Tensor-metric WSE beats SLIC/Sobel at equal actual bpp | **Planned/preregistered** | **Submission blocker for proposed paper** | BENCH-007 Stage 1 then untouched Stage 2 |
| Direct SLIC/Sobel common-renderer control exists | **Absent** | **Submission blocker** | Implement and test `local_slic_sobel_control` without paper-name laundering |
| Target-rate RDO, monotone envelope, and robust BD-rate exist | **Absent** | **Submission blocker** | BENCH-007 harness and edge-case tests |
| Held-out DIV2K validation evidence exists | **Absent** | **Submission blocker** | Frozen hashes and Stage-2 run after passing the gate |
| Edge-band, texture-band, signed bleed, and contributor mechanism support the result | **Partly implemented diagnostics; result absent** | Blocks causal interpretation | BENCH-007 metric definitions and preregistered plots |
| Native external validity is adequate | **Limited** | Blocks broad competitiveness claim | Full-resolution/native-rate lanes with official code and central decoded-pixel scoring |
| Overall compression SOTA | **Unsupported** | Claim forbidden | Broader codec frontier, subjective/perceptual protocol, standard rates, scaling; likely not the right claim |
| Progressive WSE is an embedded codec | **Unsupported** | Claim forbidden | Stream order preserving useful prefixes with actual prefix bytes/RD |
| Runtime/memory scaling is paper-grade | **Partial/local** | Blocks strong systems claim | Native-resolution curves, encoder search cost, decode latency/FPS, peak RAM/VRAM |
| Statistical inference is paper-grade for the main claim | **Planned** | **Submission blocker** | Image-cluster bootstrap, preregistered multiplicity handling, raw per-image curves |
| Reproducible artifact has one-command paper reproduction | **Absent** | Strong artifact/review risk | Frozen environment, data manifests, figure Makefile/script, archived release and checksums |
| Complete manuscript exists | **Absent** | **Submission blocker** | Paper source, bibliography, equations, algorithm, experiments, limitations, data/license statement |

## 7. Visual-evidence audit

Nearest papers repeatedly use different figure roles rather than one omnibus result: Image-GS shows
point distributions plus error maps and an optimization sequence; Structure-Guided Allocation
separates representation and codec pipelines, initial distributions, budget curves, optimized
distributions, RD curves, ablations, and spatial bit allocation; Contour-aware 2DGS combines
synthetic boundary cases with edge-focused metrics; Instant-GI visualizes its learned density,
dithering, Delaunay, and ellipse-feature pipeline. StructSplat previously had numerical plots and
reconstruction sheets but lacked a visual explanation of its distinct tensor/WSE relationship.

The plan below does not copy any paper's artistic layout. It records the scientific role that each
figure must serve.

| Figure | Claim served | Exact data/computation | Visual encoding and comparability constraints | Status |
|---|---|---|---|---|
| F1 Method overview | Defines the pipeline and separates source-only encoder analysis from transmitted decoder state | Input → tensor → density/metric → WSE → RS field → normalized render → fit → SSPL1 | Diagram marks source-only analysis, decoder-complete bytes, cold decode, and the renderer equation | **Implemented by DOCS-002** as deterministic SVG |
| F2 Tensor anatomy | Shows the actual signal driving all three initialization jobs | Raw energy, coherence, labels, tangent field, density PMF from production code | Same crop; fixed label colors; robust display transform disclosed; raw arrays saved | **Implemented by DOCS-002** |
| F3 Sampling/primitive anatomy | Shows irreducible tensor-metric WSE mechanism and distinguishes it from Gaussian support | Selected sites, fixed-display unit-area metric shapes, actual initialized RS ellipses | Explicit legend: metric ellipse is not Gaussian ellipse; `(x,y)` and tangent convention stated | **Implemented by DOCS-002** |
| F4 Renderer anatomy | Explains normalized overlap and candidate mechanism metrics | Initial reconstruction, denominator, effective count, entropy, dominant owner, raw maps | Exact production support math; common scales per compared panel; initialization-only banner | **Implemented by DOCS-002** |
| F5 Causal allocation comparison | Establishes what tensor metric adds beyond density/orientation/direct structure | Factorial arms at identical count, actual rate, fit budget, start policy | Same image/crop/scale; sites and reconstructed zooms; no named-paper label for local transplants | **Blocked on BENCH-007/control** |
| F6 Actual-rate phase diagram | Main empirical claim | Raw per-image RD points, monotone envelopes, BD-rate over overlap, component bytes | Actual stream bpp only on x-axis; intervals clustered by image; missing targets visible | **Blocked on BENCH-007** |
| F7 Mechanism figure | Tests sparse edge coverage rather than aggregate-score storytelling | Edge/texture MSE, signed cross-edge bleed, effective contributors, entropy versus rate | Predeclared bands; same thresholds across arms; raw distributions plus paired deltas | Diagnostics partly implemented; **result missing** |
| F8 Optimization/resources | Bounds systems value and cost | PSNR/MS-SSIM/LPIPS vs iteration/time; init/search/fit/encode/decode; RAM/VRAM/FPS vs pixels/N | Equal horizons and hardware; uncertainty across images; search cost charged | Existing fragments; **paper-grade run missing** |
| F9 Representative success/median/failure | Prevents cherry-picked qualitative evidence | Cases chosen by preregistered paired-delta quantiles after aggregate decision | Target, reconstructions, identical amplified error scale and zooms; rate printed | **Result missing** |
| F10 Supplement/audit | Enables falsification | Tensor scale/operator sensitivity, seed sensitivity, cold parity, stream components, per-image curves, negative arms | Raw tables, hashes, failure rows, configuration fingerprints | Infrastructure partial; **BENCH-007 artifacts missing** |

### Implemented DOCS-002 bundle

The new generator writes a vector encoder/decoder overview and fourteen lossless panels:

1. processed input;
2. robust-log tensor energy;
3. coherence;
4. flat/edge/corner labels;
5. cyan tangents and orange normals;
6. robust-log density PMF;
7. sites with unit-area tensor-metric shapes;
8. actual initialized one-sigma RS Gaussian ellipses;
9. initial normalized reconstruction;
10. initial absolute RGB error;
11. normalized-renderer denominator;
12. effective contributor count;
13. responsibility entropy; and
14. dominant Gaussian owner.

The SVG distinguishes source-only tensor/density/WSE analysis from transmitted Gaussian state and
cold decode. The bundle also writes raw NPZ arrays, resolved JSON configuration, source,
implementation, and output hashes, repository and environment provenance, diagnostic identity
checks, and a montage that says the outputs are not optimized, held-out, or comparative evidence.

## 8. Independent research lanes

These lanes update rather than replace `research-portfolio-2026-07-13.md`. Candidates remain
separate until the actual-rate evidence selects a regime.

### Productive recombinations

**P1 — Actual-rate tensor/WSE phase diagram (N2/N4 program).** Hold renderer, fitter, codec, and
search fixed; factor density, orientation, and exclusion metric. Prediction: only the tensor metric
arm improves sparse edge strata. Kill if it loses to SLIC/Sobel or the gain vanishes after coding.

**P2 — Two-part layout/attribute codelength audit (N2-T).** Measure `R(layout) +
R(attributes|layout)` instead of assuming blue-noise regularity is free. Prediction: if WSE matters
for compression, Morton/layout deltas or conditional attributes must get cheaper at matched
distortion. Kill if random/gradient layouts code equally well after sorting.

**P3 — Common-renderer/ native-renderer bridge (N2/N4).** Cross initial field and renderer while
keeping native-authentic results separate. Prediction: some apparent method ranking is an
allocation-by-objective interaction. Kill the interaction claim if rankings are invariant.

**P4 — Fixed-byte removal/birth exchange (N2-T).** Adapt exact counterfactual removal scores from
ownership formulations, then spend released stream bits on a new site. This is a control, not broad
novelty. Kill if removal rank does not predict post-recovery distortion.

**P5 — Progressive-order codec preservation (N2).** Compare WSE prefix geometry before and after
codec ordering, with actual prefix streams. Kill if Morton/entropy ordering erases the prefix value
or joint layer optimization is required as in P-GSVC.

### Assumption surgery / exploratory candidates

**E1 — Treat tensor scale as uncertain, not fixed.** Sweep derivative/integration scales and infer
a stability region rather than report one hand setting. Prediction: a genuine mechanism persists
over a nontrivial scale interval; otherwise it is parameter selection.

**E2 — Replace pointwise “edge” with responsibility flux.** Measure normalized mass crossing a
tensor-normal ridge without using an external segmentation mask. Prediction: flux, not energy
alone, localizes sparse cross-edge bleed. Kill if it reduces to the existing Sobel loss.

**E3 — Infer useful structure from decoder-visible state.** The source tensor is not decoder
visible; ask which geometry can be regenerated from a decoded base layer. This is the bridge to T1,
not permission to hide tensor maps as free side information.

**E4 — Treat failure localization as a primary output.** Stratify by edge distance, texture,
coherence, contributor count, and entropy. Prediction: allocation arms occupy distinct strata; kill
global claims if no stable stratum exists.

### Primitive/grammar-changing candidates

**T1 — Decoder-synchronized structural geometry (N3-T, provisional confidence 30–45%).** A decoded
base layer deterministically regenerates tensor/WSE default geometry; the enhancement stream sends
colors and corrections. The primitive becomes conditional on shared decoded state. Prediction:
layout/orientation bytes fall at <=1 bpp. Kill if base quantization changes site identity enough that
corrections erase the savings. Threats: predictive coding, SGI seeds, P-GSVC layers, and learned
latent decoders.

**T2 — Adjacency-coded ownership complex (N3-T, 20–35%).** Replace independent overlapping blobs by
sites plus shared interfaces carrying transition parameters. Prediction: boundary cost scales with
interface complexity, not Gaussian count. Kill if topology/interface bytes or fitting instability
dominate. SAD, vector graphics, meshes, and contour-gated methods are severe threats.

**T3 — Erasure-robust splat descriptions (N3-T, 20–35%).** Optimize groups so arbitrary packet
subsets retain coverage, changing success from a prefix to subset robustness. Prediction: 20% random
loss degrades more gracefully for <=5% clean-rate penalty. Kill if ordinary base/enhancement coding
dominates at realistic loss.

## 9. Cross-domain transfer audit

| Candidate | Donor mechanism | Preserved structure | Broken correspondence / required invention | Recipient-specific prediction |
|---|---|---|---|---|
| P1 | Statistical-physics phase diagrams | A control parameter changes interaction/coverage regime | Finite heterogeneous images have no thermodynamic limit; call it a regime map unless a stable transition is measured | Tensor delta changes slope/sign at a reproducible contributors-per-pixel range |
| P2 | Minimum description length | Model structure is charged by transmitted explanation length | Entropy models and tables can hide cost; use the actual decoder-complete stream | WSE helps only if layout/conditional attribute bytes fall at matched distortion |
| T1 | Predictive/scalable coding | Encoder and decoder share a previously decoded state | WSE site identity is discontinuous under base noise; canonical arithmetic/corrections are required | Position/orientation bytes drop materially without equal distortion loss |
| E4 | Mixture-model responsibility diagnostics | Normalized weights define soft membership, entropy, and effective count | Pixels are spatially correlated and kernels are truncated; use image-level inference | Method gains localize to low-count/high-entropy or boundary-flux strata |
| E1 | Metrology/uncertainty analysis | A measured mechanism must survive instrument/scale choice | Tensor scales are algorithm parameters rather than physical instruments | Claimed effect has a stable scale plateau and calibrated orientation uncertainty |
| BENCH-007 protocol | Randomized trials / optimal experimental design | Freeze interventions, endpoints, and subgroups before outcomes | Seeds and rate points are repeated within images; cluster at image level | The predeclared paired effect replicates without development-set tuning |
| T2 | Planar meshes / defect and interface theory | Shared relations carry discontinuity information | Natural image topology changes with scale and texture; topology must be encoded | Boundary cost follows interface length/complexity rather than site count |

Terminology-removal and causal-preservation tests leave P2, T1, E4, and the experimental-design
transfer precise without donor vocabulary. The phase-diagram analogy is useful only as a measurement
program; claiming a literal phase transition would fail the counter-analogy test. T2 and T3 retain
high adoption barriers and are not authorized before BENCH-007.

## 10. New-evidence programs

### D1 — Frozen sparse-rate mechanism map

- **Varied:** allocation arm, actual target bpp {0.5, 1.0} in Stage 1, then the full frozen rate
  set only after passing.
- **Measured:** actual SSPL1 bytes, PSNR/MS-SSIM/LPIPS, edge/texture errors, signed bleed,
  denominator, effective count, entropy, stream components, search time, and failures.
- **Conventional expectation:** direct SLIC/Sobel or gradient allocation captures the useful
  structure; tensor-metric repulsion adds no terminal coded value.
- **Surprising signature:** tensor metric wins at both sparse rates and the gain co-occurs with the
  predicted edge/coverage movement, not more candidates or time.
- **Promotion rule:** the exact BENCH-007 gate.
- **Controls:** common candidate ladder, independent fits, cold-decode parity, source hashes,
  image-cluster inference, failure visibility, and no endpoint extrapolation.
- **Productive failure:** a regime/failure atlas that closes the narrow compression claim.

### D2 — Renderer ownership atlas

- **Varied:** fixed sites/attributes through normalized Gaussian overlap and carefully matched
  ownership controls; start with synthetic steps, junctions, thin lines, and texture.
- **Measured:** denominator, active/effective contributors, entropy, dominant-owner churn,
  cross-edge flux, gradient conditioning, and RD.
- **Conventional expectation:** sharper ownership, not WSE geometry, explains sparse boundary gains.
- **Surprising signature:** tensor metric improves coverage at matched ownership statistics, or a
  specific overlap statistic predicts every gain/failure.
- **Controls:** identical fields where equations permit, parameter-bit accounting, numerical parity,
  and synthetic ground truth.
- **Productive failure:** identifies whether the normalized Gaussian grammar, rather than
  initialization, is the limit.

## 11. Pareto shortlist

Scores are 0–5 and are intentionally not collapsed into one rank.

| Candidate | Apparent novelty | Falsifiability | Importance | Feasibility | Cheap first test | Informative failure | Publication potential |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 actual-rate phase diagram | 2 | 5 | 5 | 4 | 4 | 5 | 5 if positive; 3 if negative benchmark paper |
| P2 two-part codelength | 2 | 5 | 4 | 4 | 4 | 5 | 4 |
| P3 common/native bridge | 2 | 5 | 4 | 3 | 3 | 5 | 4 |
| E4 ownership/failure atlas | 2 | 5 | 4 | 4 | 5 | 5 | 4 |
| T1 decoder-synchronized geometry | 4 | 5 | 5 | 2 | 3 | 5 | 5 |
| T2 ownership complex | 3 | 5 | 5 | 2 | 2 | 4 | 5 |
| T3 erasure-robust descriptions | 3 | 4 | 3 | 2 | 3 | 4 | 3 |

The immediate Pareto choice is P1: it has the highest information value and determines whether T1
is worth attempting. E4 is the best low-cost measurement addition and is partly enabled by the new
renderer diagnostics. T1 remains the highest-upside formulation change, but implementing it now
would skip the evidence gate.

## 12. Recommended first and decisive experiment

Run BENCH-007's frozen eight-image Stage-1 actual-rate killing pilot at 0.5 and 1.0 bpp, comparing
tensor/on-edge WSE to the strongest direct SLIC/Sobel nonlearned control, with the other allocation
arms diagnostic.

- **Null:** tensor-metric WSE has no positive rate-distortion advantage over the direct control.
- **Positive signature:** the preregistered PSNR/BD-rate gate plus edge/coverage movement, with no
  >10% fit-plus-search regression.
- **Strongest conventional explanation:** density or orientation alone, unequal RDO search, or
  normalized-renderer overlap explains the apparent gain.
- **Minimal implementation:** target-rate candidate planning, the SLIC/Sobel local control, central
  cold-decode scoring, raw envelopes/BD-rate, and the new diagnostic maps on selected cells.
- **Abandonment:** apply BENCH-007's rule without post-hoc rescue. If it fails, close the exact
  tensor/WSE compression claim and keep the harness/negative mechanism map.

## 13. Priority publication queue

1. Implement BENCH-007 substrate and direct SLIC/Sobel control with focused tests.
2. Freeze manifests/hashes and run the Stage-1 killing pilot.
3. Stop/reframe on failure; on pass, run untouched DIV2K validation plus Kodak replication.
4. Produce F5–F9 from preregistered outputs, including failure cases and actual component bytes.
5. Write the manuscript around the result that occurred, not the hoped-for result.
6. Add the outside-class codec context and bounded native-authentic lanes.
7. Package one-command reproduction, environment lock, data/license notes, archived release, and
   supplemental raw per-image evidence.
8. Only then evaluate T1 or a renderer-formulation task through a new registered spike.

## 14. Audit limitations

This is a broad but not exhaustive novelty search. Patents, non-English literature, theses,
unpublished branches, and work released after 2026-07-14 may be missing. Search did not locate an
official Structure-Guided Allocation repository; that means “not found under this search,” not that
no code exists. Some very recent 2026 papers have limited independent replication. Figure-role
analysis used primary PDFs/project pages but did not reproduce external experiments. No new
BENCH-007 numerical result was generated in this task. Novelty confidence therefore remains
provisional, and the publication verdict can change only with new evidence, not with stronger prose.
