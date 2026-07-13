# StructSplat SOTA audit: the 168 KiB benchmark and the July 2026 frontier

**Audit date / literature cutoff:** 2026-07-13

**Repository:** StructSplat

**Report under audit:** [`results/storage_budget_168k_external_present/index.html`](../../results/storage_budget_168k_external_present/index.html)
**Existing research portfolio:** [`ara/evidence/research-portfolio-2026-07-13.md`](research-portfolio-2026-07-13.md)

## Executive verdict

The linked experiment is a useful and unusually well-instrumented **within-StructSplat
optimizer/policy study**. It is not evidence that StructSplat is state of the art in image compression,
and it is not a native comparison with current 2D Gaussian methods.

The main reasons are decisive rather than cosmetic:

1. Its nominal 168 KiB field costs **71.68--81.15 bits per pixel** on the 160-pixel-side
   targets, roughly three times uncompressed RGB's 24 bpp. The actual StructSplat codec stream is
   about **22.0--22.1 bpp**, while the lossless target PNGs average **17.99 bpp**.
2. The GaussianImage++, Image-GS, and GaussianImage rows in the main report are local policy
   analogues using StructSplat's normalized renderer and fitter. The report states this correctly;
   their labels do not make them native method results.
3. The evaluation uses four COCO training images, two seeds, one high-capacity operating point, and
   max side 160. Current papers evaluate full Kodak, DIV2K, or 27--76 megapixel images across rate
   curves and often include an actual entropy-coded bitstream.
4. Several broad claims that might once have distinguished StructSplat now have direct prior art:
   structure-guided allocation/orientation/precision, progressive allocation, learned instant
   initialization, boundary-aware rasterization, normalized ownership mixtures, and clustered
   quantization.

The defensible present contribution is narrower and still valuable: StructSplat is a controllable,
mechanism-rich testbed for handcrafted structure priors under a normalized 2D kernel mixture. Its
publishable question is no longer "does structure help?" but **when, under an actual rate budget,
does tensor-driven blue-noise placement add something that SLIC/Sobel allocation, learned priors,
and ownership-based representations do not?**

## 1. What the repository implements

StructSplat renders a partition-of-unity mixture

\[
\hat I(p)=\frac{\sum_i c_i o_i G_i(p)}{\sum_i o_i G_i(p)+\epsilon},
\]

with anisotropic Gaussian support, optional opacity, and compact-support fade. The implementation is
explicit in [`src/structsplat/render.py`](../../src/structsplat/render.py): the support weight is
`exp(-q/2)` and the accumulated numerator is divided by the accumulated weight.

The structural pipeline is a tensor-derived density/orientation field followed by WSE or a quadtree
variant, then continuous fitting and optional residual growth, geometry loss, color solve, QAT, and
codec export. The shipped initializer is `quadtree_wse` in
[`src/structsplat/config.py`](../../src/structsplat/config.py).

This normalized renderer matters for the novelty audit. It belongs to the same broad mathematical
family as Image-GS and Soft Anisotropic Diagrams: exponentiated local scores are normalized into
pixel responsibilities. StructSplat uses Gaussian quadratic overlap; SAD adds a top-K ownership
map, additively weighted anisotropic distance, and independent radius and temperature.

## 2. What the linked benchmark actually establishes

The report completed 320/320 cells: 40 methods, four images, two seeds, 5,376 final Gaussians,
10,000 requested iterations, and exact CUDA rendering. The strongest local results are:

| Local policy | Mean PSNR | Mean MS-SSIM | Iteration AUC | Total time |
|---|---:|---:|---:|---:|
| SS best default | 48.7992 | 0.999751 | 44.669 | 15.162 s |
| SS best + cosine LR | **51.4881** | 0.999878 | 46.135 | 15.075 s |
| SS best + final color solve | 50.9996 | **0.999880** | 45.070 | 14.594 s |
| GaussianImage fixed, local analogue | 47.3401 | 0.999780 | **46.398** | 13.785 s |
| SS best + L1 only | 47.8830 | 0.999650 | 43.528 | **11.361 s** |

The causal conclusions supported inside this regime are:

- A horizon-wide cosine schedule is the largest float-field endpoint gain: +2.689 dB over the
  pinned default in all eight pairs.
- A terminal fixed-geometry least-squares color solve adds +2.200 dB on average.
- Adaptive growth improves early convergence (+0.789 AUC) and adds +0.175 dB at the exact cap.
- Sobel geometry consistency yields only about +0.08--0.16 dB and costs time.
- Generation-density covariance filtering, low-pass warmup, tensor-weighted loss, feature scale
  caps, and simple relocation do not establish a broad improvement here.
- The Bonferroni familywise relation remains inconclusive for the headline cosine, color-solve, and
  adaptive-growth candidates even where the mean-based promotion table says "yes." Four image
  clusters cannot support a strong generalization claim.

The separate official GaussianImage run is informative but not a component ablation. Native
GaussianImage averages 35.657 dB and 6.392 seconds versus the pinned StructSplat default, a
-13.142 dB / +8.288 second quality-speed tradeoff. Its renderer, loss, parameterization, optimizer,
and schedule all differ.

### A stale-baseline warning

The report's `SS best default` is `aniso_onedge + WSE`, as defined in
[`benchmarks/fair_density_control_compare.py`](../../benchmarks/fair_density_control_compare.py),
whereas the repository's shipped default is now `quadtree_wse`. The name is historical, not the
current public default. In the same report, on-edge and quadtree-WSE variants are effectively tied
at the endpoint, so the benchmark does not establish quadtree-WSE as the causal source of the
headline 51.49 dB result.

## 3. Why 168 KiB is not a compression operating point here

The rate follows directly from [`image_storage.csv`](../../results/storage_budget_168k_external_present/image_storage.csv):

| Prepared target | Pixels | 168 KiB analytical rate | Lossless target PNG |
|---|---:|---:|---:|
| COCO ...000009 | 160 x 120 | 71.680 bpp | 17.927 bpp |
| COCO ...000025 | 160 x 106 | 81.147 bpp | 19.424 bpp |
| COCO ...000030 | 160 x 107 | 80.389 bpp | 13.996 bpp |
| COCO ...000034 | 160 x 106 | 81.147 bpp | 20.606 bpp |

The analytical rate is `172032 * 8 / (width * height)`. It is not normalized to image size, so the
same byte budget becomes a different rate for each aspect ratio. The codec rows reduce the field to
roughly 48 KiB, but the cosine stream still averages 22.107 bpp and about 45.99 dB, larger than the
lossless PNG target on average. This is therefore best described as an **overcomplete fitting and
convergence experiment with a byte cap**, not a rate-distortion compression benchmark.

Other validity limits:

- The same four images have supported repeated repository selection and are not a held-out test set.
- Two seeds do not address method-search multiplicity. Familywise correction across five reported
  metrics is not correction across roughly forty searched policies and their prior tuning history.
- AUC integrates PSNR over iteration, not wall-clock time. Early-stop curves are held constant after
  stopping; 229 rows plateau-stop and 91 reach the maximum horizon.
- Max side 160 erases the high-resolution scaling problem and makes common low-rate settings yield
  only a handful of primitives. It cannot test the megapixel regime targeted by SGI.
- Twenty-four rows overfill the nominal capacity and are correctly excluded from strict equal-rate
  winners, but their presence reinforces that count and rate must be enforced by the bitstream, not
  inferred from a nominal policy.

## 4. Current state of the art: primary-source map

Results below are not put into one synthetic leaderboard: their datasets, renderers, training priors,
rate definitions, and hardware differ. They define the experiments StructSplat must reproduce or
bridge before making comparative claims.

| Method | Frontier and primary evidence | Direct implication for StructSplat |
|---|---|---|
| [Structure-Guided Allocation](https://arxiv.org/abs/2512.24018) (2025) | SLIC/Sobel structural classes, geometry-consistent regularization, and adaptive covariance precision; reports -43.44% Kodak and -29.91% DIV2K BD-rate versus GSImage while retaining >1,000 FPS. | The broad structure-aware allocation/orientation/precision claim is occupied. This is the most important matched handcrafted baseline. |
| [Soft Anisotropic Diagrams](https://arxiv.org/abs/2604.21984) (SIGGRAPH 2026) | A top-K soft anisotropic ownership partition with learned radius/temperature and removal-delta pruning; reports 2.55--3.29 dB over Image-GS at 0.2--0.5 parameter BPP and 46.0 dB in 2.2 s on Kodak at about 16 parameter BPP. The paper explicitly says its BPP is a packed-parameter proxy without entropy coding. | Strongest representation-level threat. StructSplat's normalized renderer should be tested as one point in the same score-family continuum, not treated as categorically separate. |
| [SGI](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_SGI_Structured_2D_Gaussians_for_Efficient_and_Compact_Large_Image_CVPR_2026_paper.pdf) (CVPR 2026) | Seed-organized neural Gaussians, context entropy coding, and multiscale fitting on 27--76 MP images; reports up to 7.5x compression over nonquantized and 1.6x over quantized 2DGS, with 1.6--6.5x faster optimization. | Defines the high-resolution/storage frontier absent from the 160-pixel report. It also shows that structure must reduce transmitted redundancy, not only improve initialization. |
| [AIR](https://arxiv.org/abs/2605.20820) (2026 preprint) | Self-supervised stage-wise residual prediction, stage control, and adaptive quantization; 160--300 ms feed-forward reconstruction. At 52k Kodak Gaussians it reports 32.17 dB in 162 ms; its DIV2K adaptive quantizer reports 30.23 dB at 3.28 bpp in 300 ms. | Defines the amortized-encoding frontier. The repository's 256-pixel four-image AIR run is useful environment evidence but not a paper-protocol replication. |
| [GaussianImage++](https://arxiv.org/abs/2512.19108) (AAAI 2026) | Distortion-driven densification, covariance filtering, and attribute-separated learned quantization. It reports 35.41 dB at 10k Gaussians on Kodak and 31.1 dB at 1.08 bpp for its codec. | Native densification/QAT is required. The report's residual analogue and isolated covariance-filter transplant do not reproduce the method. |
| [Image-GS](https://www.immersivecomputinglab.org/publication/image-gs-content-adaptive-image-representation-via-2d-gaussians/) (SIGGRAPH 2025) | Gradient-informed initialization, error-guided progressive optimization, top-K normalized rendering, random access, and natural LOD. | Owns broad content-adaptive progressive allocation and normalized Gaussian-mixture territory. |
| [P-GSVC](https://arxiv.org/abs/2603.10551) (MMSys 2026) | Jointly optimized base and enhancement Gaussian layers for quality and resolution scalability; reports up to +2.6 dB over sequential image-layer training. | Progressive WSE order is useful engineering evidence, not a new progressive-Gaussian claim. A codec must preserve and jointly optimize the transmitted order. |
| [CGVQ](https://arxiv.org/abs/2607.05667) (SIGGRAPH 2026 poster) | K-means groups appearance/anisotropy before per-cluster codebooks; reports about 20% lower bpp at similar quality and +1.68 dB over GaussianImage at the same 15k primitives, with a substantial encode/decode speed tradeoff. | The newest direct quantization baseline as of this audit. Generic clustered VQ is occupied; a StructSplat codec needs a structure-conditioned entropy argument beyond clustering. |
| [Contour-Aware 2DGS](https://arxiv.org/abs/2512.23255) (ICCE 2026) | Segmentation-region-constrained rasterization prevents cross-boundary mixing and helps especially at very small Gaussian counts. | Direct threat to boundary-gated and interface work. A new claim must avoid external segmentation, account for mask bits, or introduce a different relational primitive. |
| [WIPES](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_WIPES_Wavelet-based_Visual_Primitives_ICCV_2025_paper.html) (ICCV 2025) | Localized wavelet rather than Gaussian primitives; reports 45.87 dB on Kodak and 40.32 dB on DIV2K in its fitting table. | A Gaussian-only benchmark cannot establish the best local primitive. Frequency-bearing primitives are a required representation control. |
| [Instant-GI](https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Instant_GaussianImage_A_Generalizable_and_Self-Adaptive_Image_Representation_via_2D_ICCV_2025_paper.html), [Fast-2DGS](https://openaccess.thecvf.com/content/WACV2026W/WVAQ/html/Wang_Fast_2DGS_Efficient_Image_Representation_with_Deep_Gaussian_Prior_WACVW_2026_paper.html), [EllipssianNet](https://openaccess.thecvf.com/content/WACV2026/html/Kim_EllipssianNet_Image-guided_Sampling_of_2D_Gaussians_for_Gaussian_Splatting_WACV_2026_paper.html) | Learned coarse fields or samplers followed by little or no fitting. Instant-GI reports up to an order-of-magnitude training reduction; the 2026 methods further specialize learned Gaussian priors and image-guided anisotropic sampling. | The learned-initialization frontier is crowded. StructSplat's viable distinction is a no-training, interpretable prior with demonstrated RD or out-of-domain robustness. |

For image compression in the ordinary production sense, 2D Gaussian methods are not the overall
SOTA. Even GaussianImage++ remains below the learned-codec baselines in its own high-rate table,
and SAD carefully frames its BPP as representation storage rather than a replacement for mature
codecs. A 2026 practical learned-codec study reports subjective bitrate savings against AV1, AV2,
VVC, ECM, and JPEG-AI while decoding 12 MP images on a phone; that is a different deployment and
evaluation frontier, but it is the correct outside-class control
([CVPR 2026 primary source](https://openaccess.thecvf.com/content/CVPR2026/html/Tatwawadi_What_Matters_in_Practical_Learned_Image_Compression_CVPR_2026_paper.html)).

## 5. Claim and novelty audit

| Candidate claim | Status after the audit | What would make it defensible |
|---|---|---|
| Structure-aware 2D Gaussian allocation is new | **Rejected as broad novelty.** Structure-Guided Allocation, Image-GS, Instant-GI, EllipssianNet, and AIR cover it from handcrafted and learned directions. | Claim a specific tensor-to-blue-noise relationship and beat SLIC/Sobel under the same renderer, actual bpp, and time. |
| Structure-aware orientation / precision is new | **Rejected as broad novelty.** The structure-guided paper contains both orientation regularization and adaptive covariance bitwidth. | Show that one unified tensor metric jointly controls location, orientation, and code length better than independently designed signals. |
| Normalized anisotropic Gaussian rendering is distinctive | **Weak.** Image-GS and SAD explicitly occupy normalized local mixture/ownership formulations. | Establish a theorem or experiment identifying an irreducible benefit of Gaussian overlap versus independent sharpness/radius and explicit ownership. |
| Progressive Gaussian coding | **Occupied.** Image-GS, LIG, and P-GSVC are direct prior art. | Demonstrate an actual embedded stream with every prefix or packet subset valid and a measurable RD advantage. |
| Boundary-gated Gaussian rendering | **Directly threatened.** Contour-Aware 2DGS already gates rasterization by segmentation regions. | Infer interfaces without an external mask, transmit their cost, and model the boundary as a shared relation rather than duplicated per-splat metadata. |
| WSE ordering is a new representation | **No.** Weighted sample elimination owns the ordering idea; P-GSVC owns progressive Gaussian layers. | Keep it as a correctness/LOD mechanism and test whether the codec preserves its prefix value. |
| StructSplat is SOTA | **Not established.** The linked report is local, high-rate, low-resolution, and mostly non-native. | Full held-out native RD curves, actual bitstreams, standard data, and scaling/latency results. |

The most defensible current research identity is: **an interpretable, training-free structural prior
and causal experimentation substrate for normalized local representations**. That identity can
become publishable if it wins in a sparse actual-rate regime or yields a new mechanism/evidence
result that learned and SLIC-based methods do not expose.

## 6. Updated research portfolio

This section augments rather than replaces the larger causal-capacity portfolio. Novelty labels are
provisional: N1 is incremental, N2 is a new relationship or formulation, N3 changes the primitive or
objective, N4 is a new evidence program, and `-T` denotes a cross-domain transfer.

### Productive recombinations

**P1 -- Actual-bpp structure phase diagram (N2/N4).** Replace the single absolute byte cap with
0.25, 0.5, 1, 2, and 4 actual bpp. Compare tensor-WSE, SLIC/Sobel structural classes, Image-GS
gradient sampling, uniform/WSE, and random allocation under one renderer and optimizer. Prediction:
the structured prior has a positive edge-band and convergence effect only below a critical
contributors-per-pixel regime. Kill it if any gain appears only in the overcomplete regime or only
under proxy bytes.

**P2 -- Gaussian-to-SAD score-family continuation (N2).** Add independent per-site temperature and
reach radius while holding the Gaussian metric and active-set policy fixed, then progressively
introduce top-K ownership. This is not a claim to SAD's mechanism; it is a causal bridge identifying
which degree of freedom explains its advantage. Prediction: independent sharpness matters at step
edges, while WSE placement matters in smooth/texture coverage. Kill it if the extra controls do not
improve held-out sparse-rate distortion after paying their bits.

**P3 -- Two-part MDL codec audit (N2-T).** Optimize and report `R(layout) + R(attributes | layout) +
lambda * D`, not count times a constant. Fit explicit entropy models to Morton deltas, tensor-relative
angles/scales, colors, and cluster assignments. Prediction: WSE regularity is valuable only if its
spatial correlations reduce `R(layout)` enough to offset any distortion loss. Kill it if a simple
random/gradient layout codes as cheaply after sorting.

**P4 -- Exact fixed-byte leave-one-out exchange (N2-T).** Use the normalized compositing equation to
compute a splat's exact removal delta, then spend the released coded bits on the best candidate birth.
SAD already uses removal-delta pruning, so this is an important control rather than a broad novelty
claim. Prediction: coupled death/birth beats monotone densification at <=1 bpp. Kill it if ranking
does not predict post-recovery distortion or costs more than the saved fitting time.

**P5 -- Common/native causal bridge (N2/N4).** Cross the initialization and renderer axes: replay
StructSplat, SLIC/Sobel, and Image-GS initial fields through one renderer; separately replay matched
fields through native renderers. Prediction: a substantial part of the apparent native gap is
renderer/objective interaction rather than placement. Kill it as a research lane if rankings are
stable across both axes; the result is still valuable benchmark evidence.

### Exploratory candidates

**E1 -- Segmentation-free responsibility boundary flux (N2-T).** Measure responsibility mass that
crosses a tensor-normal ridge and penalize only flux inconsistent with the reconstructed edge. Unlike
Contour-Aware 2DGS, no external mask is supplied. Prediction: sparse edge-band error falls without
hurting texture; abandon if it merely sharpens noise or reproduces GCR.

**E2 -- Persistence-stable structural salience (N2-T).** Allocate a small protected budget to edges
or level-set components that persist across blur scale, rather than to large instantaneous Sobel
magnitude. Prediction: thin, low-contrast but stable contours survive at <=0.5 bpp. Abandon if
persistence is just a more expensive ranking of tensor energy.

**E3 -- Codelength-conditioned quantization clusters (N2-T).** Replace CGVQ's appearance/anisotropy
K-means with clusters chosen to minimize total code length in the tensor/adjacency graph, including
cluster-ID overhead. Prediction: structural neighborhoods reduce rate at equal decoded distortion;
abandon if gains disappear after IDs and codebooks are counted.

**E4 -- Mechanism-resolving ownership atlas (N2-T/N4).** Stratify errors by edge distance, texture,
overlap entropy, effective contributor count, and quantization sensitivity. Prediction: WSE,
temperature, and boundary controls win in disjoint strata; abandon global architectural claims if
the gains cannot be localized reproducibly.

### Transformational candidates

**T1 -- Decoder-synchronized structural geometry (N3-T, provisional confidence 35--50%).** Transmit
a low-rate base layer first. Both encoder and decoder deterministically reconstruct its structure
tensor, density, WSE sites, and default orientations; the enhancement stream transmits colors and
only deviations from this regenerated geometry. The primitive is no longer an independently stored
Gaussian but a conditional Gaussian derived from common decoded information. Prediction: position
and rotation bits fall materially at <=1 bpp without a comparable distortion increase. Strongest
threats are P-GSVC's layers, SGI's seed decoder, general predictive coding, and CodecSplat's latent
decoder; no searched 2D image method was found to regenerate handcrafted tensor-WSE geometry from a
decoded base layer. Kill it if base quantization makes site identity unstable or correction bits erase
the saving.

**T2 -- Adjacency-coded ownership complex (N3-T, provisional confidence 25--40%).** Store sites plus a
planar adjacency graph whose shared interfaces carry transition parameters; interiors carry smooth
color models. This changes the grammar from independent overlapping blobs to regions and shared
relations. SAD, Contour-Aware 2DGS, meshes, and vector graphics are strong threats. Prediction:
interface cost scales with boundary complexity rather than Gaussian count and improves sparse sharp
edges. Kill it if graph/interface bytes exceed the saved splats or topology is unstable during fit.

**T3 -- Erasure-robust multiple-description splat code (N3-T, provisional confidence 25--40%).**
Optimize groups so any sufficiently large packet subset, not only a prescribed prefix, is a valid
reconstruction. This changes the success criterion from progressive ordering to graceful random
loss. Progressive image coding, multiple-description coding, P-GSVC, and WSE prefixes are strong
prior art; the possible delta is subset-robust Gaussian responsibility balancing. Prediction: random
10--30% packet loss degrades smoothly with small clean-channel overhead. Kill it if redundancy costs
more than an ordinary base/enhancement stream at realistic loss rates.

## 7. Cross-domain transfer audit

| Candidate | Donor mechanism and preserved structure | Domain mismatch and required adaptation | Main failure mode | Falsifiable prediction |
|---|---|---|---|---|
| P1 / D1 | Statistical-physics phase diagrams: characterize qualitative changes as density crosses a control threshold. Preserve coverage/interaction density. | Images are heterogeneous and finite; use local effective contributors and hierarchical image effects rather than claim a thermodynamic limit. | A smooth trend is overinterpreted as a phase transition. | Structured-init delta changes sign or slope at a reproducible overlap/count threshold. |
| P3 | Minimum description length and two-part codes: select the model by transmitted explanation length plus residual distortion. Preserve rate as evidence, not a nominal count. | Entropy models can be misspecified and decoder overhead matters; use real arithmetic-coded streams and include tables/models. | A flexible entropy model hides complexity off-budget. | Tensor/WSE layouts lower total layout-plus-attribute bits at matched distortion. |
| T1 | Predictive/scalable coding and common information: reconstruct enhancement state from an already decoded base. Preserve decoder synchronization. | WSE site selection is discontinuous under base-layer noise; define canonical fixed-point arithmetic and correction symbols. | Geometry correction entropy cancels the saving. | Base-derived geometry reduces total position/orientation rate by >=25% at equal distortion. |
| E2 | Persistent homology: retain structures stable over scale rather than high at one derivative scale. Preserve lifetime as salience. | Natural texture creates many topological events; restrict to luminance/gradient filtrations and count all side information. | It ranks ordinary strong edges and adds cost only. | Weak persistent contours have lower edge-band error than Sobel allocation at sparse rate. |
| T3 | Multiple-description and erasure/fountain coding: distribute essential information so arbitrary subsets remain useful. Preserve subset robustness. | Gaussian responsibilities are coupled and missing sites alter normalization; train with packet dropout and group-balanced denominators. | Clean-channel RD penalty dominates. | At 20% random loss, PSNR loss is materially below prefix coding at <=5% clean-rate penalty. |
| P5 / E4 | Randomized trials and mechanism-focused clinical measurement: preregister interventions, endpoints, and subgroups. Preserve causal contrast and held-out confirmation. | Pixels and seeds are correlated; randomize at image/method-pair level and use image-cluster inference. | Repeated tuning leaks into the test set. | A preregistered effect replicates on Kodak-24 and DIV2K validation with the same sign. |

## 8. New-evidence discovery programs

### D1 -- Rate/coverage phase-transition map

Intervention: sweep actual bpp and independently vary site count, attribute precision, and renderer
sharpness. Record effective contributors per pixel, responsibility entropy, uncovered mass, edge-band
distortion, texture-band distortion, convergence per second, and actual component codelengths.

Competing explanations:

- tensor-WSE helps because sparse coverage is better;
- it helps because initial covariance orientation is better;
- it only changes early optimization and has no terminal RD value;
- normalized overlap erases the advantage once coverage becomes dense.

The program is informative even if StructSplat loses: it identifies the regime in which handcrafted
structure priors cease to matter and prevents another high-capacity benchmark from answering the
wrong scientific question.

### D2 -- Renderer equivalence and ownership atlas

Intervention: build a nested family from normalized Gaussian overlap to independent temperature and
radius, then top-K ownership, keeping sites/colors and parameter bits matched where possible. Use
synthetic steps, junctions, thin lines, stochastic textures, Kodak, and DIV2K.

Observables: boundary leakage, ownership churn, gradient conditioning, top-K refresh error, random
access cost, and RD. Competing explanations are sharper score parameterization, explicit ownership,
better active-set control, or implementation speed. The result either identifies a minimal
StructSplat-compatible improvement or shows that the Gaussian-overlap grammar itself is the limit.

## 9. Pareto shortlist

Scores are 0--5; novelty and value are intentionally separate.

| Candidate | Novelty | Falsifiability | Importance | Feasibility | Cheap first test | Informative failure |
|---|---:|---:|---:|---:|---:|---:|
| P1 actual-bpp phase diagram | 2 | 5 | 5 | 4 | 4 | 5 |
| P2 score-family continuation | 2 | 5 | 5 | 3 | 4 | 5 |
| P3 two-part MDL audit | 2 | 5 | 5 | 4 | 4 | 5 |
| P4 fixed-byte exchange | 2 | 5 | 4 | 3 | 3 | 5 |
| P5 common/native bridge | 2 | 5 | 5 | 3 | 3 | 5 |
| E2 persistence salience | 2 | 4 | 3 | 3 | 3 | 4 |
| T1 decoder-synchronized geometry | 4 | 5 | 5 | 2 | 3 | 5 |
| T2 ownership complex | 3 | 5 | 5 | 2 | 2 | 5 |
| T3 erasure-robust code | 3 | 5 | 3 | 2 | 3 | 4 |

The highest-value immediate work is P1. P5 becomes active only if the common-renderer pilot exposes
a meaningful renderer/objective interaction. The highest-upside new representation is T1, gated on
P1 showing that structure survives actual coding.

## 10. Recommended first experiment

Run a preregistered **actual-rate structure phase diagram**, not another 168 KiB sweep.

**Null hypothesis.** Tensor-WSE does not improve held-out distortion or convergence over the
strongest nonlearned direct prior, SLIC/Sobel structure-guided allocation, at equal actual bpp and
wall-clock budget in the sparse regime.

**Development and test split.** Use the four current COCO images for plumbing only and separate,
declared DIV2K-training subsets for rate calibration and the eight-image killing pilot. Freeze the
protocol, then evaluate all 100 untouched DIV2K validation images only on a pilot pass.
All 24 Kodak images are a development-exposed replication set because this repository has already
used them repeatedly; they are not held-out confirmation.

**Rate points.** 0.25, 0.5, 1, 2, and 4 actual bpp at native resolution. Include every header,
entropy table, codebook, mask, and model needed by the decoder. Also report parameter-proxy BPP, but
never substitute it for the stream rate.

**First-stage common-renderer arms.** Tensor-WSE; SLIC/Sobel structural allocation; Image-GS-style
gradient allocation; uniform WSE; and random fixed count. Match renderer, loss, optimizer, schedule,
final coded rate, and time. Factor location density and covariance orientation so the tensor mechanism
is identifiable.

**Native confirmation arms.** Official Structure-Guided Allocation, Image-GS, GaussianImage++, SAD,
and WIPES at overlapping rate points. Native results answer method competitiveness; common-renderer
results answer causality. Do not collapse the two questions.

**Primary endpoint.** PSNR BD-rate on the held-out images. Secondary endpoints: MS-SSIM/LPIPS,
encoding time, decoding time/FPS, peak memory, edge/texture-band errors, effective contributor count,
and stream component bytes.

**Promotion rule.** Continue the tensor-WSE scientific claim only if it delivers either (a)
PSNR BD-rate <=-10% versus SLIC/Sobel, or (b) at least +0.25 dB at both 0.5 and 1 bpp, with an
image-cluster 95% interval above zero, no >10% encoding-time regression, and a preregistered mechanism
observable moving in the predicted direction.

**Abandon or reframe if** the gain exists only above 4 bpp, disappears after actual coding, vanishes
under the direct SLIC/Sobel control, or cannot be reproduced in a native bridge. In that case,
position StructSplat as a research harness and redirect method work to T1 or the P2 renderer study.

## 11. Audit limitations

Novelty is provisional: patents, non-English work, unpublished code, and papers released after the
cutoff may be missing. Several 2026 sources are recent preprints or posters without independent
replication. Cross-paper numbers are contextual, not directly comparable. No new native benchmark was
run for this audit; the conclusion follows from repository artifacts, exact rate accounting, existing
native runs, and primary-source method/protocol comparison. The audit intentionally did not modify
the implementation or promote a default.

## 12. Repository handoff created from this audit

- Experiment contract: `tasks/BENCH-007-actual-rate-structure-phase-diagram.md`
- Conditional causal bridge: `tasks/BENCH-008-common-native-causal-bridge.md`
- High-risk rate-saving spike: `tasks/COMP-005-decoder-synchronized-structural-geometry.md`
- Copy-paste continuation prompt:
  `ara/prompts/continue-structsplat-actual-rate-research.md`

These documents convert the audit into staged, falsifiable work. They do not claim that the
experiments have already been implemented or run.
