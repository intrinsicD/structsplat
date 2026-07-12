# Research Portfolio: Causal Capacity Management for StructSplat

**Repository/domain:** StructSplat; normalized 2D Gaussian image representation, fitting, and compression.

**Literature cutoff:** 2026-07-13.

**Sources searched:** arXiv, CVF Open Access, AAAI proceedings, PMLR, SIAM, ACM/project pages,
official GitHub repositories, and the live StructSplat task/code/evidence tree. Primary sources
included [GaussianImage](https://arxiv.org/abs/2403.08551),
[Image-GS](https://arxiv.org/abs/2407.01866),
[GaussianImage++](https://ojs.aaai.org/index.php/AAAI/article/view/37572),
[Instant GaussianImage](https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Instant_GaussianImage_A_Generalizable_and_Self-Adaptive_Image_Representation_via_2D_ICCV_2025_paper.html),
[EigenGS](https://openaccess.thecvf.com/content/CVPR2025/html/Tai_EigenGS_Representation_From_Eigenspace_to_Gaussian_Image_Space_CVPR_2025_paper.html),
[SGI](https://arxiv.org/abs/2603.07789), [AIR](https://arxiv.org/abs/2605.20820),
[Soft Anisotropic Diagrams](https://arxiv.org/abs/2604.21984),
[SteepGS](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.html),
[ResGS](https://arxiv.org/abs/2412.07494),
[generalized matching pursuit](https://proceedings.mlr.press/v54/locatello17a.html),
[weighted sample elimination](https://www.cemyuksel.com/research/sampleelimination/), and
[adaptive finite-element error control](https://epubs.siam.org/doi/10.1137/S0036144502409093).

**Key unresolved assumptions:** a high pointwise residual is the best birth site; activity is a
causal deletion score; more Gaussians should be added monotonically; a field is best understood as
independent blobs rather than an overlap/responsibility system; global per-attribute precision is
adequate; and independent CUDA trajectories are precise enough for sub-0.1 dB decisions.

## 1. Frontier map

| Component/frontier | Current substrate | Strongest evidence | Remaining failure |
|---|---|---|---|
| Initialization | Tensor density, anisotropic WSE, quadtree-WSE, on-edge/flanking controls | ABL-006 retired flanking; quadtree-WSE is the shipped high-budget choice | Initial placement is no longer the dominant unknown |
| Growth | Raw residual/tensor sampled-add, support/ranked/AbsGrad/frequency splits, relocation | Tensor sampled-add is the strongest existing broad control; many micro-levers lost | Birth sites are scored pointwise, not by predicted marginal utility of a child footprint |
| Optimization | Adam/Adan, color solve, geometry loss, checkpoint selection | GCR helps edge-heavy pixels; checkpointing repairs sparse long fits | Atomic nondeterminism and overlap conditioning obscure causal attribution |
| Representation | Normalized constant/affine colors, optional opacity/background/filtering | Background and opacity are promising; fixed covariance filters lost | Responsibilities, overlap disagreement, and boundary leakage are not explicit primitives |
| Compression | Self-describing scalar codec, QAT/rate proxy, Morton deltas | Post-fit QAT remains strongest locally | No causal structural RD controller, local entropy graph, or mixed precision |
| Measurement | PSNR/MS-SSIM/LPIPS, AUC, native-reference provenance | Strong audit discipline and resumable harnesses | Mechanism-blind aggregates; incomplete progressive-prefix and fixed-storage evidence |

Dominant paradigms are pointwise residual growth, gradient descent on a fixed field between
restructuring events, and global scalar budgets. Densely explored regions include loss swaps,
scale caps, low-pass curricula, simple relocation, optimizer-state seeding, and covariance floors.
Sparse regions are causal birth/death utility, normalized overlap diagnostics, fixed-budget swaps,
progressive within-level ordering, and structure-priced encoded rate.

## 2. Functional problem signature

A dense vector signal is approximated by a finite, permutation-invariant set of local anisotropic
kernels. Continuous attributes and discrete structural events share a finite count/byte/time budget.
Each kernel receives information only through clipped local support, but overlapping kernels are
globally coupled by a normalized denominator. The hidden control problem is to decide whether a
local error requires new support, a split, a color/geometry correction, deletion elsewhere, or more
precision. The continuous state is position/covariance/orientation/color/visibility; the discrete
state is count, birth cohort, hierarchy, event history, and codec symbols. Rotation is modulo pi,
isotropic orientation is unidentified, and relative visibility has a near-global gauge. Boundaries
clip supports; unsupported pixels approach black through epsilon. Compute is limited by repeated
render/backward work and atomic reductions; evidence is limited by trajectory noise and expensive
paired cells.

## 3. Fixation anti-library

Do not label the following transformational: another generic tensor-weighted loss, stronger scale
caps, a fixed covariance floor, another low-pass curriculum, simple residual relocation, Adam
moment seeding, generic learned initialization, additive rendering as a universal hierarchy fix,
or “add attention/multiscale/a network.” These are already tested, known, or both. Generic VQ/QAT,
residual densification, progressive Gaussian allocation, and learned warm starts are also direct
prior art. A new candidate must isolate a causal mechanism or change the state/objective.

## 4. Productive recombinations

### Candidate P1 — Kernel-matched signed-residual birth

- **Central claim:** selecting a child footprint by signed RGB residual correlation at the planned
  Gaussian scale improves immediate loss and short recovery over top pointwise absolute residual.
- **Novelty class:** N2-T, promoted to an N4 evidence program only after the pilot observation.
- **Known foundation:** matching pursuit, Gaussian dictionaries, Image-GS/ResGS residual growth.
- **Irreducible delta:** correlate the residual with the actual planned footprint before allocating
  a StructSplat child; pointwise magnitude cannot express cancellation inside that footprint.
- **Why this is not merely A + B:** the scientific question is whether footprint-scale residual
  coherence, not residual magnitude, is the missing sufficient statistic for structural birth.
- **Changed grammar or transfer mechanism:** atom–residual inner product becomes the birth operator;
  tensor geometry remains fixed so site scoring is isolated.
- **New prediction:** gains concentrate where pointwise residual changes sign within one child
  support; immediate split shock and 20-step recovery improve without more Gaussians.
- **Cheapest killing test:** raw absolute versus matched signed score on identical 64→80 and
  512→640 waves; record immediate/post-20/post-100 PSNR and score overhead.
- **Prior-art threats:** classic Gaussian matching pursuit, ResGS, SteepGS, and StructSplat's own
  residual-tensor addition. The normalized-renderer relationship is at most candidate-novel.
- **Novelty confidence:** 25–45%; known components, possibly new relationship under this compositor.
- **Scientific value:** a positive result gives a principled, cheap densification statistic; a
  negative result maps where additive sparse-approximation intuition fails under normalization.
- **Publishable if successful:** as part of a broader causal birth/death framework.
- **Publishable if partially successful:** as a regime map by residual frequency and overlap.
- **Publishable if it fails informatively:** as a counterexample showing pointwise error is adequate.

### Candidate P2 — Exact leave-one-out removal-delta pruning

- **Central claim:** actual pixel-loss increase after analytical removal ranks expendable normalized
  Gaussians better than opacity-weighted activity at equal prune count.
- **Novelty class:** N2-T; direct transfer, not a novelty claim.
- **Known foundation:** Soft Anisotropic Diagrams' closed-form removal delta.
- **Irreducible delta:** adapt exact leave-one-out loss to clipped Gaussian supports and StructSplat's
  epsilon/renderer semantics.
- **Why this is not merely A + B:** it replaces a non-causal proxy with the intervention itself.
- **Changed grammar or transfer mechanism:** deletion is evaluated as a counterfactual render.
- **New prediction:** broad redundant Gaussians can have high activity but near-zero removal cost.
- **Cheapest killing test:** analytical versus brute-force delta parity at N=128, then prune 10/20%.
- **Prior-art threats:** SAD already contains the central mechanism.
- **Novelty confidence:** 5–15%; repository adaptation only.
- **Scientific value:** establishes a causal death score needed by future fixed-budget swaps.
- **Publishable if successful:** only as one component of a larger RD controller.
- **Publishable if partially successful:** a pruning benchmark and normalized-renderer analysis.
- **Publishable if it fails informatively:** identifies SSIM/nonlocal or epsilon failure regimes.

### Candidate P3 — Progressive WSE order and embedded LOD

- **Central claim:** reversing the full WSE elimination history yields within-level prefixes with
  better coverage/spectral behavior than current candidate-index ordering, without changing N-set.
- **Novelty class:** N1/N2 correctness repair.
- **Known foundation:** Yuksel's progressive weighted sample elimination and Image-GS LOD.
- **Irreducible delta:** make StructSplat's documented anisotropic progressive-prefix contract real.
- **Why this is not merely A + B:** the full set is invariant; the experiment isolates ordering.
- **Changed grammar or transfer mechanism:** every prefix becomes a first-class representation.
- **New prediction:** N/16..N/2 prefixes improve maximum holes and low-frequency spectral energy.
- **Cheapest killing test:** compare sorted versus progressive prefixes without fitting changes.
- **Prior-art threats:** progressive WSE is known; P-GSVC/PCGS cover progressive Gaussian streams.
- **Novelty confidence:** 5–15%.
- **Scientific value:** fixes a false internal assumption and enables honest progressive streams.
- **Publishable if successful:** as infrastructure inside a stronger codec result.
- **Publishable if partially successful:** as a prefix-quality correction.
- **Publishable if it fails informatively:** retire the within-level prefix claim.

### Candidate P4 — Sensitivity-weighted mixed precision

- **Central claim:** per-Gaussian/attribute bit classes allocated by realized distortion sensitivity
  improve actual bpp–quality over fixed group precision after accounting for class-map overhead.
- **Novelty class:** N2-T.
- **Known foundation:** water-filling, adaptive quantization, structure-guided allocation, compressed GS.
- **Irreducible delta:** price each StructSplat attribute by local output sensitivity and actual bytes.
- **Why this is not merely A + B:** the hypothesis concerns correlation between local renderer
  sensitivity and encoded RD utility, not merely learnable bitwidth.
- **Changed grammar or transfer mechanism:** precision becomes structural state.
- **New prediction:** color and small-scale geometry receive systematically different bit classes.
- **Cheapest killing test:** finite perturbations at N=256, fixed total raw bits, then real encoding.
- **Prior-art threats:** adaptive bitwidth quantization is already explicit in 2025 structure-guided 2DGS.
- **Novelty confidence:** 10–25%.
- **Scientific value:** directly attacks the active codec ladder.
- **Publishable if successful:** with a geometry-conditioned entropy model.
- **Publishable if partially successful:** a sensitivity/quantization diagnostic.
- **Publishable if it fails informatively:** demonstrates class-map and estimator overhead limits.

## 5. Exploratory candidates

### Candidate E1 — Responsibility-normalized residual-moment split

- **Central claim:** residual responsibility covariance identifies which parent to split and the
  useful split axis better than pointwise target tensor or support-absolute residual.
- **Novelty class:** N2-T.
- **Known foundation:** SAD residual-moment splitting; adaptive error estimators.
- **Irreducible delta:** use unexplained residual moments inside each Gaussian's normalized ownership.
- **Why this is not merely A + B:** the residual, not target structure or Gaussian geometry, defines
  the child's direction.
- **Changed grammar or transfer mechanism:** ownership cells become diagnostic objects.
- **New prediction:** it beats tensor addition on junctions and mixed residual signs.
- **Cheapest killing test:** one difficult-four 64→80 wave versus tensor/ranked/frequency controls.
- **Prior-art threats:** SAD directly threatens the mechanism.
- **Novelty confidence:** 10–25%.
- **Scientific value:** interpretable within-support structural correction.
- **Publishable if successful:** in a normalized responsibility framework.
- **Publishable if partially successful:** failure taxonomy by overlap.
- **Publishable if it fails informatively:** shows ownership moments do not transfer from diagrams.

### Candidate E2 — Geometry-causal entropy prediction

- **Central claim:** already-decoded geometric neighbors predict Gaussian attributes better than a
  one-dimensional Morton context after graph metadata cost.
- **Novelty class:** N2-T.
- **Known foundation:** autoregressive learned codecs, HAC, SGI hash-grid context.
- **Irreducible delta:** explicit causal Gaussian adjacency rather than seed-grid or Morton-window context.
- **Why this is not merely A + B:** the test compares conditional entropy under competing causal graphs.
- **Changed grammar or transfer mechanism:** neighborhood topology becomes codec state.
- **New prediction:** rotations/scales gain more than colors from geometric context.
- **Cheapest killing test:** entropy/zlib proxy on existing Kodak fields before implementing a coder.
- **Prior-art threats:** SGI and Gaussian anchor-context codecs.
- **Novelty confidence:** 15–30%.
- **Scientific value:** cheap go/no-go for COMP-003's entropy rung.
- **Publishable if successful:** with actual arithmetic coding and RD curves.
- **Publishable if partially successful:** attribute-specific graph results.
- **Publishable if it fails informatively:** validates simpler Morton context.

### Candidate E3 — Gauge-fixed relative visibility

- **Central claim:** centered positive relative weights reproduce opacity-free rendering at zero and
  retain the observed opacity gain without an unidentified global sigmoid scale.
- **Novelty class:** N2.
- **Known foundation:** mixture-model logits and normalized positive measures.
- **Irreducible delta:** explicitly quotient the normalized renderer's global weight gauge.
- **Why this is not merely A + B:** it changes the identifiable parameterization, not the loss.
- **Changed grammar or transfer mechanism:** visibility is an equivalence class modulo global scale.
- **New prediction:** lower logit spread and more stable codec ranges than sigmoid opacity.
- **Cheapest killing test:** four existing ABL-005 cells: none versus sigmoid versus centered-exp.
- **Prior-art threats:** normalized mixtures and softmax gating are standard.
- **Novelty confidence:** 10–25%.
- **Scientific value:** explains a large early opacity signal.
- **Publishable if successful:** only with broad evidence and identifiability analysis.
- **Publishable if partially successful:** a better research parameterization.
- **Publishable if it fails informatively:** global gauge was not the practical instability.

### Candidate E4 — Mechanism-resolving overlap diagnostics

- **Central claim:** gradient-band error, effective contributor count, overlap color disagreement,
  and coverage separate edge, hole, and mixture-conflict improvements hidden by aggregate PSNR.
- **Novelty class:** N2-T measurement transfer.
- **Known foundation:** mixture effective sample size and conditional error analysis.
- **Irreducible delta:** a normalized-renderer mechanism atlas bound to each benchmark row.
- **Why this is not merely A + B:** diagnostics are falsifiers for competing causal explanations.
- **Changed grammar or transfer mechanism:** the unit of observation becomes pixel mechanism band.
- **New prediction:** GCR gains concentrate at q99 target gradients, while background gains
  concentrate in low-coverage/low-frequency pixels.
- **Cheapest killing test:** retrofit saved GCR/default PNGs, then one raw-tensor rerun.
- **Prior-art threats:** error maps and edge metrics are common; the joint mechanism attribution is local.
- **Novelty confidence:** 10–20%.
- **Scientific value:** high information gain even without algorithmic novelty.
- **Publishable if successful:** as evaluation methodology supporting a larger method.
- **Publishable if partially successful:** reusable diagnostic suite.
- **Publishable if it fails informatively:** aggregate metrics remain sufficient for this regime.

## 6. Transformational candidates

### Candidate T1 — Fixed-budget pairwise RD birth–death active set

- **Central claim:** alternating a marginal-utility birth with a counterfactual-utility death at
  fixed actual rate improves RD without monotonic count growth.
- **Novelty class:** N3-T.
- **Known foundation:** pairwise conditional gradients, exchange methods, sparse measures, RDO-Gaussian.
- **Irreducible delta:** represent fitting as active-set exchange over a continuum of Gaussian atoms,
  priced by actual encoded rate.
- **Why this is not merely A + B:** it changes admissible structural dynamics and the objective.
- **Changed grammar or transfer mechanism:** a field is a rate-constrained active measure; add/delete
  are a coupled step, not separate heuristics.
- **New prediction:** quality can improve at exactly fixed bytes and N when birth/death utilities
  are anti-correlated.
- **Cheapest killing test:** enumerate a small candidate bank and all single deletions at N=128.
- **Prior-art threats:** conditional-gradient sparse inverse problems, SAD pruning, RDO-Gaussian.
- **Novelty confidence:** 35–55%; apparently unexplored exact relationship under searched 2DGS work.
- **Scientific value:** unifies densification, pruning, relocation, and codec rate.
- **Publishable if successful:** a compact normalized-Gaussian active-set method.
- **Publishable if partially successful:** reliable marginal utility estimators.
- **Publishable if it fails informatively:** maps non-additivity barriers to greedy exchange.

### Candidate T2 — Capacity-constrained transport-cell Gaussians

- **Central claim:** partitioning image information mass into equal-price anisotropic cells and
  deriving Gaussian moments from cells yields better early AUC per primitive than sampling points.
- **Novelty class:** N3-T.
- **Known foundation:** blue noise through optimal transport, CVT, EA-GI, SAD.
- **Irreducible delta:** the primitive is an owned information cell whose moments induce a Gaussian.
- **Why this is not merely A + B:** it reverses point-first construction; cells precede atoms.
- **Changed grammar or transfer mechanism:** capacity is transported and conserved, not sampled.
- **New prediction:** local residual mass per cell is narrower and init AUC less count-sensitive.
- **Cheapest killing test:** dense entropic transport at 64–128 px, N<=256.
- **Prior-art threats:** BNOT, equal-entropy quadtree allocation, differentiable anisotropic diagrams.
- **Novelty confidence:** 30–50%.
- **Scientific value:** principled link between allocation, moments, and rate.
- **Publishable if successful:** with transport-derived covariance/bit prices.
- **Publishable if partially successful:** better initialization or an impossibility region.
- **Publishable if it fails informatively:** point processes are adequate at this scale.

### Candidate T3 — Boundary/interface responsibility primitives

- **Central claim:** a soft half-plane ownership gate or signed interface primitive reduces
  cross-boundary bleed per stored parameter beyond any pure positive Gaussian field.
- **Novelty class:** N3.
- **Known foundation:** CORE-007/008, edge primitives, anisotropic diagrams.
- **Irreducible delta:** the primitive explicitly represents a discontinuity relation, not only a blob.
- **Why this is not merely A + B:** the representable function grammar changes from positive local
  averages to side-aware interfaces.
- **Changed grammar or transfer mechanism:** boundaries become first-class state.
- **New prediction:** equal-byte step/T-junction error falls without increasing interior holes.
- **Cheapest killing test:** reference-only synthetic steps, diagonals, lines, and junctions.
- **Prior-art threats:** contour-aware splats, SAD partitions, Gaussian-windowed edge/Gabor bases.
- **Novelty confidence:** 25–45%.
- **Scientific value:** tests the limit of blob-only image representation.
- **Publishable if successful:** as a hybrid compact image primitive.
- **Publishable if partially successful:** a boundary-only specialist layer.
- **Publishable if it fails informatively:** quantifies the adequacy of positive Gaussians.

## 7. Cross-domain transfers

| Candidate | Donor field | Preserved mechanism | Broken correspondences | Required invention | Adoption barrier | Recipient-specific prediction |
|---|---|---|---|---|---|---|
| P1 | Matching pursuit / sparse inverse problems | Signed atom–residual correlation predicts greedy decrease | Rational normalization; vector color; changing geometry; SSIM | Effective normalized candidate score and cheap separable evaluation | Candidate bank cost | Larger immediate drop and shorter recovery where residual signs cancel locally |
| T1 | Pairwise conditional gradients / market-style exchange | Move mass from least to most useful active atom under a conserved budget | Nonconvex atoms; actual byte cost; joint optimizer transients | Coupled birth/death oracle and rate price | Utility estimates may be poorly ranked | Positive fixed-byte distortion decrease without count growth |
| T2 | Optimal transport / adaptive meshing | Partition conserved error/information mass into capacity-limited cells | Image structure is not a PDE mesh; anisotropic metric varies; renderer overlaps cells | Moment-to-Gaussian map and unequal bit-priced capacities | OT cost and local minima | Narrower per-cell residual mass and count-robust early AUC |
| E4 | Mixture diagnostics / experimental medicine subgroup analysis | Stratified observables distinguish mechanisms hidden in a global mean | Pixels are spatially dependent; post-selection bands can bias claims | Predeclared bands plus overlap-gauge-invariant observables | Report complexity | GCR gains localize to q99 edges; background gains localize elsewhere |
| P3 | Progressive sampling / successive-refinement coding | Every prefix is a valid lower-rate description | Normalized later atoms change earlier denominator; attributes need layered coding | True WSE order and self-contained enhancement streams | Final-stream overhead | Better prefix spectral/coverage metrics with identical terminal set |
| Stage-screen protocol | Industrial design of experiments / active learning | Select interventions that discriminate hypotheses per unit cost | Mixed categorical stages; high interactions; heteroscedastic images | Canonicalized blocked design and foldover | Review trust in partial designs | Recover dense-sweep main effects/interactions with <=30% cells |

Rarely connected donors here are adaptive finite elements/transport cells, market-style pairwise
exchange, and industrial design of experiments. The terminology-removal, structural-map,
causal-preservation, counter-analogy, native-baseline, and historical-obviousness tests downgrade
P1/P2/P3/E4 to useful transfers rather than transformational claims. T1 and T2 survive provisionally
because they change the admissible structural state/objective and imply distinct fixed-budget or
mass-conservation predictions.

## 8. New-evidence discovery programs

### Program D1 — Birth intervention matrix

- **Search space:** raw absolute, tensor-smoothed absolute, matched signed residual, and bounded
  denominator-aware scores crossed with target, unit-residual, and clipped-leverage color init.
- **Observable:** predicted versus realized immediate loss reduction, post-20/post-100 recovery,
  out-of-range child color, AUC, and wall time.
- **Conventional expectation:** highest pointwise residual should be adequate after Adam recovery.
- **Surprising signature:** signed footprint correlation predicts realized gain across images while
  pointwise residual does not.
- **Promotion rule:** >=+0.10 dB post-20 paired mean, positive on every image, retained at step 100,
  <=15% score overhead.
- **Null hypothesis:** matched scoring has no paired recovery or endpoint advantage at equal count.
- **Falsification and controls:** identical base field/candidate count/geometry/color, deterministic
  CPU pilot, multi-seed CUDA confirm, exact hashes/config, no searched arm promoted as default.
- **Reproduction package:** tests, stage-search rows, source hashes, seeds, immediate and trajectory metrics.

Pilot observation: on four 64 px COCO images after a 40-step deterministic CPU fit and one 16-child
wave, raw absolute selection averaged 19.2496 dB immediately and 21.3366 dB after 20 recovery steps;
matched signed residual at one planned child sigma averaged 20.4348 and 21.6364 dB. The +0.2998 dB
post-20 gain was positive on all four images. A naive center-exact denominator color instead fell to
14.2056 dB immediately and 20.2706 after recovery, so that unbounded color rule is rejected before
implementation. The observation is screening evidence, not a default claim.

### Program D2 — Mechanism atlas

- **Search space:** existing GCR, tensor-loss, opacity, background, cap, and default artifacts.
- **Observable:** target-gradient quantile error, signed cross-edge bleed, coverage, effective
  contributor count, and overlap color disagreement.
- **Conventional expectation:** aggregate quality gain is spatially broad.
- **Surprising signature:** each candidate affects a distinct mechanism band with little overlap.
- **Promotion rule:** repeated band-specific effects on raw renders across >=8 images and two seeds.
- **Null hypothesis:** mechanism bands do not explain candidate deltas beyond aggregate noise.
- **Falsification and controls:** predeclared quantiles, paired same-target rows, raw-tensor reruns,
  no inference from isolated 8-bit PNGs.
- **Reproduction package:** per-band CSV, definitions, target hashes, and synthetic controls.

Retrospective screening found GCR's display-RGB/gradient-error benefit largest in the top 1% target
gradient band on both COCO proxy and Kodak4, supporting—but not yet proving—its intended mechanism.

### Program D3 — Progressive-prefix certificate

- **Search space:** current index order versus reversed full elimination history at N/16..N.
- **Observable:** weighted minimum spacing, maximum hole, low-frequency spectral energy, prefix PSNR.
- **Conventional expectation:** order does not matter because terminal WSE sets match.
- **Surprising signature:** progressive order dominates low-count prefixes without terminal change.
- **Promotion rule:** improve coverage and spectral metric at most prefixes with identical terminal set.
- **Null hypothesis:** the documented progressive property has no measurable benefit.
- **Falsification and controls:** exact set equality, deterministic candidates, isotropic and anisotropic metrics.
- **Reproduction package:** prefix arrays, seeds, spectra, and tests.

## 9. Pareto frontier

Scores are 0–5 and keep novelty separate from value.

| Candidate | Apparent novelty | Falsifiability | Importance | Feasibility | First-test cost | Informative failure | Publication potential |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 matched residual | 2 | 5 | 4 | 5 | 5 | 5 | 3 |
| P2 removal delta | 1 | 5 | 4 | 4 | 4 | 5 | 2 |
| P3 progressive WSE | 1 | 5 | 3 | 5 | 5 | 5 | 2 |
| P4 mixed precision | 2 | 4 | 4 | 3 | 3 | 4 | 3 |
| E4 mechanism atlas | 2 | 5 | 4 | 5 | 5 | 5 | 3 |
| T1 RD active set | 4 | 4 | 5 | 2 | 2 | 5 | 5 |
| T2 transport cells | 4 | 4 | 4 | 2 | 2 | 5 | 4 |
| T3 interface primitives | 3 | 5 | 4 | 3 | 4 | 5 | 4 |
| Deterministic backward | 2 | 5 | 5 | 2 | 2 | 5 | 4 |

Pareto representatives: P1 fastest positive algorithmic test; E4 strongest measurement return;
P3 cheapest correctness repair; deterministic backward strongest systems/causal direction; T1
highest integrated importance; T2 strongest theory direction; T3 strongest representation change.

## 10. Recommended first experiment

Implement P1 as `FIT-017`: a new sampled-add score that filters signed RGB residual after
separable Gaussian filtering at the planned child scale, while retaining the existing tensor-aligned
child geometry, target color, count, and optimization schedule. This isolates selection from the
rejected color correction. First validate cancellation/coherence on synthetic residuals, then rerun
the four-image deterministic pilot and a bounded 160 px paired CUDA screen. Abandon if the post-20
gain is below +0.10 dB, vanishes by step 100/final, or scoring adds more than 15% wall time. Keep the
feature opt-in unless a larger fair-regime confirmation passes repository promotion gates.

## 11. Audit limitations

Search cannot prove global novelty; patents, non-English sources, private branches, and unpublished
2026 work may be missing. Several 2026 papers are recent preprints with limited independent
replication. The positive matched-residual pilot is four images at 64 px on deterministic CPU and
must not be generalized. CUDA atomic nondeterminism can exceed small candidate deltas. The GCR band
audit used saved 8-bit PNGs. SAD directly threatens removal-delta and residual-moment novelty;
Yuksel directly owns progressive WSE; adaptive bitwidth and progressive Gaussian coding are known.
The strongest prior-art threat to P1 is that classical matching pursuit plus ResGS may reconstruct
the central mechanism, leaving only a compositor-specific adaptation.
