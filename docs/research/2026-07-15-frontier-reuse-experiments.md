# StructSplat frontier, transferable mechanisms, and killing experiments

**Date:** 2026-07-15
**Literature cutoff:** 2026-07-15
**Repository state inspected:** working tree at the start of this study, excluding `ara/`
**Research prompt used:** [`docs/prompts/real-research.md`](../prompts/real-research.md)
**Claim standard:** "apparently unexplored" always means only under the search disclosed here.

## Executive answer

StructSplat is a strong causal-research substrate, but the available evidence does not establish it
as a state-of-the-art image codec. The completed actual-rate study found a low-rate tensor-WSE
advantage but rejected its promotion: `+0.3457 dB` at 0.5 bpp shrank to `+0.0089 dB` at 1.0 bpp,
mean BD-rate was only `-4.5417%` against a frozen `-10%` gate, fit plus search cost `1.4752x`, and
texture MSE regressed `7.2883%`. That makes another allocation-only compression claim a poor bet.

The 2026 frontier changes the best research targets:

1. SAD shows that explicit ownership, independent reach and hardness, persistent top-K candidates,
   and GPU-local reductions can improve quality and fitting speed together.
2. WIPES shows that a frequency-bearing primitive can beat Gaussian-only dictionaries.
3. SGI shows that hierarchy becomes compression only when the decoder regenerates structured
   Gaussians and an entropy model turns regularity into actual bits.
4. AIR shows that convergence can be amortized by distilling short per-image optimization into a
   predictor, while P-GSVC shows that progressive prefixes must be jointly trained.

The most defensible near-term opportunities at study start were: (a) ownership-aware diagnostics
and recovery-response modeling, (b) a certified bounded-contributor renderer, (c) exact gauge
invariance as an allocator correctness test rather than a promoted method, and (d) actual-rate
tests of richer local color/atom grammars. E1/FIT-019 later confined gauge grouping to correctness,
E2/FIT-020 rejected its frozen one-bend recovery predictor, and E4/COMP-006 found that exact bytes
change fine-grained precision allocation but do not make one extra standard Gaussian competitive.
The clean unresolved axes are now exact backward performance and a real equal-byte richer-atom/
attribute-codec test. The exact SAD
responsibility-density score is **not novel**; the literature audit found the same mass-normalized
formula and default exponent. It is retained only as a controlled mechanism transplant.

Five bounded experiments accompany this report:

- a shared-start responsibility-density densification guard for quality and recovery;
- a top-K responsibility audit that measures quality loss and an oracle support-work ceiling, not
  implementation speed;
- an exact opacity-gauge allocation audit with a disjoint procedural recovery guard;
- a ranked deduplication perturb--recover assay with held-out target variants and a frozen response
  predictor; and
- a complete-stream marginal-RD birth/replacement/precision audit with an exact same-source replay.

Their measured decisions are filled in below from immutable result artifacts after execution.

## Search method and scope

The search used primary papers, official proceedings/project pages, and official repositories.
Queries covered exact method names and functional synonyms for 2D Gaussian image representation,
normalized mixtures, differentiable Voronoi/Apollonius diagrams, residual densification,
split/prune mixture models, top-K rasterization, structured Gaussian entropy models, wavelet/Gabor
primitives, progressive Gaussian layers, amortized Gaussian prediction, and learned image codecs.
Adjacent-field searches covered adaptive mesh refinement, centroidal and capacity-constrained
Voronoi methods, optimal transport, mixture-model identifiability and split/merge moves, database
top-K query maintenance, domain decomposition, sheaves, defect topology, variable projection,
minimum-description-length model selection, and rateless/progressive coding.

Primary direct sources:

- [GaussianImage (ECCV 2024)](https://arxiv.org/abs/2403.08551)
- [Image-GS (SIGGRAPH 2025)](https://arxiv.org/abs/2407.01866)
- [GaussianImage++ (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/37572)
- [LIG (AAAI 2025)](https://arxiv.org/abs/2502.09039)
- [Structure-Guided Allocation](https://arxiv.org/abs/2512.24018)
- [Soft Anisotropic Diagrams (SIGGRAPH 2026)](https://arxiv.org/abs/2604.21984)
- [SGI (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Pan_SGI_Structured_2D_Gaussians_for_Efficient_and_Compact_Large_Image_CVPR_2026_paper.html)
- [AIR](https://arxiv.org/abs/2605.20820)
- [P-GSVC](https://arxiv.org/abs/2603.10551)
- [WIPES (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_WIPES_Wavelet-based_Visual_Primitives_ICCV_2025_paper.html)
- [Contour-Aware 2DGS](https://arxiv.org/abs/2512.23255)
- [CGVQ](https://arxiv.org/abs/2607.05667)
- [Instant-GI (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Instant_GaussianImage_A_Generalizable_and_Self-Adaptive_Image_Representation_via_2D_ICCV_2025_paper.html)
- [EigenGS (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Tai_EigenGS_Representation_From_Eigenspace_to_Gaussian_Image_Space_CVPR_2025_paper.html)
- [Fast-2DGS (WACV 2026 workshop)](https://openaccess.thecvf.com/content/WACV2026W/WVAQ/html/Wang_Fast_2DGS_Efficient_Image_Representation_with_Deep_Gaussian_Prior_WACVW_2026_paper.html)
- [Faster-GS (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Hahlbohm_Faster-GS_Analyzing_and_Improving_Gaussian_Splatting_Optimization_CVPR_2026_paper.html)
- [GLIC (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Adaptive_Learned_Image_Compression_with_Graph_Neural_Networks_CVPR_2026_paper.html)
- [HPCM (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Learned_Image_Compression_with_Hierarchical_Progressive_Context_Modeling_ICCV_2025_paper.html)
- [NeuRBF (ICCV 2023)](https://arxiv.org/abs/2309.15426)
- [Practical learned image compression (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Tatwawadi_What_Matters_in_Practical_Learned_Image_Compression_CVPR_2026_paper.html)
- [Revising densification in Gaussian splatting](https://arxiv.org/abs/2404.06109)
- [Fritzke's growing cell structures/RBF allocation precursor (NeurIPS 1993)](https://proceedings.neurips.cc/paper_files/paper/1993/file/df7f28ac89ca37bf1abd2f6c184fe1cf-Paper.pdf)

The search was broad but not exhaustive. Several 2026 works are recent preprints; inaccessible
supplementary code, patents, theses, and terminology outside the query vocabulary remain novelty
risks. Reported paper numbers are descriptions of their native protocols, not locally reproduced
leaderboard rows.

## 1. Frontier map

### 1.1 Repository frontier

| Component | Live mechanism and evidence | Binding limitation |
|---|---|---|
| Representation | Finite anisotropic 2D Gaussians with position, RS covariance, opacity, constant RGB, and opt-in affine local RGB. | Constant color is weak on phase/texture; affine color is not supported by the native CUDA backward or SSPL1 codec. |
| Renderer | Normalized compact-support weighted sum plus additive/reference comparators; exact CUDA path exists. | Normalization couples overlapping primitives; exact backward performs overlap-dependent work and atomics. The tiled CUDA experiment was `1.69x` slower and `-0.1328 dB`. |
| Initialization | Ten live strategies including quadtree-WSE, tensor/edge WSE, gradient, SLIC-like and uniform controls; opt-in progressive WSE order preserves the terminal set. | BENCH-007 rejected tensor-WSE as an actual-rate method. WSE ordering is discarded by Morton codec order. |
| Fitting | AdamW/Adan variants, cosine schedule, losses, analytical color solve, growth/pruning/relocation, checkpoint selection, QAT. | Geometry and color remain ill-conditioned; small endpoint gains often reverse in recovery. The fair fitter-knob study is incomplete. |
| Convergence | Same-final-count checkpoint selection gains `+0.4884 dB` over 72 trajectories; a small pyramid repair gained `+1.0601 dB` final PSNR but is not confirmed. | Historical endpoints were not converged; progressive-capacity effects are confounded by iteration allocation. |
| Growth | Residual, support, ranked and moment-preserving controls are implemented. Signed matched residual gained `+0.5199 dB` immediately, then lost `-0.0318 dB` at 20 and `-0.2301 dB` at 100 recovery steps. | Site scores do not model normalized ownership or parameterization gauge; immediate improvement is a poor proxy for recoverability. |
| Expressiveness | Affine local color gained about `+1.06 dB` on a tiny two-image/two-budget smoke. | No fair large CUDA/codec evidence, and frequency-bearing competitors such as WIPES are stronger direct controls. |
| Compression | SSPL1 is a real self-contained stream: scalar attribute quantization, Morton/delta positions, zlib (optional PNG payload), and post-fit QAT. | No learned/context entropy model, no affine fields, no embedded prefix. PNG payload smoke was 2,756 versus 2,348 bytes for zlib. |
| Actual RD | BENCH-007 supplies equal-search, cold-decode, exact-cap actual-rate evidence. | The tensor-WSE gate failed; Stage 2, BENCH-008, and decoder-synchronized structural geometry are not authorized as rescue. |
| Learned encoder | A feed-forward predictor exists and runs, but remained `1.8563 dB` behind the hand quadtree-WSE prior in its bounded comparison. | No strong amortized structured predictor comparable to AIR/Instant-GI. |

Positive local signals are hypotheses, not a combined result. In particular, four-pair ABL-005
estimates (`opacity=constant +0.9082 dB`, cosine LR `+0.5757 dB`, moment-preserving split
`+0.2973 dB`) are correlated partial evidence and cannot be added arithmetically.

### 1.2 How the strongest external methods work

| Method | Representation and compositor | Allocation / optimization | Performance / compression mechanism | Protocol caveat |
|---|---|---|---|---|
| GaussianImage | Eight-parameter 2D Gaussian ellipses, direct RGB, unsorted accumulated summation. | Per-image gradient fitting without a learned decoder. | Attribute quantization including color VQ and entropy coding in its codec lane. | Additive accumulation is not StructSplat's normalized renderer; published no-entropy formulas and actual streams must stay separate. |
| Image-GS | Top-K normalized Gaussian mixture; only retained Gaussians receive gradients. | Gradient-plus-uniform position sampling, inverse-scale optimization, and periodic residual-pixel growth. | Tile CUDA, about 0.3K MACs/pixel, progressive LOD and block/shell organization for random access. | Its model-size BPP and LOD packaging are not automatically an entropy-coded self-contained stream. |
| GaussianImage++ | Direct covariance 2D splats with content-aware covariance filtering. | Sparse random start; every 5k iterations it creates splats at top residual pixels and prunes invalid/redundant ones. | LSQ+ QAT uses different precisions (reported 12/10/6 bits for position/covariance/color). | Authors explicitly report a remaining gap to learned codecs at high rate; a local covariance transplant failed badly (`-2.3587 dB` worst arm). |
| LIG | Millions of directly parameterized covariance entries plus a low-resolution base and normalized high-frequency residual level. | Hierarchical per-image optimization; invalid non-PSD Gaussians are discarded. | Targets very large images and representation efficiency. | It is not a complete competitive entropy codec; its multiscale/residual grammar directly occupies generic hierarchy claims. |
| Structure-Guided Allocation | Gaussian representation with SLIC/Sobel complexity classes. | Dynamic class allocation shifts from a structured 6:2:1-like rule toward uniform as budget grows; Sobel geometry consistency guides fitting. | Learns covariance precision and groups hard bitwidth metadata; reports tiny metadata overhead in its protocol. | Closest handcrafted allocation baseline, but its full coupled method is not reproduced by a local sampling analogue. |
| SAD | An anisotropic additively weighted distance, learnable radius and temperature, determinant-normalized metric, and softmax partition over a per-pixel top-K site set. | Gradient/uniform initialization; responsibility error-density split; exact renormalized removal-delta prune. | Temporally reused top-K map, jump-flood/local propagation plus global probes; tile-local hash gradient reduction; GPU-resident optimizer. | Its BPP is a fixed parameter accounting rather than an audited entropy stream. Quality and speed arise from representation/implementation co-design. |
| SGI | A seed defines a local region; two lightweight MLPs generate multiple local Gaussians from seed attributes. | Multi-scale coarse-to-fine seed fitting on very large images. | Binary hash-grid context probability model, quantizer refinement, and entropy coding at seed/attribute level. | Evaluated on roughly 27--76 MP images; benefits do not transfer to a tiny direct-field codec without the generator and entropy model. |
| AIR | A multi-stage network predicts residual Gaussian sets. | Predict--Optimize--Distill: briefly refine each predicted stage, then regress the predictor to those one-to-one Gaussian targets; joint fine-tuning follows. | Feed-forward per-image encoding and adaptive quantization. | Amortized training cost and dataset prior differ from training-free per-image optimization. |
| P-GSVC | Explicit base and enhancement Gaussian layers, each corresponding to quality/resolution prefixes. | Jointly supervises the final level and cyclically selected intermediate levels to prevent prefix drift. | Layered scalable representation; video adds inter-frame initialization/selection. | A progressive ordering alone is not this method; layer headers and all prefix bytes must count. |
| WIPES | Localized wavelet-like visual primitive: an oscillatory carrier under an anisotropic Gaussian envelope. | Per-image fitting with a custom differentiable splatter. | A frequency vector gives one primitive multiple lobes, increasing texture expressiveness at some rendering cost. | It directly occupies generic "Gaussian plus frequency" proposals and must be the primitive control. |
| Contour-Aware 2DGS | Gaussian renderer with segmentation-region constraints. | Warm-up then prevents cross-region contribution/mixing. | Reduces low-count boundary bleed by transmitting/using region structure. | Any boundary claim must count segmentation bits and compare a segmentation-free control. |
| Instant-GI / EigenGS / Fast-2DGS | Learned position/attribute predictors or domain eigenspaces initialize Gaussian fields. | Dataset-level amortized prediction, dithering/Delaunay features, eigencolor combinations, or a budget-conditioned deep prior; per-image refinement may follow. | Replaces much of expensive fitting with inference when many related images amortize training. | Training corpus, multi-hour training, checkpoint/memory, aligned-domain assumptions and inference hardware are part of the method. |
| Faster-GS | Preserves the Gaussian scene representation while reorganizing training work. | Per-Gaussian backward accumulation, fused activation/gradient/Adam work and periodic spatial reordering target atomics and optimizer overhead. | Reports up to about 5x faster 3DGS training with preserved visual quality in its native benchmarks. | It is a 3DGS implementation result; a 2D normalized renderer needs a fresh gradient oracle and end-to-end profile. |
| GLIC / HPCM and strong learned codecs | Learned analysis/synthesis transforms, graph or hierarchical progressive contexts, quantization and practical entropy coding. | Dataset training amortizes image statistics; rate-distortion losses optimize the actual codec family. | Rich decoded context sharply reduces symbol entropy; GLIC reports sizable BD-rate gains against VTM-9.1 in its native study. | This is the outside-class compression frontier with a much heavier learned decoder. Leading among Gaussian methods does not imply overall compression SOTA. |

The central lesson is causal rather than architectural: the largest frontier gains couple a new
state representation to the operator that exploits it. SAD's top-K speed needs explicit ownership
and a persistent neighborhood; SGI's compression needs generated structure plus an entropy model;
WIPES' quality comes from changing the atom; AIR's speed comes from amortization. Copying only a
loss term or initializer cannot reproduce those mechanisms.

One donor connection is already latent in StructSplat. With opacity, its Gaussian log weight is
`log w_i(x) = log o_i - 0.5 q_i(x)`: opacity is a per-site logit bias, the closest Gaussian
analogue of SAD's additive radius. A separate Gaussian “temperature” is largely absorbed by
covariance scale, whereas this bias is not. The partial ABL-005 opacity signal (`+0.9082 dB` over
four pairs) is therefore a plausible known-mechanism validation, not a new primitive result. It
still needs an equal-trainable-scalar and equal-actual-byte comparison because opacity adds a ninth
scalar to an eight-scalar constant-color Gaussian.

## 2. Functional problem signature

Domain-neutral formulation:

> From finite samples of a bounded signal, infer a compact, locally queryable latent program that
> allocates state, precision and evaluation work across the domain, synthesizes arbitrary queries,
> and transmits enough state to satisfy distortion, rate and latency constraints.

For a normalized StructSplat field, the key map is

`I_hat(x) = sum_i w_i(x) c_i(x) / (sum_i w_i(x) + eps)`,

where `w_i` combines compact-support anisotropic geometry and opacity. The trainable state mixes
continuous variables (position, scale, angle, opacity, RGB/slopes) with discrete choices (count,
split/prune events, ordering, bit depth, stream coding, candidate membership).

Important structure:

- **Local-to-global coupling:** every primitive is local, but normalization couples all overlapping
  contributors; moving one changes both numerator and denominator.
- **Symmetries:** primitive permutation is exact; splitting one weight into identical co-located
  weights whose sum is unchanged is also render-equivalent. Allocation rules need not respect
  those symmetries even when rendering does.
- **Non-identifiability:** opacity, support size, overlap and color can trade off; many fields render
  nearly the same pixels but have different gradients, support work and coded length.
- **State/work mismatch:** Gaussian count is neither actual bytes nor render work. Work depends on
  support visits/candidate selection; bytes depend on entropy, headers and side information.
- **Binding bottlenecks:** constant-color atom expressiveness, overlap-dependent CUDA backward,
  recovery after discrete growth, absence of a context entropy model, and loss of prefix order in
  the codec.

### Fixation anti-library

The following are excluded unless made precise by a new invariant and a direct control:

- “add attention,” “make it learned,” “make it adaptive,” or “make it multiscale”;
- more Gaussians, iterations, datasets, or hyperparameter search without a new hypothesis;
- routine L1/Charbonnier/SSIM/Sobel loss swaps;
- generic edge allocation, residual growth, covariance filtering, progressive order, boundary
  masks, frequency-bearing splats, clustered VQ, or seed-generated Gaussians—the direct territory
  is already occupied;
- reporting primitive count, `56N+1728`, float payload or checkpoint size as actual bpp;
- a structural decoder whose seed, mask, network, codebook or base layer is not counted;
- combining partial positive knobs and claiming their separate deltas add;
- optimizing immediate post-split PSNR and calling it convergence;
- calling a StructSplat analogue by an external method's name;
- reopening BENCH-007's failed images or protected Stage 2 with post-hoc variants.

## 3. Independent idea generation

Ideas were generated in separate lanes before scoring. This matters because otherwise the recent
SAD/SGI/WIPES vocabulary makes every proposal look like a minor variant of those papers.

### Productive lane, before audit

1. responsibility-normalized densification;
2. exact removal-delta prune followed by count-neutral reallocation;
3. persistent top-K contributor maps with an explicit discarded-mass certificate;
4. per-Gaussian backward accumulation with fused optimizer work;
5. jointly trained WSE prefixes preserved in a real embedded stream;
6. affine local color plus structure-conditioned entropy coding.

### Assumption-surgery lane, before audit

1. require all allocation decisions to respect render-equivalent opacity splitting;
2. replace “a primitive is the budget unit” with divisible ownership mass;
3. replace independent fields with tile-local charts that must agree on overlaps;
4. replace count constraints with exact marginal codelength constraints;
5. replace residual magnitude with recovery sensitivity as the growth target;
6. permit typed atoms chosen by local description length rather than a Gaussian-only dictionary.

### Primitive/grammar lane, before audit

1. a **quotient field**, whose element is an equivalence class of render-identical split/merged
   splat parameterizations;
2. an **ownership transport plan**, moving continuous budget mass between responsibility cells;
3. a **chart-and-glue program**, containing tile-local fields plus an explicitly coded transition
   operator;
4. a **defect-charge graph**, measuring mismatches between ownership adjacency and target
   discontinuity topology;
5. a typed local atom program choosing constant, affine, compensated/dipole or wavelet atoms;
6. a codelength-priced birth/death operator using cold-stream marginal bytes.

### New-evidence lane, before audit

1. opacity-split gauge counterexamples for every growth/prune score;
2. top-K tail-mass and quality phase diagrams across count, content and support density;
3. perturb--recover experiments separating immediate improvement from optimizer compatibility;
4. exact-signal atom suites (step edges, corners, chirps, lattices and smooth ramps) with ground
   truth approximation rates;
5. decoded-byte attribution and marginal-RD traces for every discrete field operation.

The audit removed “responsibility density,” “top-K,” “frequency splats,” “seed structure,” and
“joint progressive layers” as standalone novelty claims. They remain useful controls or donor
mechanisms.

## 4. Productive recombinations

### P1 — SAD responsibility-density transplant

**Central claim:** Under StructSplat's normalized renderer, responsibility error density with
`alpha=0.7` selects moment-preserving split parents that recover faster than existing residual and
raw-support site rules.

**Null hypothesis:** At equal start, split, count and optimizer, post-20 and post-100 PSNR are no
better than the strongest existing site rule.

**Novelty class:** N1/N2-T implementation transfer; not a new research mechanism.

**Known foundation:** SAD equations 15--16 already use responsibility mass, weighted error and
`E_i / max(m_i, eps)^alpha`, with `alpha=0.7`; responsibility-weighted densification also has a
nearer 3D Gaussian lineage in Rota Bulò et al.

**Irreducible delta:** A renderer-semantic transplant and causal `alpha=1` versus `0.7` diagnostic
inside a compact-support normalized Gaussian field.

**Why this is not merely A + B:** It does not pass that test. The donor already supplies the
mechanism; only its behavior under a different renderer and split operator is unknown.

**Changed grammar or transfer mechanism:** Replace kernel support error with normalized ownership
error per owned mass.

**New prediction:** If overlap-density bias is material, `alpha=0.7` should beat both raw support
and center residual after recovery, not merely immediately after splitting.

**Cheapest killing test:** The frozen FIT-018 four-arm, eight-pair shared-start CPU guard.

**Prior-art threats:** SAD is dispositive against novelty; Image-GS and GaussianImage++ occupy
residual growth; 3D Gaussian densification propagates pixel errors by alpha responsibility.

**Novelty confidence:** 0--5% as a standalone contribution; 85--95% confidence that the transplant
comparison itself was absent from this repository as of the cutoff.

**Scientific value:** Calibrates whether explicit responsibility is a useful causal bridge and
whether SAD's sublinear mass exponent survives a change in representation.

**Publishable if successful:** Only as an ablation inside a larger genuinely new method.
**Publishable if partially successful:** A renderer-dependent mechanism note.
**Publishable if it fails informatively:** Yes, as evidence that SAD's allocation gain does not
factor from its representation/split design.

### P2 — removal-delta count-neutral reallocation

**Central claim:** Pruning the primitive with the smallest exact renormalized removal loss and
placing a replacement at the largest under-served ownership region improves fixed-count recovery
over opacity pruning plus residual birth.

**Null hypothesis:** Exact deletion value does not improve post-recovery quality or stability at
equal count and work.

**Novelty class:** N2-T at most.

**Known foundation:** SAD derives exact removal delta for a normalized partition; split/prune and
birth/death moves are classical in mixtures and adaptive approximation.

**Irreducible delta:** Pair an exact normalized-Gaussian deletion oracle with a strictly
count-neutral replacement and measure the full deletion/birth interaction.

**Why this is not merely A + B:** The paired intervention could expose a conservation law between
owned mass removed and new mass allocated, but without that constraint it collapses to known
prune-plus-grow.

**Changed grammar or transfer mechanism:** The atomic action is a coupled transport of one budget
slot, not independent thresholded pruning and growth.

**New prediction:** At equal N, it should reduce catastrophic deletion tails and improve the lower
quantile of pairwise post-recovery PSNR even when the mean gain is small.

**Cheapest killing test:** Dense-oracle deletion deltas for N=64 on four fixtures, followed by one
64-to-64 relocate and 20/100 recovery steps.

**Prior-art threats:** SAD removal delta; mixture-model split/merge; 3DGS pruning/relocation;
optimal-transport resampling.

**Novelty confidence:** 15--30%; exact combination and prediction may be unreported, while all
components are known.

**Scientific value:** Tests whether deletion error, rather than opacity/activity, is the missing
half of allocation.

**Publishable if successful:** As part of gauge-aware budget control.
**Publishable if partially successful:** If it isolates deletion stability but not final quality.
**Publishable if it fails informatively:** It would show that discrete replacement, not site
valuation, dominates recovery.

### P3 — certified bounded-contributor renderer

**Central claim:** Most pixels can be rendered and differentiated with at most K=16 contributors
while a tail-mass certificate identifies the small set requiring fallback, cutting support work
without a material quality loss.

**Null hypothesis:** Candidate selection plus fallback costs erase work savings, or K needed for a
quality-safe tail is too large/dense.

**Novelty class:** N2-T systems recombination; top-K itself is known.

**Known foundation:** Image-GS renders a top-K normalized mixture; SAD maintains persistent top-K
site maps; StructSplat already has compact support and a tile index.

**Irreducible delta:** A certificate based on retained denominator mass, coupled to an exact
compact-support fallback and audited against the CUDA oracle.

**Why this is not merely A + B:** A hard fixed K is exactly known. The only surviving delta is an
adaptive correctness/work contract that decides when the compact-support renderer may truncate.

**Changed grammar or transfer mechanism:** Evaluation becomes `bounded fast path + certified
fallback`, rather than unconditional enumeration of every support overlap.

**New prediction:** At a fixed PSNR-loss ceiling, retained K correlates more strongly with
responsibility entropy than raw active count, enabling lower work on low-entropy pixels.

**Cheapest killing test:** Benchmark-only exact top-K reconstructions for K={1,2,4,8,16}, report
retained mass, full-vs-truncated PSNR and support-evaluation ceiling, then repeat across fields.

**Prior-art threats:** SAD candidate propagation, Image-GS top-K, approximate mixture evaluation,
tile-based splatting and error-bounded kernel summation.

**Novelty confidence:** 10--25% for the full systems composition; 0% for top-K truncation.

**Scientific value:** Converts vague “fewer overlaps” into a quality/work phase diagram and an
implementable performance contract.

**Publishable if successful:** With a real kernel, backward parity, end-to-end speed and broad
fields.
**Publishable if partially successful:** The phase diagram can still explain when fixed-K methods
win.
**Publishable if it fails informatively:** Yes; it locates selection/index overhead or heavy tails
as the blocker.

### P4 — per-Gaussian backward and fused optimizer

**Central claim:** Reversing normalized-renderer backward traversal so one block accumulates one
Gaussian's local gradients and writes once, then fusing activation gradients and Adam, materially
reduces fit time without changing pixels or optimization trajectories beyond numerical tolerance.

**Null hypothesis:** Re-reading local pixels and denominator state costs more than the atomics saved,
or the normalized quotient gradient prevents useful fusion.

**Novelty class:** N1/N2-T implementation transfer.

**Known foundation:** Faster-GS identifies backward atomics and optimizer traffic as dominant in
3DGS and combines per-Gaussian backward, fused gradients/Adam and periodic spatial ordering.

**Irreducible delta:** Derive and validate the one-write-per-Gaussian backward for StructSplat's 2D
normalized quotient, using cached rendered color and denominator under compact support.

**Why this is not merely A + B:** The performance mechanism is directly inherited. The only new
technical content is making it correct and worthwhile under a normalized 2D renderer.

**Changed grammar or transfer mechanism:** Backward ownership changes from pixel threads scattering
atomics to primitive blocks gathering their support.

**New prediction:** Speedup should grow with average positive support contributions per Gaussian
and atomic contention, while low-overlap fields may regress because pixel/denominator reads
dominate.

**Cheapest killing test:** Profile the exact CUDA path by kernel and atomic count; implement a
forward-compatible reference gather backward for colors/opacity first, require gradient parity,
then benchmark one dense-overlap and one sparse-overlap field.

**Prior-art threats:** Faster-GS is direct; per-primitive rasterizer backward and fused optimizers
are established GPU techniques.

**Novelty confidence:** 0--10% as a method, 70--85% that this exact normalized-2D implementation
has not yet been tested in StructSplat.

**Scientific value:** It targets the measured bottleneck without sacrificing renderer semantics or
quality and provides a stronger native baseline before approximate top-K work.

**Publishable if successful:** As systems evidence only with broad GPU/field scaling.
**Publishable if partially successful:** If profiling establishes an overlap-density phase change.
**Publishable if it fails informatively:** It shows candidate reduction or a representation change
is necessary because gather bandwidth dominates.

### P5 — jointly optimized, actually embedded prefixes

**Central claim:** Cyclic prefix supervision plus an order-preserving SSPL stream improves
prefix-rate distortion without degrading the full field at the same actual bytes.

**Null hypothesis:** Prefix constraints reduce full-quality efficiency or stream metadata/order
destroys the geometric prefix benefit.

**Novelty class:** N2; direct method territory is occupied by P-GSVC.

**Known foundation:** P-GSVC jointly trains layered prefixes; StructSplat has terminal-set-
preserving WSE order and pyramid prefix metrics.

**Irreducible delta:** A controlled test of whether WSE's geometric prefix survives joint image
loss and a self-contained entropy stream.

**Why this is not merely A + B:** Without an actual embedded stream it is merely WSE plus P-GSVC.
The only defensible delta is preserving a fixed terminal set/order while measuring cold-decoded
prefix bytes and quality.

**Changed grammar or transfer mechanism:** Prefixes become first-class decoded programs optimized
under multiple truncation points.

**New prediction:** Joint prefix loss improves worst-prefix PSNR more than it harms full-field PSNR,
but only if the codec retains prefix order rather than Morton-sorting the full field.

**Cheapest killing test:** N=256 on four safe fixtures, prefix N={32,64,128,256}, cyclic versus
final-only loss, with a minimal order-preserving SSPL prototype and every header byte counted.

**Prior-art threats:** P-GSVC, scalable/layered coding, embedded wavelet bitstreams, progressive
point-cloud codecs.

**Novelty confidence:** 10--25%; likely an engineering recombination.

**Scientific value:** Connects the confirmed geometric prefix property to the only compression
question that matters: decoded prefix RD.

**Publishable if successful:** As a strong embedded-codec result with conventional controls.
**Publishable if partially successful:** If it identifies the quality/order tradeoff.
**Publishable if it fails informatively:** It closes the progressive-WSE compression narrative.

## 5. Exploratory candidates

### E1 — opacity-split gauge audit

**Central claim:** Current allocation scores make different decisions for render-identical fields
created by splitting one primitive's opacity mass, and that non-invariance predicts unstable
growth/recovery.

**Null hypothesis:** Rankings and recovery are unchanged, or ranking changes have no causal effect.

**Novelty class:** N2 evidence/concept candidate.

**Known foundation:** Mixture non-identifiability, permutation symmetry and component splitting are
well known; alpha=1 responsibility means have a relevant scale property.

**Irreducible delta:** An explicit render-equivalence transformation and allocation-invariance
test for normalized splat fields.

**Why this is not merely A + B:** It yields a new falsifiable invariant requirement rather than
adding a component.

**Changed grammar or transfer mechanism:** Evaluate operators on render-equivalence classes, not
raw parameter rows.

**New prediction:** Alpha=1 per-child responsibility density is invariant to equal mass splitting,
whereas alpha=0.7 scales each child's score by `2^(alpha-1)`; grouped scores should predict recovery
better than either raw ranking.

**Cheapest killing test:** Construct exact split-equivalent fields, verify pixel parity, compare all
site rankings and then one controlled growth event.

**Prior-art threats:** Identifiability-aware mixture learning, split/merge MCMC, gauge symmetry in
neural networks, component aggregation in adaptive mixtures.

**Novelty confidence:** 30--50% for this exact diagnostic; 5--15% for the general symmetry idea.

**Scientific value:** Can explain why a good score fails after discrete refinement.

**Publishable if successful:** As the empirical foundation of a quotient-space allocator.
**Publishable if partially successful:** A diagnostic paper/section.
**Publishable if it fails informatively:** It rules out gauge sensitivity as the recovery cause.

### E2 — perturb--recover response spectroscopy

**Central claim:** The best growth intervention is the one whose perturbation aligns with slow
recoverable modes of the fitted field, not the one with the largest immediate PSNR gain.

**Null hypothesis:** Local response/curvature diagnostics do not predict 20/100-step recovery.

**Novelty class:** N2-T measurement transfer.

**Known foundation:** Linear response, influence functions, Hessian-vector products and optimizer
stability analysis; FIT-017 already supplies a striking immediate-versus-recovery reversal.

**Irreducible delta:** Use standardized split perturbations as probes and predict long recovery
from short trajectory geometry.

**Why this is not merely A + B:** It transfers a measurement operator, not a named optimizer, and
creates observations the current benchmark does not contain.

**Changed grammar or transfer mechanism:** Candidate interventions are system-identification
impulses; response trajectories, not one endpoint, become the evidence.

**New prediction:** Signed matched-residual splits have a large immediate projection but excite
high-curvature/poorly conditioned modes, predicting their observed reversal.

**Cheapest killing test:** Reuse FIT-017/FIT-018 shared starts, log loss and parameter displacement
for the first 20 steps, estimate directional curvature with finite differences, and regress only
across images (not seeds as independent samples).

**Prior-art threats:** Influence functions, neural tangent/linearization analysis, loss-landscape
diagnostics and adaptive mesh recovery estimators.

**Novelty confidence:** 25--45% for the splat-growth application and prediction.

**Scientific value:** A negative method result can still yield a reusable intervention-screening
diagnostic.

**Publishable if successful:** With cross-method prediction on held-out interventions.
**Publishable if partially successful:** If it explains only catastrophic reversals.
**Publishable if it fails informatively:** It shows nonlinear discrete effects dominate local
response.

**Resolved by FIT-020:** the ranked-deduplication treatment produced ample late variation, but the
preregistered bend did not improve held-out prediction or branch selection. This kills the exact
one-bend operationalization, not all recovery-aware management or learning-curve prediction.

### E3 — typed atom approximation suite

**Central claim:** A small disclosed synthetic suite can identify which signal classes require
Gaussian, affine, compensated/dipole or oscillatory atoms before expensive natural-image runs.

**Null hypothesis:** Synthetic approximation slopes do not predict natural-image texture/edge
behavior at matched bytes and support work.

**Novelty class:** N1/N2 evidence program.

**Known foundation:** Nonlinear approximation theory, sparse coding, wavelets/Gabor dictionaries,
WIPES and StructSplat affine colors.

**Irreducible delta:** A renderer- and byte-accounted bridge from exact signal grammar to observed
image failure modes.

**Why this is not merely A + B:** It is not a new atom; value comes from a discriminating evidence
program and preregistered transfer prediction.

**Changed grammar or transfer mechanism:** Compare approximation rate against N, scalars, bytes and
support evaluations—not only final PSNR.

**New prediction:** Affine color should dominate constant splats on ramps/edges but not chirps;
WIPES-like carriers should reverse that ordering on narrow-band textures.

**Cheapest killing test:** Fit exact 64x64 ramps, step edges, corners, sinusoids and chirps for
N={16,32,64}; include every atom parameter in an analytical and actual prototype rate.

**Prior-art threats:** Classical approximation benchmarks and WIPES ablations.

**Novelty confidence:** 10--25%; intended primarily as new evidence.

**Scientific value:** Prevents a costly mixed-primitive implementation without a distinct regime.

**Publishable if successful:** As theory/evidence supporting a new grammar.
**Publishable if partially successful:** A useful method-selection diagnostic.
**Publishable if it fails informatively:** It argues for optimizer/codec work over atom changes.

### E4 — marginal cold-stream RD attribution

**Central claim:** Actual marginal bytes and distortion for birth, split, affine-upgrade and
precision-upgrade actions differ enough from count proxies to change the selected operation.

**Null hypothesis:** Count/analytical-bit ranking agrees with cold-stream marginal RD, or entropy
context makes marginal attribution too unstable.

**Novelty class:** N2 measurement/optimization candidate.

**Known foundation:** Lagrangian rate-distortion optimization, MDL, entropy-constrained vector
quantization and codec ablations.

**Irreducible delta:** Apply each local field edit, fully re-encode, and use the resulting byte
delta as its causal price rather than a surrogate.

**Why this is not merely A + B:** It becomes more than routine RDO only if the counterfactual byte
oracle changes allocation decisions and yields a reusable local price model.

**Changed grammar or transfer mechanism:** The budget unit is a stream edit with context-dependent
codelength, not a Gaussian.

**New prediction:** Affine-upgrade actions win on smooth gradients, while births win near localized
detail; Morton/context interactions make their prices spatially heterogeneous.

**Cheapest killing test:** On N=64 fields, enumerate 16 candidate edits, cold-encode each SSPL1
counterfactual, and compare proxy versus exact marginal-RD rankings.

**Prior-art threats:** Classical codec RDO, learned entropy-model latent allocation, MDL model
selection and rate-aware Gaussian pruning.

**Novelty confidence:** 15--30%; likely known in principle, possibly informative in this explicit
field setting.

**Scientific value:** Directly attacks the repository's count-versus-rate mismatch.

**Publishable if successful:** As the oracle and training target for a rate-aware allocator.
**Publishable if partially successful:** An attribution study.
**Publishable if it fails informatively:** It validates cheaper proxies and avoids a complex loop.

**Resolved 2026-07-15:** COMP-006 scoped the executable test to standard birth, matched
birth-for-death replacement, and the complete SSPL1 precision envelope because affine and
frequency-bearing syntax does not exist in codec v1. Exact and raw-bit oracles agreed on only
14/36 rows, but broad action class agreed in 34/36. Birth lost `-1.0714 dB` to the strongest
control with a wholly negative family-bootstrap interval. Keep the exact-byte oracle; close the
standard-birth allocator claim. Richer atoms remain a new grammar task, not an E4 rescue.

## 6. Transformational candidates

These are deliberately not implementation recommendations. Each first needs its killing test and
a deeper prior-art audit.

### T1 — quotient field with gauge-invariant operators

**Central claim:** Treating a normalized splat field as an equivalence class under permutation and
opacity-mass split/merge yields allocation and regularization operators that are more stable than
row-wise operators.

**Null hypothesis:** Quotient-aware decisions do not improve ranking stability, recovery or
compression after controlling for compute.

**Novelty class:** N3 candidate; the formulation changes the state space, while the underlying
symmetry is known.

**Known foundation:** Quotient parameter spaces, finite-mixture identifiability, gauge symmetry,
optimal transport between mixtures, component split/merge.

**Irreducible delta:** Define the field state as `[theta]` under render-preserving mass
refinements, then require birth/death/regularization to commute with those refinements.

**Why this is not merely A + B:** Subtracting mixture symmetry and ordinary regularization still
leaves a new admissibility condition on every field operator: `A(refine(theta))` must project to
the same rendered action as `A(theta)`.

**Changed grammar or transfer mechanism:** Rows cease to be physical budget units; equivalence
classes and ownership-mass groups are the primitive objects.

**New prediction:** Two pixel-identical fields with different opacity splits should choose the
same spatial region and equivalent total mass for the next refinement; current raw/support/SAD
alpha-0.7 row scores need not.

**Cheapest killing test:** E1's exact gauge counterexample plus a grouped-score prototype. Abandon
if existing scores are already stable or grouped decisions do not improve recovery on a disjoint
synthetic set.

**Prior-art threats:** Identifiable mixture parameterizations, quotient geometry of mixtures,
measure-valued optimization, Wasserstein gradient flows, split/merge invariant point processes.

**Novelty confidence:** 20--40% pending dedicated mixture/measure-optimization and patent search.

**Scientific value:** Supplies a principled explanation for duplicate/split instability and a
testable design law for allocation.

**Publishable if successful:** A representation/operator theory plus empirical recovery result.
**Publishable if partially successful:** A useful invariant diagnostic.
**Publishable if it fails informatively:** It shows row non-identifiability is benign in practice.

### T2 — unbalanced ownership transport

**Central claim:** Model refinement as a regularized transport plan that moves divisible ownership
mass from over-served cells to under-served residual regions, allowing creation/destruction only at
an explicit codelength price.

**Null hypothesis:** Transport reduces to known relocate/split heuristics or is too expensive and
no more recoverable.

**Novelty class:** N3-T formulation transfer.

**Known foundation:** Unbalanced optimal transport, capacity-constrained centroidal Voronoi
tessellations, particle resampling, measure-valued sparse approximation.

**Irreducible delta:** A joint plan over donor ownership mass, recipient residual demand and exact
rate price, followed by a discrete projection back to splats.

**Why this is not merely A + B:** Adding OT to residual maps is cosmetic unless the plan conserves
or prices mass and predicts coupled donor/recipient moves. That coupled action is the remaining
delta.

**Changed grammar or transfer mechanism:** Budget is continuous mass during allocation and becomes
discrete only during a controlled projection.

**New prediction:** On two images with equal residual histograms but different spatial donor--
recipient distances, transport should choose different reallocations and reduce large parameter
jumps relative to prune-plus-birth.

**Cheapest killing test:** N=32 synthetic two-region fields; solve a small entropic unbalanced OT
problem from responsibility mass to residual demand; compare post-20 recovery with matched
count-neutral relocation.

**Prior-art threats:** CVT/Lloyd relocation, optimal quantization, differentiable particle filters,
Wasserstein mixture reduction, optimal-transport mesh adaptation.

**Novelty confidence:** 15--35%; the cross-domain formulation may be occupied under mesh/mixture
terminology.

**Scientific value:** Unifies pruning, relocation and growth under a conservation/pricing law.

**Publishable if successful:** With scaling approximation and broad actual-RD evidence.
**Publishable if partially successful:** If the transport diagnostic predicts which heuristic
wins.
**Publishable if it fails informatively:** It establishes that discrete projection or local
optimization, not allocation distance, is binding.

### T3 — sheaf-consistent local Gaussian charts

**Central claim:** Fit tile-local fields in parallel and transmit a sparse overlap/gluing operator
so local reconstructions agree, reducing global optimizer and renderer coupling without seam loss.

**Null hypothesis:** Gluing metadata/optimization costs exceed parallel gains or seams require a
global field equivalent to the baseline.

**Novelty class:** N3-T formulation transfer.

**Known foundation:** Domain decomposition, Schwarz methods, finite-element mortar methods,
partition of unity, sheaf consistency, tiled neural fields and block random-access codecs.

**Irreducible delta:** A decoded object comprising local normalized-splat charts plus an explicit,
testable overlap-consistency map; neither independent tiles nor one global field can express that
state directly.

**Why this is not merely A + B:** “Tile the image and blend” is known. The surviving delta is to
make compatibility residuals and transition metadata first-class optimized/coded objects.

**Changed grammar or transfer mechanism:** Global state is assembled from local sections that are
valid only when their restrictions agree on overlaps.

**New prediction:** For fixed overlap width, fitting time should scale nearly with the largest tile
rather than image area while seam error is bounded by the measured compatibility residual.

**Cheapest killing test:** Four overlapping 64x64 charts on a 112x112 synthetic edge/corner image;
parallel local fits, one sparse linear glue solve, exact seam and byte accounting.

**Prior-art threats:** Tiled INR/texture codecs, block 2DGS, domain-decomposed RBF approximation,
partition-of-unity methods, neural sheaf fields.

**Novelty confidence:** 10--30%; high terminology and historical-obviousness risk.

**Scientific value:** Offers a different path to scale/random access than a faster global kernel.

**Publishable if successful:** With a principled seam bound and large-image scaling.
**Publishable if partially successful:** A parallel fitting method with explicit seam tradeoff.
**Publishable if it fails informatively:** It quantifies why normalized local fields resist
decomposition.

### T4 — ownership-adjacency defect charges

**Central claim:** Discrete mismatches between the target's discontinuity graph and the field's
ownership-adjacency graph predict hard reconstruction errors that residual magnitude alone misses.

**Null hypothesis:** Defect charge adds no predictive power over residual, gradient and
responsibility entropy.

**Novelty class:** N3-T diagnostic/operator candidate.

**Known foundation:** Topological defects in materials, graph discrepancy, mesh-quality
indicators, contour-aware rendering, Voronoi/Delaunay adjacency.

**Irreducible delta:** A signed local charge counting missing, spurious or misoriented adjacency
events between target discontinuities and soft ownership cells, used to trigger split/rotate/glue
actions.

**Why this is not merely A + B:** Edge-guided sampling is known. A defect must be invariant to edge
magnitude scaling and predict a specific topological edit; otherwise the idea collapses to another
edge score.

**Changed grammar or transfer mechanism:** Errors are local failures of adjacency/type, not only
scalar pixel residuals.

**New prediction:** Two patches with matched Sobel/residual histograms but different junction
topology will receive different edits, with the largest benefit at T-junctions and thin gaps.

**Cheapest killing test:** Synthetic step, crossing, T-junction and one-pixel gap suite; build hard
dominant-owner adjacency, define charge without tuning natural images, and test whether it predicts
the best of split/rotate/add actions.

**Prior-art threats:** Contour-Aware 2DGS, topology-aware mesh adaptation, segmentation graph
losses, medial-axis/Voronoi image representations, Euler characteristic losses.

**Novelty confidence:** 20--45%; the exact operator is unsearched in several adjacent literatures.

**Scientific value:** Could identify a failure class that MSE, Sobel and mass statistics conflate.

**Publishable if successful:** As a new diagnostic plus targeted allocator.
**Publishable if partially successful:** If it only predicts junction failures.
**Publishable if it fails informatively:** It falsifies topology as an independent bottleneck.

## 7. Cross-domain transfers

### Transfer portfolio summary

| Candidate | Donor field | Preserved causal mechanism | Broken correspondences (at least three) | Native competitor | Adoption barrier / enabling change | Transfer class |
|---|---|---|---|---|---|---|
| Ownership transport | Unbalanced optimal transport | Move scarce mass from supply to demand while pricing creation/destruction. | Pixels are not conserved particles; splats have geometry/color; projection is discrete; entropy price is non-metric. | Relocate/prune/grow, CVT | OT was too costly; small entropic solvers and GPU Sinkhorn make a killing test cheap. | N3-T |
| Certified top-K | Database incremental top-K/query maintenance | Reuse a candidate set and update only entries whose bound can change the answer. | Scores move during fitting; queries are dense pixels; discarded weights affect a normalized denominator; GPU divergence matters. | Tile enumeration, SAD/Image-GS | Dynamic GPU state and proof overhead; persistent tile buffers and tail-mass bounds now exist. | N2-T |
| Local charts | Sheaves/domain decomposition | Solve local sections independently and enforce overlap compatibility. | Image boundaries are learned; chart state is nonunique; normalization crosses overlaps; metadata must be coded. | Global field, tiled INR | Seam handling and side bits; compact support and random-access demand create a reason to pay them. | N3-T |
| Defect charge | Materials/topological defect theory | Localize global structural failure as discrete charge whose annihilation suggests an edit. | No physical lattice; target edges are noisy; charge definition is not conserved; color has no material analogue. | Sobel/segmentation allocation | Risk of metaphor-only transfer; ownership adjacency and synthetic topology tests make it measurable. | N3-T |
| Response spectroscopy | Experimental design/system identification | Apply standardized impulses and infer hidden dynamics from the response. | Optimizer is nonlinear/time-varying; interventions change dimension; seeds are repeated measures; Hessian is huge. | Endpoint ablation | Logging/finite differences add cost; existing shared-start recovery harness provides controlled impulses. | N2-T measurement |
| Embedded prefixes | Rateless/scalable coding | Optimize nested partial messages so every prefix is a useful reconstruction. | Splat rows have unequal/contextual bit cost; Morton changes order; gradient updates couple prefixes; decoder needs headers. | P-GSVC, scalable codecs | Codec discards WSE order; SSPL already offers a real stream to extend. | N2-T |
| Slot interference graph | Compiler register allocation | Assign scarce slots using an interference graph so simultaneously active values do not collide. | Splats may overlap beneficially; slots carry geometry not registers; activity changes continuously; spills mean approximation error. | Count pruning/clustering | Graph build may dominate; tile support index supplies the interference graph nearly for free. | N2-T exploratory |

The first, third and fourth transfers are from fields rarely connected directly to per-image
Gaussian splatting. Response spectroscopy is deliberately a measurement transfer rather than a
method transplant.

### Transfer map A — unbalanced OT to ownership allocation

**Domain-neutral functional problem:** Reallocate a scarce divisible resource from low-value supply
to spatially distributed unmet demand with movement and creation/destruction costs.

| Structural role | Donor: unbalanced OT | Recipient: StructSplat |
|---|---|---|
| State | source/target measures | responsibility mass and residual-demand measure |
| Observation | mass density and ground cost | per-pixel ownership, error, spatial/codelength cost |
| Action/operator | transport, create, destroy | relocate/split/prune/project splats |
| Objective/energy | transport + marginal divergence | predicted distortion + movement + exact/proxy rate |
| Invariant | mass balance when balanced | explicit budget conservation in the balanced limit |
| Noise/uncertainty | empirical measure error | stochastic fitting and residual noise |
| Boundary condition | bounded transport domain | image boundary and compact support |
| Failure mode | diffuse or costly plan | projection destroys the continuous optimum |

**Preserved causal mechanism:** Joint donor/recipient choice prevents a locally attractive birth
from ignoring where capacity can be removed cheaply.

**Broken correspondences:** Pixels are not conserved material (scientifically productive);
transported mass must become anisotropic colored atoms (correctable by projection); actual entropy
cost is context-dependent and non-metric (scientifically productive); global Sinkhorn work may be
fatal at scale.

**Required invention:** A local/sparse ground graph, rate-aware unbalanced penalty and stable
projection from transported mass to valid splats.

**Recipient-specific prediction:** Distance-separated donor/demand layouts with identical residual
histograms should yield different actions and recovery.

**Native competitor:** Exact removal-delta relocate plus residual birth.
**Adoption barrier:** OT solve and discrete projection cost.
**Enabling change:** Sparse tile adjacency and small entropic GPU solvers.
**Transfer novelty:** donor 0/5; correspondence 2/5; adaptation 3/5; prediction 4/5.

### Transfer map B — incremental top-K queries to rendering

**Domain-neutral functional problem:** Maintain a small answer set for many repeated queries while
the scored items move slowly, with a certificate that omitted items cannot materially alter the
answer.

| Structural role | Donor: database query processing | Recipient: renderer/fitter |
|---|---|---|
| State | indexed items and cached result set | tile candidates and per-pixel top contributors |
| Observation | score/bound changes | geometry/opacity changes and retained denominator mass |
| Action/operator | incremental repair, bounded top-K | propagate/recheck candidates, exact fallback |
| Objective/energy | query latency under exactness/recall | support work under pixel-error bound |
| Invariant | answer correct under bounds | render matches oracle within declared tolerance |
| Noise/uncertainty | stale scores | floating point and changing supports |
| Boundary condition | query/index partition | tiles and clipped support boxes |
| Failure mode | churn invalidates cache | high-entropy overlap forces large K/fallback |

**Preserved causal mechanism:** Temporal coherence makes repair cheaper than rebuilding/enumerating.

**Broken correspondences:** Weight rankings change continuously (correctable); omitted items alter
the denominator even when individually small (scientifically productive); millions of GPU queries
penalize branching (correctable); approximate recall is not automatically an image-error bound
(scientifically productive).

**Required invention:** Cheap per-pixel/tile tail bound, churn trigger and fused repair/fallback.

**Recipient-specific prediction:** Candidate churn, not N, should predict iteration cost; entropy
should predict the K required for a fixed error.

**Native competitor:** Rebuilt compact-support tile index and unconditional support enumeration.
**Adoption barrier:** Candidate-state memory and divergent fallback.
**Enabling change:** Existing tile index plus observed K=16 tail behavior.
**Transfer novelty:** donor 0/5; correspondence 2/5; adaptation 2/5; prediction 3/5.

### Transfer map C — sheaf/domain decomposition to local fields

**Domain-neutral functional problem:** Infer a global object from independently optimized local
sections whose overlap restrictions must be compatible.

| Structural role | Donor: sheaves/domain decomposition | Recipient: image field |
|---|---|---|
| State | local sections and transition/restriction maps | tile splat fields and overlap glue |
| Observation | boundary residuals | pixel/color/denominator disagreement in overlap |
| Action/operator | local solve plus compatibility update | parallel fit plus sparse glue solve |
| Objective/energy | local energy + interface penalty | distortion + seam + coded metadata |
| Invariant | compatible sections define global solution | overlap reconstructions agree within tolerance |
| Noise/uncertainty | discretization/interface error | optimization and quantization error |
| Boundary condition | subdomain interface | image/tile boundaries |
| Failure mode | slow global interface mode | seams or glue state as large as global coupling |

**Preserved causal mechanism:** Most computation is local; a much smaller interface problem carries
the global dependency.

**Broken correspondences:** Splat parameterizations on neighboring charts are non-identifiable
(scientifically productive); normalized denominators do not decompose linearly (scientifically
productive); image edges may coincide with interfaces unpredictably (correctable); glue bytes may
be fatal for compression.

**Required invention:** A parameterization-independent overlap observable and sparse coded
transition operator.

**Recipient-specific prediction:** Wall time approaches the slowest tile plus interface solve, and
seam error tracks compatibility residual independently of interior PSNR.

**Native competitor:** One global compact-support field; independent blended tiles.
**Adoption barrier:** Seam correctness and metadata.
**Enabling change:** Compact support, random-access use cases and multi-GPU/large-image frontier.
**Transfer novelty:** donor 0/5; correspondence 3/5; adaptation 3/5; prediction 3/5.

### Transfer map D — defect theory to ownership adjacency

**Domain-neutral functional problem:** Detect sparse structural mismatches in a locally ordered
system and choose edits that remove them rather than minimizing only a smooth energy.

| Structural role | Donor: material/topological defects | Recipient: ownership diagram |
|---|---|---|
| State | lattice/order parameter and defects | dominant-owner adjacency and target edge graph |
| Observation | winding/coordination mismatch | missing/spurious/misoriented adjacency events |
| Action/operator | move, create or annihilate defect | split, rotate, glue or add a primitive |
| Objective/energy | elastic energy + defect cost | distortion + defect charge + rate/work |
| Invariant | topological charge in a closed region | candidate graph invariant under color/edge scaling |
| Noise/uncertainty | thermal/disorder noise | texture and edge-detector noise |
| Boundary condition | material boundary | image boundary/segmentation-free edge ends |
| Failure mode | false defects in disorder | texture produces meaningless graph charge |

**Preserved causal mechanism:** A sparse discrete descriptor distinguishes states with similar
scalar energy but different repair requirements.

**Broken correspondences:** No physical conservation law (scientifically productive); target edge
graph is estimated (correctable); ownership is soft and changes under optimization (correctable);
texture may create dense “defects” (possibly fatal).

**Required invention:** A stable soft-to-graph map and an edit-specific signed charge.

**Recipient-specific prediction:** Junction/gap images with matched residual statistics require
different operations, predicted by charge.

**Native competitor:** Sobel residual allocation and Contour-Aware 2DGS.
**Adoption barrier:** High metaphor risk and noisy topology.
**Enabling change:** Explicit ownership diagnostics and exact synthetic topologies.
**Transfer novelty:** donor 0/5; correspondence 3/5; adaptation 4/5; prediction 4/5.

### Transfer map E — system identification to refinement diagnostics

**Preserved causal mechanism:** Controlled impulses reveal hidden modes that passive endpoint
observation cannot distinguish. State is the fitted field/optimizer; observation is loss and
parameter response; action is a standardized split; objective is prediction of recovery. Broken
correspondences are changing dimensionality, nonlinear Adam state, correlated seed replays and
expensive curvature; all are correctable for a small screen. The required invention is an
intervention-normalized response descriptor. The recipient-specific prediction is the FIT-017
immediate-gain/recovery-reversal ordering. The native competitor is simply running all candidates
to 100 steps. The adoption barrier is diagnostic overhead; shared-start harnesses are the enabling
change. Transfer novelty: donor 0/5, correspondence 3/5, adaptation 3/5, prediction 4/5.

### Transfer map F — rateless/scalable coding to Gaussian prefixes

**Preserved causal mechanism:** Every transmitted prefix must be independently useful, so training
optimizes a family of truncations. State is an ordered stream/layers; observation is prefix RD;
action is cyclic prefix supervision and entropy coding. Broken correspondences are unequal row
length, entropy context across truncation, Morton reordering and renderer coupling. The required
invention is a prefix-stable context model and order-preserving stream. The recipient prediction is
that joint supervision improves worst-prefix RD only when physical bytes retain order. P-GSVC and
ordinary scalable codecs are native competitors. The adoption barrier is redesigning SSPL1;
confirmed WSE prefixes and a validated codec are the enabling changes. Transfer novelty: donor
0/5, correspondence 1/5, adaptation 2/5, prediction 3/5.

## 8. New-evidence discovery programs

### Program D1 — render-equivalence/gauge suite

**Question:** Which field operators respect transformations that leave every rendered pixel
unchanged?

**Construction:** Starting from a fitted field, generate exact or numerically certified equivalents
by permuting rows and replacing one primitive weight `w` with two co-located copies `rho*w` and
`(1-rho)*w` with identical geometry/color. Sweep `rho`, overlap and background coverage. Verify
the render before evaluating any operator.

**Measurements:** pixel max error; score/rank Kendall agreement after projecting child scores back
to the parent group; selected spatial region; immediate/post-20/post-100 recovery; actual bytes and
support work.

**Distinct prediction:** Renderer-equivalent inputs expose rank changes for row-wise scores;
group/quotient operators remain invariant and yield lower recovery variance.

**Killing result:** If rank changes do not alter the selected region/recovery on preregistered
synthetics, quotient machinery is unnecessary. This program survives the audit because it can
falsify the central T1 mechanism cheaply.

### Program D2 — contributor-tail phase diagram

**Question:** When does a normalized compact-support field actually need its full overlap set?

**Construction:** Across image classes, N, initialization, opacity, support radius and fit horizon,
render the exact field and cumulative top-K fields for K={1,2,4,8,16,32}. Distinguish clipped
rectangle visits, numerically positive weights and retained responsibility mass.

**Measurements:** PSNR/MS-SSIM/LPIPS versus full and target; mass quantiles; active/effective count;
entropy; upper-bound work removed; measured selection/index/forward/backward time for any real
kernel.

**Distinct prediction:** Entropy and tail mass, not N alone, determine the safe K and the benefit of
candidate caching.

**Killing result:** If K=32 still fails the quality tolerance on a material fraction of pixels or
selection cost consumes the support-work ceiling, stop the bounded-contributor lineage. The first
single-field probe is reported below but is not enough to answer the program.

### Program D3 — intervention response atlas

**Question:** Can early response dynamics predict whether a discrete field edit will recover?

**Construction:** Apply standardized moment split, signed residual split, relocate, removal/birth
and gauge-grouped split to identical fitted checkpoints. Replay a fresh optimizer from each branch,
logging dense early time points and 20/100 endpoints.

**Measurements:** immediate distortion projection; directional finite-difference curvature;
gradient/step alignment; parameter displacement; time/iteration AUC; image-clustered uncertainty.

**Distinct prediction:** Large immediate gains that excite high-curvature, weakly aligned modes
reverse under recovery; a short response descriptor predicts those reversals across intervention
types.

**Killing result:** Leave-one-image-out prediction no better than ranking by immediate gain.
FIT-020 reached that boundary for one ranked-deduplication intervention: reversals are real, but
the frozen bend is no better than early/static controls and changes no held-out action. Do not
expand this atlas on the exposed targets. Reopening requires a materially different mechanism,
disjoint targets, cross-intervention validation, and optimizer-state-preserving controls.

### Program D4 — exact grammar approximation and rate suite

**Question:** Is the binding limitation the atom, optimizer, or codec?

**Construction:** Exact synthetic families (constant, ramp, step, corner, disk, sinusoid, chirp,
checker and narrow line) at multiple frequencies/orientations. Fit constant Gaussian, affine
Gaussian and a direct WIPES-like control; only add a new atom after these controls.

**Measurements:** error slope versus N, trainable scalar count, analytical payload, cold-stream
bytes, support evaluations and time; recovery from matched initialization.

**Distinct prediction:** Each atom family has a characteristic approximation regime; natural-image
texture gains should agree with its synthetic regime after matching actual bytes.

**Killing result:** No repeatable regime or synthetic ordering fails to predict a disjoint natural
texture set. The program is evidence-first and survives even if no new primitive does.

## 9. Adversarial prior-art audit

### Reconstruction and facet audit

| Candidate | Can one work reconstruct it? | Combination reconstruction | Surviving novelty facets | Audit label |
|---|---|---|---|---|
| P1 responsibility density | Yes: SAD Eq. 15--16, including exponent 0.7. | No combination needed. | Local experiment only. | Likely known / direct transfer |
| P2 removal/reallocate | SAD gives removal delta; standard relocate supplies replacement. | SAD + mixture relocation. | Coupled count-neutral conservation prediction. | Known components, possibly new relationship |
| P3 certified top-K | SAD/Image-GS give fixed top-K. | Top-K + error-bounded summation/fallback. | Compact-support tail certificate and phase prediction. | Known components, possibly new systems relationship |
| P4 per-Gaussian backward | Faster-GS directly supplies the performance mechanism. | 3D mechanism + normalized quotient derivative. | Recipient implementation/evidence only. | Likely known / direct transfer |
| P5 actual embedded WSE prefixes | P-GSVC gives joint layers. | P-GSVC + WSE order + ordinary embedded stream. | Terminal-set-preserving order under actual prefix bytes. | Known components, possibly new combination |
| E1/T1 gauge audit/quotient field | Mixture identifiability gives the symmetry, not the full allocator. | Quotient geometry + adaptive allocation. | Formal operator equivariance, counterexample and recovery prediction. | Apparently unexplored under stated search; high threat |
| E2 response spectroscopy | System identification/influence functions give tools. | Intervention ablation + curvature diagnostics. | Predicting discrete splat-growth recovery reversal. | Known components, possibly new correspondence/prediction |
| E3 atom suite | Sparse approximation suites are classical. | WIPES + synthetic approximation. | Renderer/byte/work-accounted transfer evidence. | New evidence, not method novelty |
| E4 marginal stream RD | Classical RDO/MDL gives the principle. | Counterfactual SSPL encode + local edit enumeration. | Causal byte attribution in explicit fields. | Known components, possibly new evidence |
| T2 ownership transport | OT mesh/measure methods are close. | Unbalanced OT + responsibility demand + rate projection. | Joint donor/recipient/codelength prediction. | Apparently unexplored under stated search; high threat |
| T3 sheaf charts | Partition-of-unity/domain decomposition is very close. | Tiled fields + interface coding. | Parameterization-independent gluing object and seam bound. | Insufficient evidence; likely crowded |
| T4 defect charge | Topology-aware mesh/segmentation methods threaten it. | Ownership graph + target edge graph + defect theory. | Edge-scale-invariant edit-specific charge/prediction. | Apparently unexplored under stated search; very uncertain |

Facet notation for the three most novel-looking candidates:

| Candidate | Problem | Representation | Mechanism | Theory | Experiment | Combination | Correspondence | Prediction |
|---|---|---|---|---|---|---|---|---|
| T1 quotient field | known | potentially new | potentially new | potentially new | new locally | known parts | new-looking | distinct |
| T2 ownership OT | known | new formulation | known donor/new projection | potential | new | known parts | new-looking | distinct |
| T4 defect charge | known | potentially new diagnostic | uncertain | potential | new | known parts | new-looking | distinct |

### Adversarial idea tests

| Candidate | A+B / subtraction result | Grammar and prediction test | Necessity / compression test | Decision |
|---|---|---|---|---|
| P1 | Fails: subtract SAD and nothing methodological remains. | Prediction remains useful only as transfer evidence. | Not necessary and no compression claim. | Run bounded calibration; no novelty claim. |
| P3 | Fixed K subtracts away; certificate remains. | Fast-path/fallback grammar and entropy prediction survive. | Necessary only if selection overhead stays below saved work. | Expand phase diagram before kernel. |
| P5 | Most parts subtract to P-GSVC/scalable coding. | Actual order-preserving stream is measurable. | Must beat ordinary layer order at equal cold bytes. | Medium priority. |
| T1 | Quotient-space allocator remains after subtracting known symmetry. | New admissible operators and split-equivalence prediction. | Grouping overhead must be small; no rate claim yet. | Highest-information cheap novelty test. |
| T2 | “OT on residuals” fails; coupled donor/recipient mass law survives. | Continuous allocation grammar and distance-dependent prediction survive. | Must beat native relocate at equal solve time/count. | High-risk follow-up after T1. |
| T3 | “Tiled blend” subtracts away; coded glue remains. | Local-section grammar survives. | Glue bytes/time may make it unnecessary. | Park pending small seam test. |
| T4 | Generic edge allocation subtracts away only if charge is edge-scale invariant and edit-specific. | Junction/gap prediction is distinct. | Must beat Sobel/segmentation-free graph controls. | Exploratory synthetic only. |

For the transfers, removing donor terminology still leaves a recipient operation for OT, database
top-K, domain decomposition, response spectroscopy and scalable coding. The defect transfer is the
weakest under the terminology-removal and historical-obviousness tests; it stays only because its
synthetic counterexample is cheap. None should be called transformational until its homomorphism,
native-baseline and counter-analogy tests pass empirically.

## 10. Pareto frontier

Scores are 0 (weak) to 5 (strong). `First-test cost` is scored high when cheap. These are separate
judgments, not an averaged leaderboard.

| Candidate | Apparent novelty | Falsifiability | Explanatory value | Importance | Feasibility | First-test cost | Interpretability | Baseline strength | Informative failure | Publication potential |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 responsibility transfer | 0 | 5 | 3 | 2 | 5 | 5 | 5 | 5 | 4 | 1 |
| P2 removal/reallocate | 2 | 5 | 4 | 3 | 4 | 4 | 4 | 5 | 4 | 2 |
| P3 certified top-K | 1 | 5 | 4 | 5 | 4 | 4 | 5 | 5 | 5 | 3 |
| P4 per-Gaussian backward | 0 | 5 | 3 | 5 | 3 | 3 | 5 | 5 | 4 | 2 |
| P5 embedded prefixes | 1 | 5 | 3 | 4 | 3 | 2 | 4 | 5 | 4 | 2 |
| E1 gauge audit | 3 | 5 | 5 | 4 | 5 | 5 | 5 | 3 | 5 | 4 |
| E2 response spectroscopy | 3 | 4 | 5 | 4 | 4 | 4 | 3 | 4 | 5 | 4 |
| E3 atom suite | 2 | 5 | 4 | 3 | 4 | 5 | 5 | 5 | 4 | 3 |
| E4 marginal stream RD | 2 | 5 | 4 | 5 | 3 | 4 | 5 | 4 | 5 | 3 |
| T1 quotient field | 4 | 5 | 5 | 4 | 3 | 5 | 4 | 3 | 5 | 5 |
| T2 ownership OT | 4 | 5 | 5 | 4 | 3 | 4 | 4 | 4 | 5 | 5 |
| T3 local charts | 4 | 4 | 5 | 5 | 2 | 3 | 3 | 4 | 5 | 5 |
| T4 defect charge | 4 | 4 | 5 | 3 | 3 | 5 | 3 | 5 | 5 | 4 |

The Pareto set is E1/T1 (novelty information per cost), P3/P4 (performance relevance with strong
native baselines), E4 (direct compression relevance), E2 (explains recovery), and T2 (high-risk
unification). P1 is selected only as an already-started donor calibration, not because it is on the
novelty Pareto frontier.

## 11. Executed experiments

### 11.1 Experiment A — responsibility-density transfer calibration

**Why it ran:** This was selected before the independent audit located SAD's exact formula. The
audit downgraded novelty, but the frozen run remains useful because the user explicitly asked
whether donor ideas help this repository. It is now a mechanism replication, not the recommended
novelty experiment.

**Preregistration:** [`tasks/FIT-018-responsibility-error-density.md`](../../tasks/FIT-018-responsibility-error-density.md)

**Claim:** `alpha=0.7` responsibility-density splitting improves recovery over the stronger of
current residual/support site rules.

**Null hypothesis:** No `+0.10 dB` mean post-20 advantage, fewer than 6/8 positive pairs, more than
`0.05 dB` post-100 regression, more than 15% total-time overhead, or invalid/unequal counts.

**Frozen design:** Four COCO repository fixtures, seeds {0,1}, max-side 64; one shared N=64
quadtree-WSE field fitted for 40 steps; identical moment-preserving 64-to-80 splits; arms residual,
support, responsibility alpha=1 and responsibility alpha=0.7; measure immediate/post-20/post-100.
These reused fixtures are a smoke only, not publication/default evidence.

**Measured result:** All 32 arm rows completed at exact N=80 with finite scores/renders.

| Site score | Immediate PSNR | Post-20 PSNR | Post-100 PSNR | Mean scorer s | Mean total-100 s |
|---|---:|---:|---:|---:|---:|
| residual | 20.3105 | 21.5835 | 23.0925 | 0.000096 | 0.9962 |
| support | 20.3093 | 21.6255 | 23.2022 | 0.003097 | 0.9939 |
| responsibility alpha=1 | 20.3210 | 21.6274 | 23.2444 | 0.005861 | 0.9815 |
| responsibility alpha=0.7 | 20.2470 | 21.6057 | 23.1611 | 0.005718 | 1.0122 |

The stronger existing arm was support. Relative to it, alpha=0.7 was `-0.0623 dB` immediately,
`-0.0198 dB` post-20, positive on 4/8 pairs, and `-0.0411 dB` post-100; total-100 overhead was
`+1.8%`. It failed the `+0.10 dB` and 6/8 post-20 gates while passing the post-100, timing, count
and finite-value gates.

Alpha=1 was nearly the support rule's recovery twin: `+0.0019 dB` post-20 and `+0.0422 dB`
post-100, with mean parent Jaccard 0.740 versus support. SAD's sublinear exponent changed the
selected set more strongly (Jaccard 0.510) and hurt the frozen recovery comparison. This is
consistent with, but does not prove, the opacity-split gauge analysis: alpha 1 is invariant per
child while alpha 0.7 penalizes fragmentation by `2^-0.3 ~= 0.812`.

**Decision:** **Guard failed.** Stop this exact alpha-0.7 lineage; keep the implementation only as
an opt-in causal control. Do not tune alpha or residual norm on these fixtures. The alpha-1
diagnostic motivates E1/T1 on a disjoint exact-equivalence suite but does not authorize a natural-
image confirmation.

The final run enforced one CPU thread and PyTorch deterministic algorithms. A second source-frozen
replay matched every non-timing aggregate exactly. Each input and relevant source file is hashed;
combined relevant-source SHA-256 is
`32035c6988e66c3ec8a0c9a088433ab4a0833a66c2d5adc6a95dcd66b67d992b`.

### 11.2 Experiment B — top-K responsibility oracle

**Artifact:** `results/topk_responsibility_oracle_2026-07-15/`
**Benchmark:** [`benchmarks/topk_responsibility_oracle.py`](../../benchmarks/topk_responsibility_oracle.py)
**Focused test:** [`tests/test_topk_responsibility_oracle.py`](../../tests/test_topk_responsibility_oracle.py)

The only suitable saved non-ARA field was an older N=20,000, 640x480 field. The benchmark exactly
reconstructs the normalized renderer's rounded/clipped support rectangles, distinguishes rectangle
evaluations from numerically positive float32 weights, selects per-pixel top-K weights, and
validates the full tilewise reconstruction against the exact CUDA renderer.

Full quality was 34.62571 dB / 0.994055 MS-SSIM. Full-oracle versus CUDA maximum absolute error was
`1.34e-5`, below the frozen `2e-5` parity threshold.

| K | Target PSNR | Delta versus full | MS-SSIM | Mean retained mass | p05 retained mass | Ideal reduction versus positive contributions |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.61907 | -20.00664 dB | 0.760053 | 37.00% | 18.04% | 96.74% |
| 2 | 21.10620 | -13.51950 dB | 0.897496 | 58.60% | 33.42% | 93.47% |
| 4 | 28.33474 | -6.29097 dB | 0.973430 | 80.76% | 56.77% | 86.95% |
| 8 | 33.43626 | -1.18945 dB | 0.992198 | 95.16% | 82.65% | 73.94% |
| 16 | 34.57605 | -0.04966 dB | 0.994010 | 99.61% | 97.71% | 49.59% |

There were 12,328,179 clipped-rectangle contributions and 9,413,898 numerically positive weights;
the audit materialized 28,881,152 tile-candidate/pixel pairs. K=16 is the only near-lossless
survivor on this field. K<=8 is killed. At K=16, a hypothetical free winner oracle could remove
49.59% of positive contributions (61.50% of rectangle visits).

That percentage is **not a speedup**. This implementation first evaluates every candidate weight
and then selects the winners. A real result requires persistent/bounded selection, backward, memory
and end-to-end timing. Given Faster-GS, exact per-Gaussian backward should be profiled before an
approximate top-K kernel.

Reproduction:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
PYTHONPATH=src \
TORCH_EXTENSIONS_DIR=/tmp/structsplat_torch_extensions \
python -m benchmarks.topk_responsibility_oracle \
  --image tests/test_images/COCO_train2014_000000000009.jpg \
  --field results/coco4_current_20k_500/COCO_train2014_000000000009_aniso_flanking_20k_500.npz \
  --outdir results/topk_responsibility_oracle_2026-07-15 \
  --device cuda --validation-renderer cuda \
  --ks 1 2 4 8 16 --tile-size 16 \
  --sigma-cutoff 3.0 --aa-dilation 0.0 --parity-max-abs 2e-5
```

The artifact records field/image hashes, environment, dirty-source status, per-source hashes and
combined relevant-source SHA-256
`f4f69da4ede522f1eb61654d524da7f9f897c0b77c9d257b776253a24f70e6e5`.

### 11.3 Experiment C — opacity-gauge allocation

**Detailed report:** [`2026-07-15-opacity-gauge-experiment.md`](2026-07-15-opacity-gauge-experiment.md)
**Preregistration:** [`tasks/FIT-019-opacity-gauge-equivalence.md`](../../tasks/FIT-019-opacity-gauge-equivalence.md)
**Artifact:** `results/fit019_opacity_gauge_guard_v2_2026-07-15/`

FIT-019 executed the recommended E1/T1 killing experiment on eight disjoint 48x48 procedural
families x seeds {0,1}. Exact half-opacity refinements preserved rendering within `8.345e-7` but
changed raw alpha-1 physical-group multisets on both seeds for all 8/8 families. Aggregate-first
scores matched within `2.701e-6` relative error at alpha 0.7 and `2.747e-6` at alpha 1, with
canonical/gauge top-8 equality in all 16 checkpoints at both alphas. The commutation mechanism is
therefore confirmed.

Recovery utility is rejected. Quotient alpha 1 versus raw gauge-row alpha 1 was `+0.2111 dB` at
post-20 but won only 5/8 target families and became `-0.6007 dB` at post-100. It was `-0.0665 dB`
against canonical support at post-20. These fail the target-breadth, late-retention and support-floor
gates. All immediate/recovered counts were N=40 and finite; timing stayed inside the guard but
varied in sign across replay and is not a speed result.

An adversarial review found that v1 had logged but not gated several preregistered invariants. The
audit-corrected v2 changed only gate/provenance implementation. All 4,352 shared non-timing v1/v2
row comparisons are exact, and an untouched-source v2 replay matches every deterministic payload.
The artifact carries a verified 24-file source snapshot with combined SHA-256
`89f52281e5596e7225cf278be74eeaabc423c54db2f176ecf4d5bfa5d2b99f23`.

**Decision:** keep the exact quotient as a correctness oracle only. Do not add production lineage
metadata, approximate sibling groups, defaults, or a natural-image confirmation. ADR-0014 records
that boundary.

### 11.4 Experiment D — ranked deduplication perturb--recover assay

**Detailed report:** [`2026-07-15-perturb-recover-spectroscopy.md`](2026-07-15-perturb-recover-spectroscopy.md)
**Preregistration:** [`tasks/FIT-020-perturb-recover-spectroscopy.md`](../../tasks/FIT-020-perturb-recover-spectroscopy.md)
**Artifact:** `results/fit020_response_spectroscopy_v1_2026-07-15/`

FIT-020 executed E2 as a narrower, fail-closed killing test. Six procedural families x six fixed
variants were divided into four training and two held-out variants; three seeds were averaged
before modeling. Four arms applied eight births to the same checkpoint and moved along a ranked
ticket-deduplication path from C5 through C8. The benchmark collected 432 complete 200-step
trajectories and compared a static intervention model, a step-10 early model, and the same model
plus one preregistered bend.

The signal guard passed strongly (`SD(y)=3.2529 dB`; 35/36 held-out cells had
`|y| >= 0.10 dB`), but the response model failed. Its RMSE was `2.9641 dB` versus `2.9616 dB` for
the early baseline, sign accuracy was the same `69.44%`, response improved only 2/6 families, and
bias was `-1.0455 dB`. Both models selected C5 on all 12 held-out targets and had `1.1116 dB`
regret, worse than observed-step-10 selection at `0.7669 dB`. The decision is **stop**.

Concentrated fixed arms had positive descriptive late means versus C8, with C6 strongest, but the
effect is dominated by sinusoid/chirp targets and was not a preregistered policy claim. It cannot
rescue the descriptor. The primary and measurement-equivalent post-writer-fix replay agree exactly
on every compared non-timing measurement field after excluding source provenance, as well as every
paired row, the normalized aggregate, manifest, and decision. ADR-0015 keeps the predictor and
selector benchmark-only.

### 11.5 Experiment E — marginal cold-stream RD attribution

**Detailed report:** [`2026-07-15-marginal-cold-stream-rd.md`](2026-07-15-marginal-cold-stream-rd.md)
**Preregistration:** [`tasks/COMP-006-marginal-cold-stream-rd.md`](../../tasks/COMP-006-marginal-cold-stream-rd.md)
**Artifact:** `results/comp006_marginal_rd_dev_v1_2026-07-15/`

COMP-006 executed E4 as a frozen complete-stream killing test. Eighteen disjoint development
targets from six procedural families, with seeds `{0,1}` treated as repeated measurements, each
produced one persisted/cold-decoded N=64 parent. The benchmark evaluated 16 standard births, the
same 16 candidate rows in fixed-donor count-neutral replacements, no-edit, and an exhaustive 875-
mix global precision envelope at recovery steps 0 and 20. All headers, ranges, framing, attribute
streams, and zlib/Morton context entered the integer byte cap.

Both the primary and exact same-source replay completed 36 cells and 33,840 cold streams. At the
preregistered step-20, +16-byte cap, birth lost `-1.0714 dB` mean and `-0.9533 dB` median paired
PSNR to the strongest control. The family-bootstrap 95% interval was
`[-1.2873, -0.8417] dB`; every one of 18 target means and all six family means were negative.
Precision was the strongest control in 23/36 cells and replacement in 13/36. The decision is
**stop**, and the odd-variant confirmation split remains unscored.

The exact-rate infrastructure remains useful but narrower than the proposed allocator story.
Actual and raw-bit oracles selected the same row in only 14/36 cells, and actual selection gained
`+0.2131 dB` mean PSNR over the proxy oracle. All 22 disagreements concerned control allocation;
broad action class agreed in 34/36, and exact bytes never changed structural-birth selection. The
dominant precision pattern traded means bits for color/scale/rotation bits. Birth itself improved
no-edit by `+0.9267 dB`, but the control improved it by `+1.9982 dB`: opportunity cost, not absent
residual signal, killed the claim. ADR-0016 keeps the oracle benchmark-only.

## 12. Recommended work after COMP-006

E1, E2, and the executable standard-birth part of E4 are resolved. Do not tune their groups, bend,
horizon, cap, donor, candidate bank, bit box, or exposed procedural targets. Exact rate should stay
as a gate, but standard residual birth is no longer the next compression lever.

Two independent experiments now have the best information value:

1. **Performance — exact per-Gaussian backward.** Profile the normalized CUDA backward by overlap
   density and primitive count, implement per-Gaussian accumulation/fused optimizer work only
   behind forward/backward gradient oracles, then measure end-to-end fit time, memory, and quality.
   Faster-GS supplies the donor mechanism; this repository still needs its own normalized-renderer
   evidence. Do not mix in approximate top-K until exact work is localized.
2. **Compression/expressiveness — real equal-byte richer-atom grammar.** Add a versioned codec
   experiment for compact luminance slope/affine color and a WIPES-like carrier. Count syntax,
   quantizers, entropy state, and decode work; compare against standard birth, replacement, and
   per-mix-QAT precision under complete-stream caps on new targets. A proxy payload is prohibited.

Within the existing grammar, attribute/rate co-design is the lower-risk codec direction: per-group
learnable quantization, real Morton-context/range coding, and `R + lambda D` training.
GaussianImage++, HAC, SGI, HPCM, and GLIC show why the entropy model and decoded structure must be
co-designed. Any such method must still pass COMP-006's cold-stream oracle and an independent
natural-image confirmation; it cannot inherit a claim from the post-hoc precision diagnostic.

## 13. Requested-axis decision table

| Axis | New evidence in this study | Decision |
|---|---|---|
| Quality | COMP-006 birth improves no-edit by `+0.9267 dB`, but the strongest control improves by `+1.9982 dB`; birth loses by `-1.0714 dB` and every target mean is negative. | No promoted quality improvement; one-more-standard-row is not the next lever. |
| Convergence | Twenty fresh-QAT steps narrow the birth deficit by about `0.52 dB` but do not reverse it; only steps 0/20 were tested. | No convergence improvement. A new study needs optimizer-state controls, longer curves, and disjoint targets. |
| Performance | K=16 exposes a 49.59% positive-contribution oracle ceiling on one field; COMP-006 CPU timing is an exhaustive encoding audit. | Promising work bound, **no measured speedup**. Profile exact backward independently. |
| Compression | Exact bytes improve oracle selection by `+0.2131 dB` over raw bits, but only within control allocation; birth loses in all families and confirmation is sealed. | Keep exact-RD infrastructure; stop this standard-birth formulation. Test attribute/entropy co-design as a new task. |
| Expressiveness | COMP-006 uses only constant-RGB standard Gaussians because SSPL1 has no richer syntax. | Still untested. Build a real codec for compact slope/affine and WIPES controls, then compare equal complete bytes/work. |

## 14. Audit limitations

- The responsibility experiment reuses four repository fixtures and two repeated seeds; it is a
  mechanism smoke, not independent-image publication evidence.
- The top-K result is one older flanking-strategy field. It cannot establish a universal K or any
  speedup and may not represent current high-quality fields.
- FIT-019 uses procedural targets and fresh Adam restarts. It establishes exact allocator
  commutation and bounded restart-recovery behavior, not production optimizer-state continuation
  or natural-image quality. Its response correlations are descriptive and strongly influenced by
  target composition.
- FIT-020 also uses procedural targets and fresh Adam. It rejects only one frozen bend on one
  ranked-deduplication path. It does not refute recovery-aware Gaussian management or learning-
  curve prediction broadly. C5/C6 quality and AUC means are post-hoc, family-sensitive
  descriptions; all arms are N=40, and no byte or production timing claim is available.
- COMP-006 uses 18 small procedural development targets, N=64, one +16-byte primary cap, fresh
  Adam, standard no-opacity constant-RGB rows, and SSPL1/zlib. It rejects that operational birth
  formulation, not structural compression, natural-image densification, learned entropy models,
  long-horizon continuation, or richer atoms. Its complete-container deltas are non-additive and
  cannot be treated as patch costs or sequential local prices.
- Recent SAD, structure-guided, contour-aware, AIR and CGVQ results are author-reported preprints;
  some code/protocols remain unavailable or independently unchecked.
- Paper BPP values mix parameter accounting, learned/native streams and actual entropy-coded bytes.
  This report never treats them as one leaderboard.
- The outside-class learned-codec comparison has different decoder complexity, training data and
  objectives. It establishes the compression ceiling, not a common-method ablation.
- The repository documentation had stale descriptions (five versus ten initializers, old optimizer
  and CUDA status, and unresolved-rate wording). Live code/tasks took precedence; this study does
  not silently rewrite unrelated historical documents.
- The novelty search may miss patents, theses, non-English terminology and very recent concurrent
  work. T1--T4 require dedicated searches before any publication claim.
- No protected validation set was consumed and no failed BENCH-007 lineage was reopened.
