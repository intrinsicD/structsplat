# Codec-native continuous Gaussian observation: research portfolio

Status: exploratory portfolio; CORE-016 selects one default-off killing pilot. This document maps
the design space and evidence program. It does not claim novelty or state of the art.

## Functional signature and objective

The target system is not merely image compression. It is a cold, self-contained observation
program usable by realtime-gs:

```text
(encoded image bytes, optional alpha, canvas/crop transform)
    -> packet bytes
    -> continuous query q(x,y) = (RGB, alpha, structural density)
    -> bounded 2D proposals + multiview 2D-to-3D lift
    -> realtime 3D Gaussian renderer
```

The rate is the complete packet size. Quality has at least four domains: decoded pixel centers,
off-grid continuous queries, structural proposal adequacy, and downstream 3D render quality.
Performance includes encode, cold decode, query throughput, lift convergence, and final render
cost. No single row count or PSNR substitutes for this vector.

A useful formal objective is a constrained rate-distortion-compute problem:

```text
min_P  B(P) + lambda_p D_pixel(P) + lambda_c D_continuous(P)
             + lambda_3 D_downstream(P) + lambda_t T_cold/query/lift(P)
subject to exact alpha/crop semantics and a bounded decoder.
```

`D_continuous` is unidentifiable from one raster without an explicit sampling model or additional
views. Development therefore uses interpolation controls and ringing guards, not “ground-truth
continuous image” language.

## Frontier map and prior-art threats

| Frontier | What it contributes | Threat to a novelty claim | Remaining systems question |
|---|---|---|---|
| GaussianImage / Image-GS-style explicit image Gaussians | Optimized differentiable 2D primitives | Direct prior art for image-to-Gaussian fitting | Can cold query and downstream lift improve without storing appearance rows? |
| SGI structured 2D Gaussians | Seed-generated structured parameter sharing and compact large-image representation | Direct threat to structured allocation/grammar claims | How does a structured image representation behave as a realtime-gs teacher? |
| GSICO | Maps Gaussian attributes into structured images for conventional coding | Direct threat to “use an image codec for Gaussian data” | Can the conventional payload be the appearance authority rather than an attribute container? |
| Structure-guided Gaussian allocation | Feature-aware allocation of Gaussian capacity | Direct threat to tensor/edge-guided placement claims | Can structural allocation be rate-decoupled from appearance fidelity? |
| Gaussian RBF/cardinal interpolation | Stable Gaussian-kernel interpolation and coefficient prefiltering | Direct threat to interpolation novelty | What finite, bounded decoder gives an acceptable ringing/throughput tradeoff? |
| Conventional JPEG/WebP/AVIF/JXL and learned image codecs | Mature transform/prediction/entropy coding | Any broad image-compression claim is presumptively weak | Can their decoded samples drive continuous Gaussian queries with no hidden side channel? |
| Multiresolution splines, wavelets, Laplacian pyramids | Coarse-to-fine residual coding | Direct threat to hierarchy/residual novelty | Which basis exposes useful physical proposals to 3D lifting? |

The defensible contribution class is N1/N2 systems recombination and boundary design: a charged
codec-backed appearance program, a separate sparse structural measure, and a verified realtime-gs
adapter. Algorithmic novelty remains unestablished.

## Anti-library: measured failures to avoid

- One stored Gaussian per pixel is exact but duplicates mature raster storage and produces the
  worst structural count.
- Local contraction with average-SSE acceptance can pass global metrics while leaving grid, block,
  hole, or ring artifacts.
- Fixed-scale WSE elimination retains features but creates support gaps once survivor spacing
  exceeds the original kernel reach.
- Retained quadtree ancestors consume capacity without guaranteeing high-frequency correction.
- Optimizing only touched rows underfits the transition; optimizing a uniform 3x3 halo helps at one
  count and redistributes error at another.
- Full-raster coefficient storage disguised as a Gaussian count or uncoded NPZ byte proxy is not a
  compression result.
- A structural field with colors sampled only at its centers is not an appearance-complete teacher.

## Candidate generation

### Recombination candidates

1. **Field V2 + variable projection + actual codec.** Solve signed appearance coefficients for
   fixed structural geometry, then rate-distort the coefficient blocks with a real entropy coder.
2. **Short amortized encoder + bounded correction.** Predict rows and coefficients in one pass,
   then run only a few rate-aware variable-projection/geometry steps.
3. **Regional Lagrangian allocator.** Spend explicit bytes among texture blocks, edges, flat
   regions, alpha boundary, and structural proposals using measured marginal distortion per byte.
4. **SGI-style deterministic seed grammar + residual raster.** Generate most geometry from compact
   seeds and conventionally code only a signed residual appearance plane.
5. **Codec-native dual plane.** Let a conventional raster payload own appearance queries and use a
   separate sparse structural measure only for lifting proposals.
6. **Cardinal-prefiltered dual plane.** Add a deterministic Gaussian interpolation prefilter to the
   fifth candidate so wider overlap can replay pixel centers.

### Assumption surgery

1. Replace “a Gaussian row must own color” with “the observation must answer color queries.”
2. Replace “appearance fidelity and proposal density share one rate knob” with two independently
   priced planes.
3. Replace “fewer rows means more compression” with complete cold packet bytes.
4. Replace “optimizer iterations define convergence” with time-to-downstream-quality, allowing a
   zero-optimizer decoder.
5. Replace “the raster is discrete ground truth” with an explicit off-grid sampling assumption and
   a separate ringing guard.
6. Replace “each view owns all structure” with future shared multiview geometry plus per-view
   appearance deltas.

### New primitive and grammar candidates

1. **Gaussian lifting frame:** a deterministic dense appearance basis whose coefficients are
   decoded rather than optimized.
2. **Structural quotient measure:** a sparse nonnegative measure representing only proposal
   importance and anisotropy, not RGB reconstruction.
3. **Query packet IR:** a bounded grammar compiling payloads and metadata into a query backend.
4. **Seed-to-cluster grammar:** compact seeds expand into local proposal constellations with
   predictable query/index occupancy.
5. **Formal dual-plane operator:** `Q(x)=A(x;C,K,alpha)` and `S(x)=sum_i m_i G_i(x)`, where `C` may
   be signed, `m_i >= 0`, and no equality between appearance weights and structural mass is assumed.
6. **Formal cardinal decoder:** for finite separable `K`, solve
   `K_y C K_x^T = Y (K_y 1)(K_x 1)^T` under an explicit conditioning bound; store `Y`, derive and
   hash `C`.

## Cross-domain transfers

| Donor domain | Mechanism transferred | StructSplat hypothesis |
|---|---|---|
| Wavelet lifting | Predict/update transforms with exact sample reconstruction | Treat the raster codec as a coarse authority and derive a continuous correction basis |
| RBF/spline interpolation | Cardinal prefilter before basis evaluation | Wider Gaussian overlap can preserve sample centers without fitting |
| Finite elements | Static condensation separates internal from boundary degrees of freedom | Eliminate appearance rows while preserving a compact physical/proposal interface |
| Multigrid | Separate coarse correction from high-frequency smoothing | Price structural coverage and appearance residuals on different levels/planes |
| Compiler intermediate representations | Stable producer/consumer contract with multiple backends | Define a packet/query IR independent of the current NumPy or CUDA evaluator |
| Columnar databases | Independently encoded columns and predicate-specific reads | Decode/query appearance and structural mass at different rates and access patterns |
| Model-predictive control | Transactional local update with explicit guardrails | Accept topology changes only under pixel/patch/downstream Pareto guards |
| Experimental design | Development/confirmation split and killing rules | Stop exposed-image tuning before any held-out or multiview claim |

These are mechanism transfers, not evidence that the transplanted method will work.

## Portfolio scorecard

Scores are `1` weak to `5` strong and are design priors, not measurements.

| Candidate | High-frequency potential | Complete-rate potential | Cold speed | RTGS fit | Falsifiability | Risk |
|---|---:|---:|---:|---:|---:|---:|
| Explicit fitted rows + actual coder | 4 | 2 | 1 | 5 | 4 | optimizer/coder complexity |
| Amortized encoder + correction | 4 | 4 | 4 | 4 | 3 | training and domain shift |
| Regional Lagrangian allocator | 4 | 4 | 2 | 4 | 5 | many interacting rate knobs |
| Seed grammar + residual raster | 4 | 5 | 4 | 4 | 4 | structured-codec prior art |
| Codec-native dual plane | 5 at samples | 4 | 5 | 4 | 5 | off-grid ringing and structural insufficiency |
| Shared multiview structure + deltas | 5 | 5 | 3 | 5 | 3 | requires a real multiview protocol |

CORE-016 selects the codec-native dual plane because it has the cheapest decisive killing test and
directly attacks the measured coupling failure. It is not selected as a production winner.

## Selected pilot

The v2 packet contains three canonical ZIP members: manifest, conventional appearance payload, and
lossless Field V2 structure. The appearance decoder optionally performs bounded separable Jacobi
prefiltering and evaluates a finite normalized Gaussian lattice. The structural allocator warps a
seeded Halton sequence through structure-tensor density and exports exact-count anisotropic rows
with independent mass. The optional adapter returns a paired realtime-gs structural field and query
backend without editing realtime-gs.

The killing rule requires at least one exposed packet to be artifact-safe at decoded pixel centers,
smaller than the supplied source file, materially faster to produce than the contextual iterative
fit, and query-compatible with realtime-gs. Passing only permits continued study.

## Development evidence and decision

The selected exposed C0001 configuration passes that narrow killing rule: the complete packet is
3,896,344 bytes, decoded pixel-center error is below display quantization, and the paired backend is
consumed by the existing synthetic two-view `CompactCarveInitializer`. The source-file ratio is
`3.662x`, but it compares a full source JPEG with a crop packet; the fairer crop-local canonical PNG
ratio is only `1.139x`. The component-summed reference encode estimate is 3.29 seconds versus a
contextual 315-second historical fit with different preprocessing and work.

The off-grid diagnostic prevents promotion. Against bilinear interpolation of the decoded raster,
the sampled query gives 49.38 dB, but `3.78%` of sampled channels overshoot their local 2x2 envelope
and `0.0244%` leave `[0,1]` (range `-0.0051..1.0084`). Bilinear is only a control, not continuous
scene truth. The selection is also post-hoc on the exposed image, there is no held-out dataset, and
the 512 structural proposals have not been validated on real multiview reconstruction.

Decision: keep ADR-0032 and CORE-016 default-off as a surviving systems pilot. Do not promote a
compression, convergence, visual-artifact, continuous-quality, or downstream-3D claim.

## Decisive next evidence programs

1. **Continuous phase/ringing assay.** On analytic edge/texture functions and supersampled natural
   images, sweep codec, sigma, radius and conditioning; report phase-stratified error, envelope
   escape, spectral retention, decoder cost, and a preregistered overshoot limit.
2. **Held-out complete-rate assay.** Freeze development choices, then compare full-frame packet
   bytes against source, PNG/WebP/AVIF/JXL and explicit Gaussian packets at matched pixel and
   off-grid distortion. Include decode memory and query throughput.
3. **Real multiview downstream assay.** Produce paired packets for calibrated views, run unchanged
   realtime-gs lifting/training, and measure 3D quality, time-to-target, total bytes, and render FPS
   while sweeping structural count independently of appearance rate.
4. **Kernel/backend assay.** Implement a fused CUDA/texture query oracle and require numerical
   parity with NumPy before claiming realtime query performance.

Kill the candidate if the cardinal field cannot meet a frozen ringing guard at useful throughput,
if structural count does not predict downstream quality, or if the complete-rate advantage vanishes
under matched full-frame preprocessing.

## Primary references

- [GSICO: Gaussian Splatting Image Compression](https://arxiv.org/abs/2601.14510)
- [SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation](https://arxiv.org/abs/2603.07789)
- [Structure-Guided Gaussian Allocation for Image Representation](https://arxiv.org/abs/2512.24018)
- [Cardinal Interpolation with Gaussian Kernels](https://arxiv.org/abs/1008.3168)
- [An analytic approximation of the cardinal Gaussian interpolation operator](https://doi.org/10.1016/j.amc.2009.08.037)

Repository context and the exact audit are in
[`architecture.md`](../architecture.md),
[`ADR-0032`](../adr/0032-codec-native-dual-plane-observation.md), and
[`2026-08-06-codec-native-dual-plane-results-audit.md`](2026-08-06-codec-native-dual-plane-results-audit.md).
