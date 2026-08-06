# Proposed Additive Observation Field V2

## Status and authority

This document is the design and task-map authority for the proposed next StructSplat production
pipeline. It is **not** an accepted architecture decision, a shipped default, or evidence that the
proposal works. The current implementation remains the normalized pipeline described in
`docs/architecture.md` and governed by ADR-0003, with the opt-in additive renderer already
authorized by ADR-0006.

`tasks/INDEX.md` and the referenced task files remain the outcome and protocol authorities. A task
may refine this proposal only within its stated decision boundary. A result-bearing task must bind
its protocol before execution, retain negative outcomes, and update this document only after its
result is independently reviewed. The motivating research state is recorded by N207--N210 and
observations O95--O102, whose artifact-committed architecture/constraint/heuristic entries remain
non-claim knowledge; none is a promoted method result.

## Design inputs and evidence boundary

The proposal carries forward five different kinds of information without treating them as one
strength of evidence:

- **Current contract:** ADR-0003 and CORE-012 keep normalized rendering and the single maintained
  conversion entry point authoritative. ADR-0006 makes additive rendering an exact opt-in mode;
  it did not select an additive production field.
- **Live diagnostic:** the source-bound `frame_00008` audit shows that the three current arms alter
  compositor, target/mask policy, containment, initialization, topology schedule, and commit gates
  together. Its no-boundary arm also has a loss/gate/profile mismatch. The run may motivate
  BENCH-019/020, but its quality or timing differences cannot select a renderer, loss, or stage
  order. See `ara/evidence/frame00008-three-arm-audit-2026-08-03/run.md` and staged O95/O96.
- **Semantic fact:** the current additive external field stores scalar amplitude times RGB, not
  spherical harmonics. RGB reconstruction identifies the product but not a unique amplitude/color
  factorization; any density meaning therefore needs independent supervision and downstream
  validation.
- **Local experiment history:** exact/alternating color solve, post-fit versus fit-time QAT,
  normalized pyramids, topology tails, and prior SGI-inspired entropy work already provide controls
  and negative boundaries. New tasks reuse those implementations and must not promote a rejected
  mechanism by renaming it.
- **External prior art:** SGI, LocoADC, AIR, WIPES, Faster-GS, and related methods establish direct
  controls for structured coding, regional allocation/merge, amortization, richer atoms, and fused
  optimization. Transfer into this workload is a hypothesis until locally and natively measured.

The key unavailable evidence is exactly what Gate A requests: whether any Stage-1 field diagnostic
predicts fixed-protocol realtime-gs utility, and whether direct additive, dual additive, or
normalized semantics are best once the other axes are matched. Until those results, the field and
objective below are candidate interfaces rather than accepted answers.

## Objective

Build one explicit 2D observation field for realtime-gs and image representation that can occupy a
Pareto frontier across four independently reported quantities:

1. reconstruction and downstream lifting quality;
2. encoding latency and throughput;
3. convergence at fixed work and wall-clock budgets; and
4. cold-decoded complete-container bytes plus decode/query cost.

The system does not collapse these into one undocumented score. A production profile maximizes
quality subject to explicit byte, encode-time, and query-latency constraints. Research reports
retain the complete Pareto set.

## Functional problem signature

The primary workload is a collection of high-resolution, calibrated, alpha-masked camera views.
Each view is fitted independently as a 2D field, then consumed by realtime-gs through point queries,
coverage/density operations, and later 3D reconstruction. The field therefore has two consumers:

- image reconstruction, which observes accumulated RGB; and
- structural lifting, which may observe component geometry, coverage, mass, and alpha.

This distinction is load-bearing. RGB reconstruction alone identifies an additive component's
premultiplied RGB coefficient, but it does not identify an arbitrary factorization of that
coefficient into scalar weight and color. A future field must either omit the scalar or define it
through an independent structural objective.

## Non-goals

- Do not delete or rewrite the normalized reference renderer or its historical evidence.
- Do not infer that additive rendering is the production winner before BENCH-020.
- Do not add spherical harmonics to a single-view 2D image field.
- Do not treat row count, analytical parameter bits, NPZ bytes, or padded-canvas bpp as actual-rate
  evidence.
- Do not make hard Gaussian containment the only way to represent alpha boundaries.
- Do not add a neural decoder to the realtime query path without measuring cold load, expansion,
  random access, and steady-state query cost.
- Do not claim regional densification, structured seeds, Predict--Optimize--Distill,
  frequency-bearing atoms, or fused per-Gaussian optimization as novel StructSplat mechanisms.

## Field contract

### Candidate semantic object

`ObservationField2D` is the proposed typed boundary. CORE-013 owns its exact schema. Conceptually it
contains:

```text
geometry:
  means_xy          float/quantized [N,2], crop-local pixel coordinates
  covariance        RS or Cholesky [N,3], positive definite

appearance:
  rgb_coeff         [N,3], authoritative premultiplied additive coefficient
  coefficient_domain signed or nonnegative, explicit and versioned
  background_rgb    optional [3], alpha-gated DC term counted in the stream

structure:
  structural_mass   optional [N], one nonnegative independently defined scalar per row

observation:
  packed_alpha      optional exact or explicitly lossy alpha payload
  canvas/crop       full-canvas and fit-window transforms
  camera            calibrated camera record when exported for realtime-gs

semantics:
  renderer_equation, support/fade/filter conventions, alpha policy, schema version
```

The candidate foreground-appearance equation is

\[
F(x)=b+\sum_i k_i(x;\mu_i,\Sigma_i)\,p_i,
\]

where `p_i = rgb_coeff_i` and `b` is either zero or the explicitly stored `background_rgb`.
BENCH-020 selects whether coefficients are nonnegative or signed/bounded residuals and whether a
DC term is present, using CORE-009 as the existing background control. The alpha policy defines
whether the exported/rendered value is `F`, an alpha-matted value, or a hard-contained field; it is
never an implicit display clamp. If structural mass survives BENCH-019/020, its separate field is

\[
S(x)=\sum_i k_i(x;\mu_i,\Sigma_i)\,m_i,\qquad m_i\ge 0.
\]

Normalized responsibilities may be derived as

\[
r_i(x)=\frac{k_i(x)m_i}{\sum_j k_j(x)m_j+\epsilon},
\]

but those responsibilities do not define RGB compositing. `rgb_coeff` and `structural_mass` have
distinct losses, quantizers, entropy contexts, and downstream meanings. `color = rgb_coeff/mass`
is not an authoritative stored quantity.

### Contract alternatives decided by BENCH-020

- **Direct 8-parameter field:** geometry plus `rgb_coeff`; alpha supplies occupancy and no scalar
  mass is exposed downstream.
- **Dual 9-parameter field:** geometry plus `rgb_coeff` and independently supervised
  `structural_mass`.
- **Current factorized 9-parameter control:** scalar weight times color, retained only as the
  incumbent comparison.
- **Normalized control:** current normalized `GaussianField`, with its exact renderer semantics
  preserved and never relabelled as additive.

Each additive family has an explicit coefficient-domain/DC sub-decision: zero-DC nonnegative
coefficients versus a counted alpha-gated DC term with bounded/signed residual coefficients. A
cheap fixed-geometry oracle may eliminate a domain before the full factorial, but the rule and
confirmation arm are frozen in BENCH-020. Stream validation, solvers, and quantizers inherit the
selected domain exactly.

Adapters must say whether they preserve pixels, component semantics, both, or neither. A converted
normalized field is not an exact additive teacher merely because its arrays have compatible shapes.

### CORE-013 reference realization

`src/structsplat/observation_field.py` now implements the default-off schema-`2.0.0` reference
object and CPU oracle. This is an implementation substrate pending distinct review, not the
BENCH-020 semantic verdict and not a production/default change. The module is NumPy-only and does
not import, wrap, or mutate the current torch fitting path.

The reference schema makes the following choices explicit:

- Geometry is crop-local RS geometry: `means_xy [N,2]`, `log_scales_xy [N,2]`, and
  `rotations_rad [N]`. Optional per-row isotropic covariance variance and global AA dilation are
  applied in covariance space before inversion.
- The peak-one kernel records infinite, elliptical-cutoff, or rounded-center covariance-AABB
  support. The AABB option exactly names the current `render.py` support rectangle, including
  ties-to-even center rounding, minimum one-pixel radius, sigma cutoff, and optional subtractive
  tail fade. A future producer cannot call an elliptical cutoff the incumbent support policy.
- Raw appearance, alpha-matted appearance, structural density/responsibilities, and display
  clipping are separate functions. `linear_rgb_unclipped` is the oracle space; neither alpha nor
  `[0,1]` clipping is applied by `appearance_raw` or `render_raw`.
- Packed alpha is row-major little-bit-order binary data over the fit crop. Its metadata says
  whether the mask was exact binary input or produced by an explicitly recorded lossy threshold.
  Boundary policy and output-matting policy are independent declarations.
- Canvas/crop transforms are integral and bounded. Optional camera data is immutable finite JSON
  under a source-owned schema name; the contract does not invent a universal camera grammar.
- Every float array is canonical little-endian float32 or float64, finite, shape-checked, copied,
  and read-only. Nonnegative coefficients/mass, positive covariance, alpha padding bits, optional
  field presence, metadata keys, and schema/container versions fail closed.

The lossless reference NPZ stores deterministic NPY members plus canonical JSON metadata,
per-array hashes, and a semantic-and-array canonical content hash. It preserves values, dtype,
shape, signed zero, semantics, crop, alpha, and camera metadata exactly. It is deliberately not
the COMP-013 compressed stream, and its ZIP/NPZ byte count is not rate evidence.
The current reader rejects unknown members and keys. Schema growth uses explicit version dispatch;
there is intentionally no generic mixed-atom, executable-generator, or neural-decoder payload
hidden inside this base contract.

Adapter declarations keep pixel and component exactness separate:

| Source | Output | Pixel exact | Component semantics exact |
|---|---|---:|---:|
| authoritative direct additive coefficients | Field V2 | yes | yes |
| current constant-color factorized additive field | opacity folded into `rgb_coeff` | yes, under the bound current support/filter settings | no; the color/opacity gauge is discarded and opacity is not relabelled mass |
| current normalized weighted sum | no field by default | no | no |
| normalized weighted sum with `permit_inexact=True` | named approximate additive control | no | no |

Affine-color legacy fields fail the constant-coefficient exact adapter rather than dropping their
gradients. A normalized adapter is structurally forbidden from claiming either exactness flag.
BENCH-020 remains responsible for deciding whether any additive candidate should proceed.

### BENCH-020 experiment substrate

`benchmarks.field_semantics_factorial` now implements the default-off decision harness, but no
semantic outcome has been run or accepted. It freezes exact direct-additive, incumbent-factorized,
normalized-plain, and maintained-normalized records; a dual-additive arm is rejected unless
BENCH-019 supplies an independently supervised structural target. A normalized equation cannot be
relabeled additive, and the direct coefficient authority cannot be represented as factorized
opacity in its canonical byte ledger.

The harness separates coefficient-domain screening, development selection, and confirmation into
distinct outcome roots. Fixed-row and equal-complete-raw-byte lanes share sealed ordered geometry
banks; DC/background, packed alpha, structural mass, factorized opacity, geometry, appearance, and
metadata bytes are separately auditable. Alpha-gated and hard-contained arms bind one consistent
target/loss/gate/profile scope. Result artifacts cannot escape their phase root, and development or
confirmation artifacts appearing before their lock invalidate the analysis.

Each successful cell preserves its field payload; the semantic manifest binds the payload format,
hash, byte count, renderer equation, coefficient policy, and authoritative pre-clamp render. The
frozen convergence contract records first observed attainment of the primary foreground-PSNR
target and normalized PSNR-time AUC over the full phase wall-time horizon, holding the terminal
observation after an early finish. Per-cell histories replay both values. Endpoint image,
alpha/outside, downstream, row/byte, wall-time, renderer-call, and memory records remain separate;
no endpoint score is used as an unreported convergence proxy.

The killing screen compares an additive candidate with both matched incumbent-additive and
normalized-plain controls in every gate lane, clusters uncertainty by capture, and advances only
one nondominated semantic/alpha candidate. A heterogeneous frontier is terminal rather than
collapsed with a hidden scalar. General scope requires three independent development capture
groups and three disjoint confirmation capture groups. Because BENCH-019 has no validated Stage-1
target or formal downstream result, the current implementation is plumbing only. A source-only
portfolio now preassigns three development and three confirmation acquisition groups, but five
groups still lack matched fields and several lack frozen adapters, keyframes, masks, or splits;
source acquisition is not benchmark readiness. INIT-010 and FIT-044--049 remain blocked on a
reviewed, sealed BENCH-020 outcome.

## Alpha and boundary policy

Packed alpha is a first-class optional stream and is counted whenever an arm stores or consumes it;
an arm that omits alpha records zero alpha bytes and cannot use alpha-gated fit, render, or query
semantics. When alpha is available, rendering and downstream queries may gate by it rather than
forcing every Gaussian's finite support to remain inside the foreground. Hard containment remains
a control for consumers that require exact support containment without a stored mask.

BENCH-020 compares alpha-gated and hard-contained policies under the same target, geometry budget,
and work. Until that result, the proposal assumes only that alpha separation is representable, not
that it wins every boundary or downstream metric.

## Objective family

The complete research objective is

\[
  D_{rgb} + \eta D_{structure} + \kappa D_{downstream} + \lambda R,
\]

subject to encode-time, memory, cold-decode, and query-latency constraints. The terms are introduced
in stages rather than optimized jointly before their utility is identified:

- `D_rgb`: masked/matted RGB MSE is the baseline for a PSNR-oriented profile. FIT-049 compares it
  against the bounded existing L1+SSIM and Charbonnier controls. MS-SSIM and LPIPS remain
  evaluation/checkpoint guardrails unless that task explicitly authorizes a training term.
- `D_structure`: a nonnegative coverage or alpha-derived target only if BENCH-019 shows structural
  mass predicts downstream utility.
- `D_downstream`: the smallest validated Stage-1 surrogate for realtime-gs lifting, or zero if no
  Stage-1 diagnostic predicts the downstream result reliably.
- `R`: an entropy estimate during QAT and the complete cold stream at selection. Actual bytes remain
  authoritative.

Post-fit QAT is the baseline because the existing local comparison did not promote fit-time
entropy-aware optimization. Joint `D + lambda R` work must beat that control rather than replace it
by assumption.

## Proposed fitting pipeline

### 0. Prepare and bind

Decode the source exactly once, materialize the fit crop and alpha bytes, preserve the crop-to-canvas
transform and camera, choose an active-crop byte budget, and bind source/config hashes. Full-canvas
bpp is reported too, but never substitutes for active-crop bpp.

### 1. Initialize geometry

INIT-010 transfers the existing deterministic initializer families to the selected semantics and
produces geometry plus optional initial structural mass. The first production candidate uses its
BENCH-021-confirmed winner. Gradient initialization and progressive WSE remain controls. A learned
predictor is a later amortized initializer and may not change the field contract.

### 2. Solve conditional coefficients

At fixed geometry, solve additive RGB coefficients with a matrix-free bounded or regularized linear
solver under the BENCH-020 coefficient domain. If structural mass is selected, solve it
independently with a nonnegative objective. The kernel operator is never materialized as a dense
`pixels x Gaussians` matrix; renderer forward and transpose products are the oracle interface.

### 3. Refine geometry

FIT-049 supplies the exact selected objective. Run short geometry blocks and alternate them with
coefficient solves. FIT-044 controls which parameter groups move; FIT-046 owns variable
projection. FIT-047 may replace full-frame inner steps with unbiased, probability-recorded tile
samples. FIT-048 decides single-scale/full-N versus coarse-to-fine/progressive stage order.
Periodic full-resolution evaluations are used for selection and stopping.

### 4. Reallocate capacity only when earned

FIT-045 compares fixed-full-N, global residual births, regional allocation, LocoADC-style
region-wise densification, and similarity-driven merging. The total row and proposal budgets remain
matched during the mechanism screen. FIT-030 later prices topology and precision actions in
complete bytes.

HIER-005 supplies a default-off contraction control for that screen. It treats the one-Gaussian-
per-active-pixel endpoint procedurally rather than as trainable rows, exposes ready quadtree cells,
and considers three bounded actions: replace two to four atoms by a moment parent, retain a parent
plus one analytic detail basis, or merge one pair when needed to reach an exact count. A cheap RGB
proxy schedules the image-sized region frontier. Within a shortlisted region, candidate appearance
distortion uses the exact continuous product of peak-one Gaussians,

\[
 \langle g_i,g_j\rangle =
 2\pi\frac{|\Sigma_i|^{1/2}|\Sigma_j|^{1/2}}
 {|\Sigma_i+\Sigma_j|^{1/2}}
 \exp\!\left[-\tfrac12(\mu_i-\mu_j)^T
 (\Sigma_i+\Sigma_j)^{-1}(\mu_i-\mu_j)\right].
\]

Before an action commits, the implementation removes the old local contribution, solves its one-
or two-column coefficient system against the true discrete image residual, and measures the exact
finite-support SSE delta. Only support-disjoint actions share a batch. The default
`pair_policy=exact_count` favors whole-cell contractions and uses pair actions only when no eligible
whole-cell action exists; `always` is the slower quality-first control. The resulting direct field
and its deterministic lossless reference container are test substrates. The configured bytes per
row, raw array bytes, and reference NPZ bytes are not COMP-013 rate, and no HIER-005 diagnostic may
advance a semantic/default/compression claim.

An explicitly enabled recovery lane interleaves bounded Adam blocks after fixed fractions of the
requested row-count reduction. The `touched` scope may move only active atom slots already
replaced, merged, or retained by a contraction; never-touched pixel leaves are detached into a
fixed base render and remain bitwise unchanged. Previously touched active rows stay eligible so
later contractions can repair their interaction. The comparison-only `all_error_weighted` scope
instead materializes every active row. It computes RGB residual MSE, applies mask-aware Gaussian
smoothing (blurred masked error divided by blurred mask), and uses one additive-renderer color VJP
to average that error under every Gaussian without a dense pixel-by-row matrix. Bounded,
approximately mean-one exposure scores multiply the post-Adam parameter update—not the raw
gradient, whose fixed scale Adam would mostly cancel.

Both scopes use separate means/scales/rotation/coefficient learning rates and per-checkpoint trust
regions; the best masked-SSE iterate is committed only on a strict improvement, otherwise the
entire checkpoint rolls back. An accepted geometry update invalidates and rebuilds the ready
proposal frontier. Recovery is disabled by default, `touched` remains its default scope, requested
counts do not change, and neither scope defines a new field semantic. CPU recovery is covered by
bit-determinism tests; CUDA additive atomic gradients make optimizer trajectories numerically
non-bit-reproducible and reports label that limitation.

When a fixed-count diagnostic fails its declared localized-artifact gate, HIER-005 can run a
separately labeled terminal rescue rather than silently weakening the gate. The rescue starts from
one cold persisted signed direct field; keeps all base means, scales, rotations, and RGB
coefficients bit-exact; chooses foreground residual peaks with stable ordering and local NMS; and
adds fixed-scale, fixed-rotation signed rows. Only the new RGB coefficients are optimized. The
unchanged base competes with every optimizer checkpoint, ordered lexicographically by the larger
of normalized raw worst-pixel and maximum black-matted 7×7-patch RMSE, then by raw SSE. The final
cold displayed 8-bit PNG metrics remain the acceptance authority. Rescue rows are explicit count
and payload overhead, not an exact-count solution, a compressed exception stream, or an
artifact-free certificate.

The task-local driver can resize a native RGB/mask pair for bounded diagnostics, but it records
native and evaluation dimensions separately. Original-file/field ratios therefore remain visibly
resolution-mismatched; same-raster PNG ratios and bits per active foreground pixel are reported in
parallel. Its HTML reports plot every available quality, localized-artifact, byte-proxy, timing,
action, recovery/repair, and renderer-parity outcome against achieved count, include full and
worst-neighborhood visuals, and snapshot the untracked task sources for dirty-run audit.

HIER-006 is the bounded parent-preserving residual-quadtree control for the opposite stage order.
It begins with every mask-present cell at a declared coarse level. Splitting a selected frontier
cell retains its parent and appends all mask-present child Gaussians; geometry is the active-pixel
moment plus fixed leaf variance, and initial signed RGB is the child's mean current residual. Only
that new coefficient block is fitted against an immutable rendered prefix. Candidate cells use
mask-normalized Gaussian-smoothed residual energy per appended row. The unchanged prefix and
optimizer checkpoints are ordered by raw normalized pixel/7×7 violation and then SSE; a
float32-scale equivalence band prevents unchanged maxima from being misread as regressions across
different reduction kernels. A cold joint render is the commit authority, so older field rows are
bit-exact and every accepted count is independently decodable.

That clean progressive property did not make the literal hierarchy competitive on the frozen
exposed C0001 diagnostic. After correcting and retaining evidence for a cross-domain comparison
roundoff defect, the unchanged protocol produced 27.805 dB at 3,986 rows and 32.882 dB at 8,192,
with displayed pixel/7×7 maxima 0.2223/0.0860 and 0.1073/0.0375. Both fail the declared 0.02/0.01
gate; the existing HIER-005 context is 30.481 dB at 4,096 and 52.356 dB with a passing gate at
8,192. The hierarchy spends 5,106 terminal rows on retained ancestors and reaches only 3,086
level-0 leaves, while summed smoothed error leaves the worst isolated boundary cell unsplit at
level 1. A repeat reproduced the displayed maxima exactly. Therefore this fixed-prefix tree is a
negative FIT-048/FIT-045 control: future hierarchical work should test quadtree scheduling without
assuming literal retained Gaussian ancestors are free, and should separately test artifact-first
allocation, joint/local coefficient reconciliation, or nested detail bases under matched complete
bytes and work.

HIER-006 reports both the full 32-byte-per-row/canonical/reference sizes and a smaller structural
proxy containing float32 RGB plus one nominal tree bit per retained node. The proxy assumes a
shared mask-derived geometry generator and omits a real tree grammar, header, entropy model, and
cold decoder. Its very large native-JPEG ratio is additionally resolution-mismatched because the
field is evaluated at 512×443. Neither number is COMP-013 rate.

HIER-007 tests the prescribed scheduler-only successor without mutating HIER-006's negative
control. A frontier split removes one active parent and activates its complete mask-present child
group; inactive ancestors are tree/history records rather than rendered rows. A frozen 2x2 factors
smoothed-energy versus artifact-first priority and new-only versus support-overlap local RGB
reconciliation. The overlap arm applies mask-normalized smoothed-error row multipliers after Adam,
freezes all nonlocal coefficients and all geometry, and accepts only a cold full-field artifact/SSE
improvement. Rejected multi-parent trials back off by deterministic priority-prefix halving.

The exposed C0001 screen rejects the combined mechanism. Parent replacement itself helps: the
energy/new-only 8,192-row arm reaches 40.035 dB and 0.0472/0.0215 displayed pixel/7x7 maxima versus
HIER-006's 32.882 dB and 0.1073/0.0375, but it still fails the fixed gate and remains below the
contextual HIER-005 8,192-row pass. Artifact-first/overlap falls to 26.035 dB with clearly visible
grid/ring defects, 1,773 attempts, and a 1.21 MB progressive-event proxy. The lexicographic hard-
maximum commit can accept early SSE regressions, artifact-first allocation over-concentrates fine
leaves, overlap updates can create late hotspots in coarse cells, and complete-child expansion
leaves no reserve to repair them at the cap. Therefore Field V2 work must not promote this policy.
A new screen should combine a smooth commit-aligned patch/tail objective, regional no-new-hotspot
and SSE trust regions, budget reservation, and a parent/child cross-fade or lifting constraint
before hard deactivation.

HIER-008 separately tests the user's direct-neighbour overlap and feature-adaptive elimination
proposal. Its full pixel lattice uses either near-delta `0.18 px` or overlapping `0.50 px`
Gaussians. A matrix-free least-squares prefit makes both endpoints exact before removal; the
overlap cell therefore does not confuse blur from reused source RGB with contraction distortion.
The scheduler factor compares HIER-005 quadtree contraction with nested WSE survivors ranked by
density-adaptive crowding divided by a same-side local Schur removal price. Every reduced field
then receives the same bounded all-row RGB/centre/log-scale optimizer, with explicit step-zero
attribution and a raw pixel/patch non-regression veto.

The frozen exposed C0001 result retains overlap but rejects fixed-scale survivor elimination. At
8,192 rows overlap raises the quadtree cell by 10.824 dB to 45.953 dB and the optimizer contributes
2.144 dB, yet the displayed `0.1077/0.0253` maxima fail the artifact gate and remain below the
contextual HIER-005 touched-recovery pass. The overlap WSE cell reaches only 22.878 dB. Although its
feature coverage succeeds, q99/max centre gaps of 1.44/2.08 px receive almost no weight from a
0.50-px Gaussian, producing visible dot holes. At 4,096, overlap/quadtree reaches 31.096 dB but
retains visible periodic rings. Thus exact overlap prefitting can enter a future HIER-005 recovery
screen, while any WSE successor must dynamically expand/merge covariance and price actual
post-removal patch distortion; the frozen static method must not enter FIT-045 or production.

HIER-009 executes that bounded recovery screen. It uses the exact-overlap lattice with HIER-005's
current-field propose/refit/commit/recover/rebuild loop, then factors touched-only recovery against
the user's requested direct 3x3 Gaussian-neighbor halo. An optional 5% deterministic reserve keeps
thin/high-feature pixel-leaf geometry exact while allowing RGB refits. Never-selected rows remain
a detached fixed base, accepted changed neighbors persist, and overfull protected regions fail
closed without stopping independent regions.

The 3x3 scope is useful but not a universal replacement. At N=4,096 it changes overlap/touched
from 39.802 dB and displayed `0.0900/0.0334` pixel/7x7 maxima to 40.801 dB and
`0.0799/0.0278`; the visible quadtree blocks become much weaker distributed texture error.
Protection further reaches 41.115 dB and a 0.0251 patch maximum. At N=8,192 the same halo spreads
error: pixel maximum falls, but PSNR drops from 47.395 to 45.963 dB and 7x7 maximum rises from
0.0229 to 0.0274. Protection recovers part of this to 46.991 dB/0.0205, still below the
delta/touched fallback, whose 52.338 dB and `0.0148/0.0053` are the only passing cell. All protected
means/covariances remain exact and every recovery checkpoint is active. Retain adaptive
neighborhood recovery and protected detail as successor components, but require an explicit local-
artifact/Pareto acceptance objective before any wider scope or default. The 159,424/290,496-byte
uncoded fields remain 5.45x/9.93x larger than the same-raster PNG and are not compression.

### 5. Quantize and entropy-code the direct field

COMP-013 defines the self-contained Field V2 stream: per-attribute scalar QAT first, spatially
indexed context entropy coding next, and codebooks only as measured controls. Alpha, indexes,
headers, ranges, and codebooks are included in the rate. COMP-008/009's negative direct-attribute
context result remains binding evidence that an entropy model alone should not be assumed to
produce SGI-like compression.

### 6. Decide whether decoded structure is necessary

BENCH-025 compares the strongest direct stream with native SGI and a bounded seed-local generator
oracle. If direct coding meets the frozen rate/quality/query target, COMP-014 closes without code.
Only a positive complete-byte result authorizes one seed-structured grammar. That grammar must
decode to the same `ObservationField2D`; its generator, seed attributes, residuals, indexes,
training prior, cold expansion, and resident memory are never free.

### 7. Cold-decode and select

Every candidate is decoded in a fresh process, rendered and queried through the declared semantics,
and scored on quality, downstream utility, actual bytes, load time, query latency, fit time, and
peak memory. The selected checkpoint is a Pareto point, not merely the final optimizer state.

## Convergence and stopping

The proposed pipeline has three explicit stopping tests:

1. conditional coefficient solve residual or KKT tolerance;
2. full-resolution distortion improvement per elapsed second over a fixed evaluation window; and
3. marginal distortion improvement per complete byte for topology or precision actions.

A sequence of rejected transactional blocks is not convergence. The current safe schedule remains
available to historical profiles; CORE-014's candidate pipeline uses stage-boundary checkpointing
and a rare global fail-closed validation, not per-block trial-and-rollback.

## Rate and performance accounting

Every result records:

- complete container bytes and active-crop/full-canvas bpp;
- row count and bytes by alpha, geometry, appearance, structural mass, index, codebook, and header;
- float and cold-decoded PSNR, MS-SSIM, LPIPS, boundary metrics, and downstream metrics;
- iterations, renderer calls, sampled pixels/tiles, nominal horizon, and early-stop state;
- init, solve, geometry, allocation, QAT, encode, decode, render, and query times;
- peak allocated/reserved GPU memory and hardware/software provenance.

Fixed-row, fixed-work, fixed-wall-time, and fixed-byte comparisons answer different questions and
are never merged into one unlabeled table.

## Implementation ownership

- `gaussians.py`: historical normalized `GaussianField`; no silent semantic mutation.
- new Field V2 module selected by CORE-013: typed additive coefficients, optional mass, alpha and
  coordinate metadata.
- `render.py` / owned CUDA extension: additive appearance and structural-density oracle equations.
- `fit.py` or a bounded new fitter module: variable projection and tile-sampled geometry work.
- `codec.py`: legacy SSPL1 plus versioned direct Field V2 cold stream and, only if BENCH-025
  authorizes it, separately versioned seed-structured dispatch; old streams remain readable.
- `pipeline.py`: current profile unchanged; CORE-014 adds a default-off candidate profile through
  the existing conversion CLI rather than a second launcher.
- `benchmarks/` and `scripts/experiments/`: frozen comparisons and portable bundles.
- realtime-gs: external downstream consumer and evaluator, pinned by commit/environment in formal
  runs; it does not become a second StructSplat task authority.

## Task graph

### Gate A — identify the contract

| Task | Decision |
|---|---|
| BENCH-019 | Which Stage-1 diagnostics, if any, predict downstream realtime-gs utility? |
| CORE-013 | Provide a lossless, versioned, default-off Field V2 semantic boundary. |
| BENCH-020 | Select additive/direct, additive/dual, or normalized semantics and alpha policy. |

No production implementation proceeds past this gate if BENCH-020 is negative or unavailable.

BENCH-019's measurement substrate is implemented default-off in
`benchmarks.stage1_downstream_objective`: clean-source/prospective protocol sealing, a passive
realtime-gs row boundary, semantic and downstream-factor digest checks, A/A replay, within-frame
rank correlations, leave-one-frame-out diagnostics, capture-cluster bootstrap, fail-closed
missing cells, and a portable report/checker contract. This is plumbing, not an outcome. The
formal protocol and result remain open until a distinct reviewer approves the exact digest.

The external realtime-gs driver checkpoint `d3e76fe` supplies the matching passive row exporter:
exact additive/normalized semantic preservation, family/A-A invariant downstream factors, sealed
JSON-pointer metrics, six artifact descriptors, explicit error rows, and receipt-required stable
assembly. Its calibrated diagnostic intentionally stopped before downstream execution and passed
the StructSplat row validator. The checkpoint and its 3+3 source portfolio remain pending distinct
implementation review, so neither is an accepted formal executor yet.

The currently available matched-field data cannot support the general-surrogate branch. Stage
`frame_00008` has two complete 26-view families while the mask-contained family was 13/26 at the
portfolio snapshot; `frame_00009` lacks the matched three-family set. The other acquired source
groups have no matched families. Before a general BENCH-019 decision, freeze the source adapters
and predictor collector, finish matched development production across independently reviewable
capture groups, and preserve the confirmation lock. A comparison restricted to the supplied
Stage frame remains useful as an explicitly workload-specific pipeline comparison, but it cannot
relax the general promotion gate.

### Gate B — select the convergence recipe

| Task | Decision |
|---|---|
| INIT-010 | Which deterministic initializer transfers to the selected field semantics? |
| FIT-049 | Which image/structure/downstream objective earns its compute and trade-offs? |
| FIT-048 | Does single-scale/full-N or coarse-to-fine/progressive staging converge best? |
| HIER-005 | Can an implicit pixel endpoint be contracted into an exact-count direct-field control? |
| HIER-006 | Can a parent-preserving residual quadtree provide artifact-safe progressive prefixes? |
| HIER-007 | Do parent replacement, artifact-first allocation, and overlap-local RGB reconciliation interact safely? |
| HIER-008 | Does exact pixel-neighbour overlap or feature/Schur WSE elimination improve contraction without visible holes? |
| FIT-044 | Does parameter-group staging help on the selected semantics? |
| FIT-045 | Does regional allocation/merging beat global and fixed-N controls? |
| FIT-046 | Does additive variable projection improve convergence or quality? |
| FIT-047 | Does unbiased tile sampling reduce work without changing the objective? |
| BENCH-021 | Which compatible combination wins under successive halving and full confirmation? |

### Gate C — rate and implementation

| Task | Decision |
|---|---|
| COMP-013 | Which Field V2 codec wins at complete actual rate and acceptable decode/query cost? |
| BENCH-025 | Is seed-generated decoded structure necessary at the target resolutions and rates? |
| COMP-014 | If authorized, does one seed-structured grammar beat the direct stream end to end? |
| FIT-030 | Which topology or precision action has the best distortion reduction per byte? |
| PORT-006 | Which fused/tiled/per-Gaussian implementation accelerates the selected algorithm? |

### Gate D — integrate and promote

| Task | Decision |
|---|---|
| CORE-014 | Assemble the selected components as one default-off maintained pipeline. |
| BENCH-022 | Does the complete candidate beat incumbent production controls on all required axes? |
| CORE-015 | If and only if BENCH-022 authorizes it, change the default and migration docs. |

### Gate E — amortization and optional research

| Task | Decision |
|---|---|
| FF-002 / FF-003 | Predict the selected field and make one checkpoint budget-elastic. |
| BENCH-023 | Does domain POD meet quality, latency, bytes, and training break-even gates? |
| CORE-008 | Only after the base pipeline is frozen, test WIPES-controlled richer atoms. |
| BENCH-024 | Test same-camera temporal warm starts/shared supports/delta coding without assuming cross-view alignment. |

FIT-042 remains the independent normalized fine-detail confirmation and is not on the production
critical path. BENCH-017, FIT-028/029, and BENCH-018 remain valid normalized transactional-pipeline
questions; they do not gate the additive candidate.

## Formal evidence plan

Every gate uses an outcome-blind development screen and a sealed confirmation when a positive
decision is possible. Data selection uses source metadata and mask geometry only. Correlated
camera views are clustered by frame/capture during uncertainty estimation and never treated as
independent images. Janelle establishes workload relevance; Kodak/CLIC or another pinned public
set establishes whether an image-codec statement transfers beyond one capture.

Native external methods retain their own renderer, optimizer, checkpoint, and rate semantics and
are centrally rescored. Local transplants are labelled as such. The required direct controls are
GaussianImage/GaussianImage++, Image-GS, AIR/Instant-GI where checkpoints are available, SGI for
structured coding, LocoADC for regional densification/merge, and WIPES for frequency-bearing
atoms. Faster-GS, FastGS, EDGS, GaussianVision, and GTC are systems/representation donors unless
their native task is reproduced; their native claims do not transfer to this 2D workload.

Primary references:

- [GaussianImage](https://arxiv.org/abs/2403.08551)
- [GaussianImage++](https://ojs.aaai.org/index.php/AAAI/article/view/37572)
- [AIR](https://arxiv.org/abs/2605.20820)
- [SGI](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_SGI_Structured_2D_Gaussians_for_Efficient_and_Compact_Large_Image_CVPR_2026_paper.pdf)
- [LocoADC](https://arxiv.org/abs/2607.17896)
- [WIPES](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_WIPES_Wavelet-based_Visual_Primitives_ICCV_2025_paper.html)
- [Faster-GS](https://openaccess.thecvf.com/content/CVPR2026/html/Hahlbohm_Faster-GS_Analyzing_and_Improving_Gaussian_Splatting_Optimization_CVPR_2026_paper.html)
- [FastGS](https://openaccess.thecvf.com/content/CVPR2026/html/Ren_FastGS_Training_3D_Gaussian_Splatting_in_100_Seconds_CVPR_2026_paper.html)
- [EDGS](https://openaccess.thecvf.com/content/CVPR2026/html/Kotovenko_EDGS_Eliminating_Densification_for_Efficient_Convergence_of_3DGS_CVPR_2026_paper.html)
- [GaussianVision](https://openaccess.thecvf.com/content/CVPR2026/html/Omri_GaussianVision_Vision-Language_Alignment_from_Compressed_Image_Representations_using_2D_Gaussian_CVPR_2026_paper.html)
- [GTC](https://arxiv.org/abs/2607.27943)

## Migration and compatibility

1. CORE-013 adds a new typed object and lossless reference serialization without changing
   `GaussianField` or SSPL1.
2. BENCH-020 selects semantics; a new ADR records the decision and explicitly supersedes only the
   production-default portion of ADR-0003 if additive wins. ADR-0003 remains the normalized
   reference history; ADR-0006 remains the additive-mode origin.
3. CORE-014 adds a named, default-off profile to `scripts/convert.py`; historical config and output
   hashes remain stable.
4. COMP-013 adds a versioned direct stream with strict old/new dispatch and cold-round-trip tests.
   BENCH-025 either selects it or authorizes separately versioned COMP-014 seed structure; both
   decode to the same semantic field boundary.
5. CORE-015 changes a default only after BENCH-022 and distinct architectural review. It supplies
   an explicit legacy profile and migration table.

## Risk register

- **Image metrics do not predict downstream utility.** BENCH-019 must either identify a usable
  surrogate or force downstream evaluation into every later gate.
- **Structural mass remains underidentified.** Select the 8-parameter contract; do not retain mass
  for symmetry.
- **Variable projection is too expensive.** Keep its matrix-free oracle and compare renderer calls
  and wall time; a quality-only win remains optional.
- **Tile sampling is biased by masks or residual sampling.** Require probability logging,
  inverse-propensity weighting, and full-objective oracle tests.
- **Regional allocation merely adds overhead.** FIT-045 fails closed against fixed-N/global and
  LocoADC controls.
- **Estimated rate misranks actual streams.** FIT-030 decisions replay every selected action through
  the complete codec before promotion.
- **Direct rows cannot reach the compression target.** BENCH-025 measures the missing structured-
  generation opportunity instead of assuming more entropy modeling will fix it; COMP-014 is
  implemented only on a positive gate.
- **Compression harms random access.** COMP-013/014 treat cold load, expanded memory, and query
  latency as hard guards.
- **Kernel speedups change trajectories.** PORT-006 requires gradient and end-to-end quality parity,
  not only a microbenchmark.
- **Learned prediction overfits a capture.** Entire frames/cameras are held out and training
  break-even is reported.
- **Optional richer atoms or temporal sharing expand scope prematurely.** CORE-008 and BENCH-024
  are conditional branches after the base production confirmation.

## Production definition of done

The redesign is production-ready only when:

- BENCH-019/020 identify and confirm the field semantics and downstream objective;
- INIT-010 and FIT-044--049 isolate initialization, objective, stage order, and optimizer work;
  BENCH-021 selects their reproducible convergence recipe;
- COMP-013, BENCH-025, and the terminal COMP-014 disposition select a complete actual-rate stream;
  PORT-006 establishes end-to-end implementation behavior;
- CORE-014 exposes one maintained default-off entrypoint and versioned field stream;
- BENCH-022 passes its predeclared quality, convergence, bytes, boundary/downstream, memory, and
  decode/query gates on a clean source and sealed confirmation;
- CORE-015 records independent review, the superseding ADR, backward compatibility, and default
  migration; and
- all evidence, claims, tasks, docs, report bundles, and `./scripts/verify.sh` agree.
