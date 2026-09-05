# Architecture

## Proposed Observation Field V2 (not current)

`docs/additive_field_v2.md` is the detailed, evidence-gated redesign and task graph for a possible
additive 2D observation field: authoritative RGB coefficients, optional independently defined
structural mass, first-class alpha, matrix-free fitting, complete-byte coding, cold query/load
measurement, and downstream realtime-gs validation. It is a proposal only. Until BENCH-020 and the
later production confirmation authorize a change, the normalized architecture and maintained
entrypoint below remain current; ADR-0003/0006 are not superseded.

CORE-013's default-off `structsplat.observation_field` module realizes the proposal's typed
schema-`2.0.0` boundary, closed-form NumPy oracle, strict lossless reference NPZ, canonical content
hash, and audited direct/factorized/normalized adapters. It is not a compressed codec, fitter,
supported conversion output, or semantic selection. Raw RGB, alpha matting, independent mass,
and display clipping remain separate operations, and legacy normalized fields cannot be declared
exact additive conversions.

HIER-005's default-off `structsplat.pixel_contraction` is an experimental direct-additive producer
for that boundary. It starts from procedural pixel leaves, contracts a quadtree frontier with
moment parents and an optional retained detail basis, re-solves every shortlisted action on its
exact finite-support pixel patch, and exports `ObservationField2D`. An opt-in recovery schedule can
interleave short differentiable additive-renderer fits. Its `touched` scope fits every active row
previously changed by a contraction while leaving never-touched leaves detached and bitwise fixed.
The separate `all_error_weighted` scope fits every active row and scales each post-Adam row update
by mask-aware Gaussian-smoothed residual energy averaged under that Gaussian through one
matrix-free renderer VJP. The default progress schedule prices recovery work by fractions of the
requested count reduction, and both scopes reject any checkpoint that does not reduce masked SSE.
For fail-closed artifact diagnostics, the same module also exposes a bounded terminal local-rescue
API for signed direct fields: it freezes every base array, selects high-error foreground pixels by
stable residual ranking plus local NMS, appends fixed isotropic geometry, and optimizes only the
new RGB coefficients. Checkpoints are ordered first by normalized raw worst-pixel/7×7-patch error
and then by SSE; an unchanged-base checkpoint is always eligible. This may increase row count and
does not certify artifact freedom—the cold displayed-PNG gate remains authoritative.
Its row-byte price is an uncoded proposal estimate, its task driver is diagnostic, and it neither
selects Field V2 semantics nor enters `pipeline.run_pipeline` or `scripts/convert.py`.

HIER-006's separate default-off `structsplat.progressive_residual_quadtree` tests the opposite
construction direction under the same direct-additive semantics. Mask-present 64-pixel cells form
the coarse layer; a selected frontier parent stays bit-exact while all mask-present children append
fixed mask-moment geometry and signed RGB residual coefficients. Only the new coefficient block is
optimized against the detached prefix. Selection uses mask-aware smoothed residual energy per
appended row, and cold candidates must improve normalized raw pixel/7×7 violation before the SSE
tie-break (with an explicit float32 roundoff equivalence band). Every accepted stage is therefore
an independently renderable prefix, but the tree/coefficients-only size remains a non-codec proxy.
The frozen exposed C0001 diagnostic rejected this mechanism: the corrected 3,986/8,192-row prefixes
reached 27.805/32.882 dB and displayed pixel/7×7 maxima of 0.2223/0.0860 and 0.1073/0.0375, failing
the 0.02/0.01 gate. At 8,192 rows, 5,106 rows were retained ancestors and only 3,086 were level-0
leaves; the terminal worst boundary pixel remained in an unsplit level-1 cell. An identical CUDA
repeat preserved displayed gate metrics exactly and changed PSNR by only `1.7e-5` dB. This makes
the prefix-frozen literal Gaussian quadtree a negative control, not a replacement for HIER-005 or
evidence for a compression/default claim.

HIER-007's default-off `structsplat.artifact_first_quadtree` keeps that tree only as a scheduler.
An accepted split deactivates its active parent and activates all mask-present children, so active
keys remain an antichain that partitions the mask and inactive ancestors do not consume Gaussian
rows. Its two explicit axes rank splits by either smoothed residual energy or raw worst-pixel/7x7
artifact priority, and optimize either new RGB rows alone or those rows plus surviving
finite-support neighbors of the removed parent. Geometry and every nonlocal coefficient stay
fixed. Mask-normalized smoothed-error exposure supplies post-Adam row multipliers, while cold full-
field artifact/SSE comparison remains the transaction authority. Rejected batches halve
deterministically and rejected singleton parents are blocked.

The frozen exposed C0001 2x2 diagnostic rejects the proposed combined policy. At 8,192 active
rows, energy/new-only is the strongest HIER-007 arm at 40.035 dB and displayed pixel/7x7 maxima
0.0472/0.0215, but still fails the 0.02/0.01 gate and trails HIER-005's contextual passing 52.356
dB row. Artifact-first/new-only, energy/overlap, and artifact-first/overlap reach 34.569, 38.830,
and 26.035 dB; their local maxima also fail, with conspicuous quadtree-aligned defects in the
combined arm. That arm requires 1,773 trials for 279 accepted stages, leaves its terminal worst
pixel inside a level-3 cell despite 6,579 active level-0 rows, and takes 1,132.8 seconds in the
reference implementation. The result supports parent replacement as a better structural control
than retained ancestors, not artifact-first/overlap reconciliation as formulated. A successor
needs a commit-aligned smooth local objective, no-new-hotspot or Pareto constraints, a reserved
late repair budget, and likely a parent-to-children continuation before deactivation.

HIER-008's default-off `structsplat.overlap_elimination` factors actual pixel-neighbour support
against topology scheduling. A matrix-free normal-equation PCG first solves signed RGB for either
near-delta (`0.18 px`) or overlapping (`0.50 px`) pixel-centred peak-one Gaussians; source RGB is
never silently reused under the overlapping kernel. Its experimental WSE branch combines dynamic
density-adaptive crowding with a static same-side local Schur removal price and emits nested exact-
count survivor sets. A common bounded optimizer then moves every survivor's RGB, centre, and log
scale under smoothed-error, structure-feature, and top-tail pixel weights. Step zero is retained,
and a later checkpoint must lower SSE without increasing either raw worst-pixel or worst-7x7 error.

The frozen exposed C0001 2x2 rejects fixed-lattice/fixed-scale WSE-Schur elimination: even with
100% top-feature coverage within 1.5 px, its 8,192-row overlap cell reaches only 22.878 dB and has
visible dot holes. Meaningful overlap is nevertheless a useful factor for expanding quadtree
contraction. At 8,192 it raises the matched quadtree arm from 35.129 to 45.953 dB and the common
optimizer earns another 2.144 dB, but displayed pixel/7x7 maxima `0.1077/0.0253` still fail. At
4,096 it reaches 31.096 dB with visible ring/grid impressions. The exact overlap prefit is stable;
the WSE failure is instead a support-spacing mismatch after deletion. HIER-008 remains absent from
maintained dispatch, and its byte values remain uncoded proxies.

HIER-009 keeps the useful exact-overlap endpoint but replaces static elimination with HIER-005's
live contraction transaction. After every support-disjoint contraction batch, recovery optimizes
either topology-touched rows alone or those rows plus the direct 3x3 rounded-centre halo around
newly touched rows; only accepted changed neighbors persist into later checkpoints. Rows outside
that accumulated scope are a detached fixed base. An optional deterministic 5% feature reserve
keeps selected pixel-leaf means and covariances exact, permits local RGB refitting, and blocks only
regions whose protected multiplicity cannot fit the existing two-atom state.

The frozen exposed C0001 four-arm diagnostic finds that this neighborhood scope is useful only in
the aggressive regime as formulated. At 4,096 rows, overlap/halo gains 0.999 dB over
overlap/touched (40.801 versus 39.802 dB), lowers displayed pixel/7x7 maxima from
`0.0900/0.0334` to `0.0799/0.0278`, and removes the obvious block lattice. The protected variant
reaches 41.115 dB and a 0.0251 7x7 maximum, but every 4k arm still fails the `0.02/0.01` gate. At
8,192, the halo lowers the isolated maximum but loses 1.433 dB and worsens the 7x7 maximum, showing
error redistribution; protection partly recovers the loss, yet only the unchanged delta/touched
fallback passes at 52.338 dB and `0.0148/0.0053`. Protected geometry is exact, every checkpoint is
active, and target counts are reached. HIER-009 therefore remains default-off: retain the halo and
protection mechanisms for a local-artifact-aware successor, not as a production or compression
result.

HIER-010 is that bounded successor and remains outside maintained dispatch. Its first pass is
HIER-005's unchanged near-delta/hard3/touched trajectory. It remeasures the cold residual and
selects an exact count of source-pixel leaves from the pointwise maximum of q99-normalized pixel
MSE and mask-aware 7x7 mean MSE, using stable row-major ranking, radius-one NMS, and a deterministic
fill. A second identical contraction reserves those leaves without increasing the final row count.
`PixelContractionResult` exposes immutable touched/protected row masks aligned with the returned
field, so the final appearance stage can distinguish topology-changed rows from exact leaves
without inferring provenance from floating geometry.

`structsplat.contraction_refinement` then applies a task-scoped matrix-free PCG projection only to
touched, non-protected RGB rows. Means, scales, rotations, untouched leaves, protected leaves,
alpha, and topology are fixed. Sparse finite-support forward/transpose tile products mirror the
maintained additive kernel without allocating a dense pixel-by-row matrix. Step zero and every
iterate are measured; a candidate is selectable only when raw masked SSE and the displayed
pixel/7x7 normalized violation do not exceed step zero and coefficients remain bounded. The
lowest-SSE safe checkpoint wins, otherwise the field is returned unchanged. This narrow
contraction refinement neither implements FIT-046's general variable-projection decision nor
selects Field V2 semantics. HIER-010's exact-7k exposed-view report remains diagnostic and prices
its extra first pass explicitly.

HIER-034 adds an explicitly opt-in `CoefficientProjectionConfig.basis_cache`: `off` preserves
streaming; `scatter` owns detached finite-support triplets; `csr` stores both sparse directions.
`structsplat.additive_basis` caches only a fixed-geometry, fixed-mask linear RGB operator and must
be rebuilt after support, geometry, mask, or row changes. Its byte ceiling bounds retained
tensors, not construction workspace. Reduction-order differences are audited separately from
timing. HIER-033's `structsplat.pixel_gradient` is a diagnostic C0-faded direct-additive Jacobian,
local Gram, and split-Hessian reference; it is not a maintained renderer or topology policy.
These experiments do not change rendering or fitting defaults.

HIER-034 also has a separately frozen shared-resource correctness profile. It keeps the same
projection/matrix/numerical gates, records worker and parent-monitor GPU occupancy, and makes
timing eligibility unconditionally false. Its elapsed times cannot authorize acceleration;
the original timing source was later executed as its separately preserved complete assay.
Its observed foreign-resource activity and rollback-to-input passing cases prevent an isolated
or accepted-refinement speed interpretation; neither profile establishes general interchangeability.
The streaming baseline itself has local repeat variability, so caching is not an isolated cause
(ARA C72; [audit](../ara/evidence/overnight-method-research-2026-09-05/run.md)).

HIER-035's task-scoped `benchmarks.hier_additive_controls` compares parameter-group Adam with
diagonal and local-block Gauss–Newton updates under explicit bounds and charged backtracking.
It returns the exact terminal step and complete work trace, not a best checkpoint. This is a
procedural additive convergence assay; shared-workstation times are descriptive, not a speed
claim or a replacement for the maintained fitter.

HIER-036's `benchmarks.hier_coupling_oracle` is a bounded dense diagnostic of cross-Gaussian
Gram entries and row/shared trust caps. It reuses the same additive equations, parameter bounds
and terminal-step convention. Its64MiB limit covers retained image-Jacobian storage only,
with a separate256-parameter ceiling; neither is a peak-memory or production-scalability claim.
The factorial driver retains three Adam learning-rate controls and exposes every dense solve
and finite trial. No maintained fitter or default is changed.

The completed HIER-033–036 findings are scoped in ARA C68–C72 and the
[morning handoff](research/2026-09-05-overnight-findings.md): the finite selector misses its gate,
local curvature has mixed procedural outcomes, and full coupling does not establish texture
preference over strongest Adam. Near-ceiling overlap/easy-fixture gains are numerical polishing.
The projected-gradient rescue design remains unrun. The task-scoped evidence packager preserves
an explicitly partial, hash-bound archive without modifying any complete original report.

The [code-driven follow-up](research/2026-09-05-code-driven-portfolio.md) proposes two bounded,
default-off investigations: FIT-050 tests a fixed-geometry normalized color ray with actual-render
revalidation; PORT-007 tests same-call coverage and tail-statistic reuse in quality evaluation.
FIT-051's separate `actual_color_ray` module evaluates every trial with the maintained renderer,
including streaming gradient/Jacobi/CG proposals and a native color VJP. Streaming proposals are
explicitly approximate across backends; only actual images enter the unchanged reference gate.
The native VJP requests cloned color coefficients only, but charges the entire existing backward
invocation; it is not a specialized kernel. The task files own the prospective protocols. None
changes the normalized equation, training backward, maintained schedule, or default policy;
results and promotion remain pending.

The frozen C0001/C0004 diagnostic rejects the full composition. Projection alone is safe under its
frozen SSE/maximum-normalized-violation transaction but adds only `+0.0109/+0.0044 dB`. The
350-leaf residual reserve followed by projection loses
`0.1884/0.1848 dB` and raises masked MSE `4.43/4.35%`; C0004's local maxima improve, while C0001's
pixel/7x7 maxima worsen. Hard global reservation is therefore not a robust count-neutral policy:
it displaces capacity elsewhere, and a coefficient-only finish cannot repair that topology choice.
HIER-005 remains unchanged. Retain the projection as default-off implementation evidence and make
any future preservation/uncontraction decision inside a locally and globally Pareto-gated topology
transaction, tested on unexposed capture groups.

HIER-011 implements that count-neutral transaction as a default-off active-set oracle. It prices
the exact masked SSE cost of deleting each finite-support row, fits one-column signed residual
atoms at stable high-error sites, and admits only support-disjoint enter/leave pairs with negative
reduced cost. The maintained cold additive renderer is authoritative: every commit must strictly
lower raw SSE while individually preserving displayed worst-pixel and worst-7x7 maxima. Entering
rows are locked, row count and field semantics never change, and the operator remains sparse rather
than materializing a pixel-by-row matrix.

The exposed exact-7k report confirms that this transaction repairs HIER-005's C0001 local failure
and improves both views, but it does not pass its frozen material-gain rule. The search exhausts
safe improving pairs after 68/5 exchanges; exchange plus touched/new-row projection gains
`+0.5416/+0.0799 dB` and lowers C0001/C0004 pixel maxima to `0.0136/0.0093`. C0004 remains below
the declared `+0.10 dB` floor. HIER-011 is therefore a useful topology/local-artifact control, not
maintained dispatch or a promoted successor.

HIER-012 isolates the larger bottleneck: HIER-010's touched-row coefficient mask. It reuses the
same fail-closed sparse-tile PCG but marks all 7,000 RGB rows trainable while keeping means,
log-scales, rotations, support, filtering, alpha, topology, and count bit-exact. On the same exposed
views, direct global projection from the HIER-005 field reaches `52.3345/56.4702 dB`, gains
`+2.2375/+2.0961 dB`, reduces masked MSE `40.26/38.29%`, and passes both displayed local gates.
It also has lower MSE than first applying HIER-011 exchange and then the identical global solve,
although exchange-plus-global retains better C0001 LPIPS and isolated-pixel max. The simpler
HIER-005-plus-global-projection composition is the selected development pipeline for a future
clean independent screen. Because both views informed the choice, it remains outside maintained
dispatch and does not pre-empt FIT-046 or BENCH-020's semantic/work decision.

HIER-013 runs that frozen screen on all 16 requested repository COCO/DIV2K images with three CUDA
replicates and reverses the development selection. Direct global projection gains only
`+0.0117 dB`/`0.269%` geometric-mean MSE and activates on two images: 42/48 cells exceed the
coefficient limit 16 and fail closed at step zero. Exchange plus global is stronger but still only
`+0.0725 dB`/`1.655%`, costs more topology work, and leaves visible lattice artifacts. Moreover,
141/192 cold-versus-in-memory maintained renders exceed the frozen `2e-6` parity threshold, so the
diagnostic bundle is not claim-ready. Global projection remains a conditional, default-off solver;
the next formulation must prospectively bound or stabilize incoming coefficients rather than
raising the cap on these consumed images.

HIER-014 tests that numerical explanation on four SHA-bound Kodak development images.  The
backward-compatible solver can now restart at zero, regularize toward zero, and render the frozen
base directly.  This successfully collapses unsafe near-null coefficients on three images (for
example `183.56` to `7.81`) with sub-micro-unit maintained parity, but it changes geometric-mean
MSE by only `-0.721%` (`+0.0314 dB`) and worsens mean LPIPS.  One image still returns the exact
unsafe fallback because every lower-SSE bounded endpoint slightly worsens the displayed local
guard; another accepted endpoint worsens the 7x7 maximum.  Subtractive and explicit-base results
are effectively identical.  The fixed-geometry gate therefore fails and the HIER-013 replay stays
closed.  Coefficient conditioning is reusable opt-in numerical infrastructure, not the missing
general image-quality mechanism; a successor must change geometry/basis or dispatch to a stronger
fixed-count representation under a newly frozen comparison.

HIER-015--020 then isolate the exact-7k normalized alternative and its residual failure. Bounded
additive geometry relaxation remains visibly latticed; direct normalized fitting is much stronger
and visually clean but can leave isolated low-coverage maxima. `normalized_refinement.py` provides
the default-off fixed-geometry RGB-tail control, while configurable normalization epsilon, counted
background rows, and `tail_recovery.py` test lower floors, broad coverage, same-field priors, and
the explicit pointwise-safe SST1 coordinate payload. None is a general repair: HIER-016--019 fail
their fresh gates, and HIER-020 stops at 14/16 repository-test images because LPIPS correctly rolls
back required isolated substitutions.

`source_patch_tail.py` is HIER-021's separate default-off residual representation. The encoder
finds `D < 1e-8` sites in an unchanged normalized field, expands them by Chebyshev radius 3, and
stores exact source RGB8 only where the displayed value differs and raw pointwise SSE strictly
improves. Canonical SPT1 uses a 16-byte raster header plus sorted seven-byte
`(flat_index, R, G, B)` records. Decode renders the unchanged field once and applies those records;
it never reads the source image. A whole-image transaction retains ordinary mode unless MSE,
pixel/7x7 maxima, MS-SSIM, and LPIPS are noninferior.

The source-bound diagnostic passes a four-image prospective safety screen and a frozen no-refit
24-prior-plus-16-repository-image replay: 24/40 replay fields select 20,137 records/141,343 raw
side bytes and all nine recorded HIER-005-relative local failures are repaired. This result belongs
to the **field plus explicit source-RGB sidecar**, not the 7,000-Gaussian field alone. NPZ+SPT1 is
reference accounting rather than a complete codec, and one dirty-source seed plus producer review
does not authorize maintained dispatch or an “everywhere” claim. See the
[HIER-015--021 evidence and results audit](../ara/evidence/hier015-hier021-exact7k-portfolio-2026-08-10/run.md).

`additive_continuation.py` is HIER-022's default-off training diagnostic. It composes two exact
additive accumulations with learned positive masses, anneals their normalized quotient to the
numerator, and persists only the resulting ordinary `GaussianField`. Endpoint/parity tests pass,
but the frozen COCO4x2 diagnostic rejects the mechanism: coverage weight `0.05` reduces mean
coverage MSE by 97.3% while the mass-free endpoint trails plain additive fitting by `0.454 dB`,
worsens LPIPS and both local maxima, and costs `2.25x` fit time. Learned coverage mass is therefore
not a maintained representation component or pipeline path. A successor must use a new task and
data selection, and must begin from the exact ordinary normalized equation rather than an
independently gauged surrogate.

`unit_gauge_continuation.py` is HIER-023's cleaner default-off successor. It calls the maintained
normalized renderer for a 35% hold, uses unit-mass numerator/denominator accumulations for a 15%
transition, and calls the maintained additive renderer for the final 50%; only exact additive-tail
states can persist. The valid DIV2K4x2 diagnostic proves path identity and endpoint integrity. The
no-reset arm reaches ordinary additive within `0.0326 dB` after only 250 exact-endpoint steps and
improves mean LPIPS, pixel/7x7 maxima, and PSNR-AUC, but it retains none of normalized rendering's
`0.6648 dB` mean advantage and one LPIPS cell fails. Adam reset is `0.0700 dB` worse. Unit gauge is
therefore useful optimization evidence, not a maintained path or proof that normalization is
dispensable at fixed count.

`endpoint_appearance_projection.py` is HIER-024's default-off fixed-geometry discriminator. It
adapts an additive `GaussianField` to the existing all-row matrix-free RGB solver, reconstructs the
result on the unchanged geometry, and applies a target-known fail-closed metric transaction. On a
new DIV2K4x2 screen, the solve gains `0.1300 dB` on ordinary-additive geometry and `0.1719 dB` on
unit-gauge geometry, but the projected fields differ by only `0.0105 dB` and remain `0.538 dB`
below normalized rendering. Local/per-cell guards fail. The wrapper is evidence infrastructure,
not a maintained fit stage; pure-additive successors must change the basis or topology rather than
retune coefficients on this consumed bank. HIER-029 adds an optional encoder-only boolean
evaluation mask: the matrix-free objective and safety measurements use active pixels, the operator
parity receipt compares that common active domain, and `reconstruction_raw` remains the actual
full-crop maintained replay. Omitting the mask preserves the historical solve exactly; neither path
persists mask state in the endpoint.

`folded_multiscale_additive.py` is HIER-025's rejected default-off basis test. It fits 16 counted
grid Gaussians to a factor-two low pass, fits 624 anisotropic WSE rows to the signed residual,
concatenates them, freezes only coarse geometry during a 100-step full-target polish, and removes
the training mask before returning one ordinary additive field. The complete remaining-DIV2K4x2
screen proves exact count, mask removal, payload purity, and one-pass parity, but loses `1.5542 dB`
to direct additive before projection and `1.4083 dB` after the identical RGB solve; perceptual,
local, AUC, and fine-detail-blur gates also fail. It is evidence that disconnected proxy-stage
training produces a worse finite span, not an available maintained hierarchy or proof that all
pure-additive fields require normalization.

`progressive_additive_capacity.py` is HIER-026's default-off capacity/topology discriminator. It
fits a shared full-target N=640 additive base, initializes 256 signed residual rows, jointly trains
all N=896 rows, and materializes both base and candidate as four-array one-pass additive fields by
stripping training-only scale caps. On prospectively bound official DIV2K validation pixels the
candidate and a cold N=960 control beat normalized N=640 in every/aggregate PSNR comparison and
improve mean structural/local metrics, but fail isolated LPIPS/local clauses and exhibit material
forest-detail smear. It is evidence that normalization buys row efficiency rather than exclusive
representability; it is not a maintained pipeline path or selected Field V2 count.

`residual_pursuit_additive.py` is HIER-028's default-off sparse allocation method. It accepts an
already projected pure-additive base, repeatedly selects the row-major highest raw-RGB-MSE pixel,
and appends one fixed 0.35-pixel isotropic Gaussian carrying that pixel's signed residual. Its
analytic construction is checked against the ordinary additive renderer, the base prefix remains
bit-exact, and the returned field persists only means, log-scales, rotations, and signed RGB. On a
prospectively bound official-DIV2K8x2 confirmation, N=960+64 passes every frozen quality and native
visual clause while a separately cold-fitted N=1024 control fails local robustness. This proves a
bounded pure-additive alternative at 1.60x rows, not a maintained pipeline/default, equal-rate,
full-resolution, or target-free encoder result. HIER-029 optionally restricts its encoder-side
argmax and reported residual maxima to a boolean selection mask while retaining the same complete
residual scan and four-array endpoint. On exposed Janelle C0001 at 1200x1038, full-frame pursuit
gains only `0.00476 dB` over its N=960 base and loses `2.85597 dB` to normalized N=640 amid a
pervasive additive hole lattice. Masked pursuit improves its additive base by `0.25367 dB` and
same-count cold control by `0.19945 dB`, but remains `2.60041 dB` below masked normalized and has
worse perceptual/structural metrics. The max-side-160 positive therefore does not extrapolate to
this resolution, and both mask hooks remain research-only. HIER-030 adds a default-off C0
support-fade option whose analytic update exactly matches the additive renderer and runs a
proportionally scaled N=4,375/6,562+438/7,000 ladder on the same exposed raster. Full-frame pursuit
reaches `35.00091 dB`, `+21.35407 dB` over HIER-029's literal N=1,024 result and `+1.34819 dB` over
normalized N=4,375, but cold additive N=7,000 is slightly better in PSNR/MS-SSIM. In the contained
arm, tail selection is eroded by the full support radius and all fields materialize certified
scales before discarding mask/cap state: every centre is inside and unit coverage/reconstruction
outside are exactly zero. The remaining foreground error is boundary-dominated, so this is
capacity and containment evidence rather than a selected full-resolution method. HIER-031 keeps
the count at exactly 7,000 and separates representability from appearance allocation. The exposed
C0001 mask contains ten pixels in three components with no legal centre under the ordinary
0.35-pixel scale floor; more ordinary rows cannot cover them. ADR-0033 therefore permits a frozen,
independently certified micro cohort during topology-free local recovery. The selected diagnostic
endpoint uses 910 such rows plus 6,090 ordinary rows, eliminates raw holes with exact outside-zero
support, and passes the frozen HIER-030 interior/detail guard. Untouched current-pipeline controls
are sharper but leave 933--955 raw holes. The hook and method remain default-off, source-exposed,
and unconfirmed; neither equal-error allocation nor automatic gains from later scaling are claimed.

CORE-016/ADR-0032 tests a different ownership boundary instead of another explicit-row
contraction. Its default-off `.sgdp` packet charges a conventional appearance payload, decodes it
into signed cardinal-prefiltered coefficients of a finite normalized Gaussian lattice, and stores
a separate exact-count Field V2 structural measure with zero RGB and nonnegative mass. The two are
presented to realtime-gs only as a pair: structural `GaussianObservationField` proposals plus an
appearance `ObservationQueryBackend`. Neither plane enters the maintained StructSplat pipeline,
and the realtime-gs checkout is not modified.

The exposed C0001 development pilot survives only its narrow systems killing test. The selected
complete packet is 3,896,344 bytes and is exact below display quantization at decoded pixel
centers; its paired backend has NumPy/torch parity and drives a synthetic two-view CompactCarve
smoke. The apparent 3.662x original-file ratio compares a full source frame with a crop packet; the
crop-local canonical-PNG ratio is 1.139x. Off-grid bilinear-control sampling still finds 3.784%
local-envelope escape and 0.0244% global range escape.

An exposed reduced-resolution downstream extension now propagates CPU structural metadata plus the
paired CUDA query backend through real 23-view CompactCarve and common 3DGS refinement. The retained
matched-cap v4 candidate uses 956,301 input bytes versus 3,850,647 for the existing RTGSV containers
(`4.0266x`), and both final models contain 10,000 Gaussians. It reaches 25.188 dB reporting
foreground PSNR versus 24.012 dB for control, with better MS-SSIM, LPIPS, and alpha IoU, and reaches
the control's terminal PSNR at step 500 versus 1,400. This does not make the end-to-end path faster:
candidate lifting and full training are slower and peak VRAM is higher. Native review still finds
soft silhouette halos, fine-detail blur, and sparse floaters; stronger all-run mask supervision and
a late fixed-topology polish lose more than their frozen quality guards permit. The result is
therefore development evidence, not a general compression, convergence, continuous-quality,
artifact-freedom, full-resolution, final-storage, or BENCH-019 result. The research boundary and
audit are
[`2026-08-06-codec-native-dual-plane-portfolio.md`](research/2026-08-06-codec-native-dual-plane-portfolio.md)
and
[`2026-08-06-codec-native-dual-plane-results-audit.md`](research/2026-08-06-codec-native-dual-plane-results-audit.md).

CORE-017 isolates that retained artifact mechanism without changing the packet. Sparse Field V2
mass still proposes source rays, but a placement-only backend reuses exact packet alpha as uniform
inside-mask support and lets CompactCarve's first-index `argmax` choose the first depth attaining
maximal multiview silhouette support. Codec appearance still owns radiance. An optional pass then
replaces only covariance/opacity with realtime-gs's local surface-cover reconciliation; it cannot
move centers, colors, lineage, or count. Both pieces are lazy, default-off research composition.

On a new exposed `frame_00009` 23-view/three-reporting-view fixed-5k diagnostic, alpha-shell
placement improves terminal reporting PSNR by 1.587 dB over ordinary interior consensus, raises
alpha IoU by 0.131, removes all 212.5M sparse-index pair evaluations from depth scoring, and reaches
the interior baseline's terminal PSNR at step 200 rather than 1,000. Adding surface cover gives
+1.381 dB, +0.140 alpha IoU, -0.00135 gradient MAE, and -0.0160 outside alpha versus baseline, but
does not remove trailing smear/double-silhouette structure or fine-detail blur. Cover without shell
placement loses 0.703 dB. The scalar gate passes and the mandatory visual gate fails, so the tested
alpha-shell route is retained only as a causal diagnostic and is not advanced to variable topology,
full-resolution confirmation, a maintained report, or any default.

CORE-018 then replaces mask support with a source-excluded depth posterior over packet-derived
DINOv2/local features. A best-two-view likelihood with an explicit dustbin chooses coarse/fine ray
depth, and independently solved candidate rays provide a reciprocal-consistency gate before
surface-cover covariance reconciliation. Sparse Field V2 still proposes rays and packet appearance
still supplies radiance; the posterior owns only geometry. The module and its public appearance-only
query seam are lazy, default-off research interfaces.

On the disjoint unmasked `karate/frame_00060` diagnostic, the no-reciprocal posterior starts 1.846
dB above interior consensus and has lower complete pretraining time, but it is 0.846 dB worse after
the 500-step fixed-topology prefix. It ends only 0.093 dB higher with worse gradient MAE and visibly
remains a translucent smeared volume. Median normalized posterior entropy is 0.960, mean selected
confidence is 0.0415, and median reciprocal support is zero. The complete reciprocal arm cannot
meet its frozen 75% primary-support floor and fails closed before optimization. CORE-018 is therefore
rejected as a pipeline: retain it only as a negative control, do not relax reciprocity on the
consumed scene, and require a spatially coherent depth/surface model for a successor. Exact evidence
and limitations are in
[`2026-08-06-core018-ray-posterior-results-audit.md`](research/2026-08-06-core018-ray-posterior-results-audit.md).

CORE-019 changes that failed geometry unit to a spatially coherent multiview depth field. A pinned,
lazy VGGT dependency predicts overlapping calibration-selected four-view groups from packet-decoded
appearance; one Sim(3) per group transfers scale, while the known cameras still own every output
ray. Robust depth fusion separates support, compatible occlusion, and free-space contradiction.
Hard feature anchors plus compatibility-aware dynamic WSE choose exactly 10,000 proposals, bounded
post-selection contraction moves only close cross-view duplicates, and fused-depth normals plus a
strictest-visible-camera footprint cap define surfel extent. The packet grammar, realtime-gs
optimizer, and supported conversion path remain unchanged.

The exposed `karate/frame_00005` four-arm diagnostic rejects this composition. The complete arm
changes the raw-known-ray tradeoff (+0.0102 MS-SSIM, -0.0206 LPIPS, and 904 fewer final rows, but
-0.3075 dB PSNR and worse gradient/p99 error), showing that support/WSE is active without producing
a uniform quality win. It starts 0.969 dB below ordinary interior consensus instead of the required
+2 dB, misses every fixed-prefix quality gate, never reaches the interior control's terminal PSNR,
and ends 0.928 dB below it. Replays flip the terminal raw/full PSNR ordering after density events.
Native reporting views contain broad gray sheets, radial streaks, floaters, black holes, and erased
detail after the common optimizer. The route therefore remains a default-off negative control; its
coherent field and compiler mechanisms are not a usable or supported geometry backend.

BENCH-020's default-off `benchmarks.field_semantics_factorial` controller now provides the sealed
selection boundary around that object: explicit semantic and alpha-policy records, fixed-row and
equal-canonical-raw-byte lanes, ordered-geometry prefix seals, three outcome-separated phases,
capture-cluster killing tests, replayable convergence histories, and a portable audited report.
It is an experiment substrate rather than a fitter or a decision. Direct, dual, and normalized
production work remains blocked until disjoint matched data, the BENCH-019 downstream response,
distinct protocol/results reviews, and sealed confirmation select one contract.

## Entrypoint (ADR-0025/0028/0029/0030)

`structsplat.pipeline.run_pipeline` is the maintained composition of the current best pipeline,
and `scripts/convert.py` is its sole supported conversion CLI. `PipelineConfig`'s defaults are the
measured recipe (C25/C50/C51/C52) plus the evidence-bound `0.75` px Janelle mask margin (C56).
They are deliberately *not* the conservative library default surface in `config.py`
(ADR-0009/0013). Passing a mask selects the arm and nothing else does:

```
image (+ optional mask)
      │
      ▼
scripts/convert.py → pipeline.run_pipeline
      ├── mask ──► masked arm: quadtree-WSE interior + boundary-tangent rows
      │                         (CORE-011), containment on (ADR-0017/0019)
      └── none ──► full-frame arm: same init, mask machinery degenerate
      │
      ▼
safe_schedule.run_safe_schedule
   bootstrap → coverage growth → detail growth → [boundary/general closure]
   → redistribution → polish
   → [optional --fine-detail: residual effective-support estimate → error-only births
      → fixed-topology convergence]
   → [optional --fine-detail-pursuit: deep high-pass/NMS births in 128-row waves
      → joint pursuit-color solve → remeasure to explicit detail targets]
   every optimizer block and topology proposal runs on a detached trial field and is committed
   only if a full-frame metric vector is Pareto-safe (FIT-023/024/025, ADR-0020..0023)
```

The full-frame arm degenerates rather than forks: `mask.signed_distance` clips an empty complement
to the image diagonal, so caps are inert and the boundary band is empty. Boundary initialization,
containment, losses, metrics, and proposals are disabled; count-matched general coverage/detail
proposals occupy the same closure slot and budget (ADR-0027). The full-frame arm has no benchmark
screen yet (BENCH-017).

The layered reference path below is what both arms are built from, and what `structsplat fit`
exposes directly as the knob-level research command.

## Pipeline
```
image (H,W,3) in [0,1]
      │
      ▼
structure_tensor.compute ──► StructureTensor{ lam1,lam2, across_edge_angle, coherence, energy, label }
      │                                   │                 │                         │
      │ energy                            │ eigenvectors    │ eigenvalue pattern      │
      ▼                                   ▼                 ▼                         │
density.py  ── pmf ──►  sampling.eliminate (WSE)  ◄── anisotropy_metric ◄─────────────┘
                          │  exact-N blue noise, density- & anisotropy-adaptive
      ▼
init.build_field ──► GaussianField{ means, log_scales, rotations, colors }   (RS params)
                          │
                          ▼
fit.fit  ──►  render.render (normalized weighted sum, differentiable)  ──►  Adam (L1 + SSIM)
                          │
                          ▼
pyramid.fit_pyramid: level 0 from image density; finer levels add Gaussians where the *residual*
structure tensor has energy (densification); append order = coarse→fine = LOD prefix.

pipeline.run_current_pipeline: frozen safe schedule; masked = boundary specialization,
unmasked = identical counts/stages with general closure and no boundary-specific work.
```

## Module responsibilities
- **NumPy, init-time, no autograd:** `structure_tensor` (selectable central/sobel/scharr operator;
  luma or Di Zenzo rgb color space), `density` (structure/gradient/variance/hybrid/uniform modes +
  the inverse-CDF warp for low-discrepancy samplers), `sampling` (WSE blue noise, Poisson-disk
  dart throwing, farthest-point, CVT/Lloyd, Halton, and opt-in terminal-set-preserving progressive
  WSE order), `config`, `mask` (CORE-010/011: exact separable EDT / signed distance / erosion /
  nearest-inside feature transform / boundary color dilation / smoothed-SDF boundary normals for
  mask-contained fitting and boundary coverage); `observation_field` is the separate default-off
  Field V2 semantic boundary and CPU oracle. It stores RS geometry, authoritative additive RGB,
  optional independent mass/filter/background/packed alpha/camera state, explicit renderer and
  coordinate semantics, and a lossless hashed reference container without importing torch.
  `codec_native_field` is the separate CORE-016/ADR-0032 packet/query oracle: Pillow owns the
  charged JPEG/WebP appearance payload, NumPy owns deterministic coefficient prefiltering and the
  finite Gaussian-lattice evaluator, and Field V2 owns only sparse structural proposals.
  `realtime_gs_adapter` imports torch and realtime-gs lazily to construct their required paired
  structural-field/query-backend view. Its placement-only alpha-support wrapper and the sibling
  `realtime_gs_surface_lift` module compose CORE-017's first-maximum shell and optional surface
  cover without importing either optional dependency at module import.
  `realtime_gs_ray_posterior` is CORE-018's separate source-excluded coarse/fine feature scorer,
  missed-observation posterior, reciprocal candidate filter, and exact-N lift. Its real disjoint
  diagnostic fails both construction and native visual gates, so it is a negative-control module,
  not a supported geometry backend. `realtime_gs_coherent_depth` is CORE-019's separate pinned-VGGT
  field, known-ray fusion/support compiler, feature-anchor/WSE selector, bounded contraction, and
  compatible surfel cover. Its packet-only field is spatially coherent, but the complete frozen arm
  fails step-zero, fixed-prefix, terminal-control, and native visual gates. All five modules remain
  absent from supported conversion.
  `pixel_contraction` is a separate HIER-005 research producer: its default leaf and contraction
  path is NumPy-only, while conversion to `GaussianField`, maintained additive rendering, and the
  optional recovery fits import torch lazily. The default `touched` recovery scope forms
  never-touched leaves as a detached fixed render. The alternative `all_error_weighted` scope
  materializes every active row, blurs residual MSE with mask-normalized Gaussian filtering,
  applies the additive color transpose once to obtain support-averaged row scores, and multiplies
  post-Adam updates rather than gradients so Adam's normalization does not cancel the weights.
  Both use bounded parameter trust regions, accept the best non-regressing step, and rebuild the
  stale proposal frontier after an accepted geometry update. `rescue_observation_field` is a
  separate topology-frozen fallback: it rejects non-signed or semantically unsupported fields,
  keeps the persisted base prefix bit-exact, fixes rescue means/scales/rotations, and trains rescue
  RGB only under a tail-aware residual objective and a worst-local-error checkpoint rule. The
  task-local diagnostics may apply an explicitly logged LANCZOS/nearest source-mask resize,
  preserve native-source and executed-source provenance, and plot quality, localized displayed
  artifact, payload-proxy, active-pixel-rate, timing, action, attribution, recovery, repair, and
  parity rows. `progressive_residual_quadtree` is the HIER-006 sibling: deterministic quadtree
  geometry and topology stay NumPy-first, while each newly appended RGB-only residual fit imports
  torch lazily. Its immutable-prefix and cold-rollback checks make structural failure inspectable,
  but retained ancestors are charged in full-field counts/bytes and its compact tree number is
  explicitly only a shared-geometry proxy. Those two producers remain absent from every
  current-pipeline/default dispatch. `artifact_first_quadtree` is the HIER-007 sibling. It replaces
  active parents rather than appending them, stores active RGB by node key, and optionally forms a
  differentiable local block from new children plus support-overlapping survivors. Its common base
  state is cloned across factorial arms; cold rollback checks topology/coefficient identity and
  active-frontier partition validity. Final-frontier and progressive coefficient-event byte
  ledgers remain non-codec proxies. HIER-007 is likewise absent from every maintained dispatch.
  `overlap_elimination` is the HIER-008 sibling: fixed-geometry forward/transpose products and PCG,
  structure/radius/Schur analysis, and WSE topology stay NumPy-first; torch is imported lazily only
  for the common bounded optimizer. It can feed solved full-lattice coefficients into HIER-005 or
  materialize fixed-lattice survivors as Field V2. Neither branch is a supported initializer,
  fitter, codec, or current-pipeline stage.
- **benchmark-only structural controls:** `structural_controls` lazily calls SLIC and keeps the
  SLIC/Sobel complexity ranking, exact-N 6:2:1 allocation, and unresolved upstream-fidelity
  assumptions explicit. `init` registers `local_slic_sobel_control`, but it is not a shipped
  default or an upstream-paper implementation.
- **torch, autograd:** `gaussians` (RS + optional opacity + optional per-Gaussian scale caps,
  ADR-0012), `render` (normalized default + additive, ADR-0006, exact CUDA variants, ADR-0011,
  and gsplat comparator, sharing one accumulator where semantics match), `metrics`, `init`
  (bridge; `build_masked_field` for CORE-010), `fit` (selectable loss/optimizer/LR-schedule/
  split-mode; opt-in mask containment via `_MaskConstraint` with isotropic ADR-0017 or certified
  anisotropic ADR-0019 caps, forced cap recertification for terminal and restored best-checkpoint
  states, under-coverage penalty, boundary tangent densification; ADR-0033's experimental
  `constraint_exempt_row_mask` preserves an independently certified fixed micro cohort through the
  ordinary scale floor only when the same rows are frozen and topology is disabled; opt-in
  FIT-022 coverage-matching regularizer — mass-neutral `(S−c)²` on the raw weight sum with
  detached opacities, feature/boundary/error targets and cosine decay), `pool` +
  `triage` (FIT-021/ADR-0020, opt-in via `triage_every`: fixed-capacity pooled row lifecycle with
  off-image parking, byte-budgeted capacity from `target_file_bytes`, and one in-place
  park→merge→split→spawn event replacing the independent topology timers); `pool` also provides
  FIT-024/ADR-0021's immutable active-prefix storage for `safe_schedule`, where preallocation is
  independent of topology policy, state checkpoints retain full field/Adam capacity, Adam update
  kernels use the active shape, and one terminal compaction restores the ordinary `GaussianField`
  interface. FIT-025/ADR-0022 separates that physical capacity from the ordinary active ceiling
  and adds an opt-in post-color-solve reserve whose covered-interior high-frequency births/splits
  remain transactional and Pareto-gated; FIT-026/ADR-0023 adds the opt-in `geometric` storage
  policy that grows physical capacity by `growth_factor` toward `capacity` on demand instead of
  preallocating it, preserving the live prefix so the fit stays bit-identical to `fixed_capacity`),
  `pyramid`, `pipeline` (CORE-012/ADR-0025: the single current-best recipe and matched
  masked/full-frame arm selection), `workflows` (ADR-0027: four folder/report orchestrators,
  registered ablations/stage variants, and optional native-baseline subprocesses),
  `codec`
  (post-fit quantization, ADR-0007; optional in-container alpha stream for masked inputs,
  ignored by pre-FIT-021 decoders).
- **read-only diagnostics:** `visualize` calls the production NumPy analysis/initialization and
  torch normalized renderer, then exports raw tensor/field/responsibility maps plus deterministic
  explanatory panels. It never fits or changes a field and is not benchmark evidence (DOCS-002).
  
  `viewer` bridges a `GaussianField` to the external igsv browser viewer (optional dependency)
  for live fit inspection via `fit(iteration_observer=..., observer_every=...)`; the embedding
  and its diagnostic-only status are ADR-0018.
- **entry:** `scripts/convert.py` is the sole current-best conversion CLI;
  `scripts/{benchmark,ablation,stage_search}.py` are evaluation workflows. All four write portable
  report bundles. `workflows.STAGE_VARIANTS` is the registry those workflows share; besides the
  recipe stages it carries `commit_gate` (BENCH-018's transactional block, applied uniformly to
  every gated phase and clamped to each phase ceiling) and `hole_budget` (FIT-028's ADR-0026
  interior coverage trade-off budget). Both register `current` first so the shipped recipe stays
  the baseline arm. Every current-profile run card also carries `gate_telemetry`: per-phase
  attempted/accepted steps, block counts, and the schedule's rejection-reason histogram, which is
  the measured surface FIT-028/FIT-029/BENCH-018 read.
  `scripts/check_report_bundle.py` is their standalone structural handoff gate:
  report-owned artifact paths serialize relative to the bundle, and the checker validates clean
  source identity, table agreement, finite metrics, contained artifacts/hashes, and portable
  local links before semantic results audit. `deprecated_scripts/` retains
  evidence-bound launchers without presenting them as supported interfaces (ADR-0028/0031, C61).
  `scripts/experiments/hier005_pixel_contraction.py` is instead a task-local diagnostic: it writes
  cold-rendered Field V2 rows, source/reference/raw byte ledgers, histories, and HTML, with explicit
  warnings that none of those byte references is a complete codec rate.
  `scripts/experiments/hier006_progressive_residual_quadtree.py` writes the corresponding generic
  workflow report for accepted hierarchy prefixes, including complete stage/checkpoint histories,
  hierarchy-depth maps, worst-error crops, count/quality/local-error/byte/time curves, dirty-source
  snapshots, and contextual HIER-005 rows. `check_report_bundle.py --allow-dirty` validates the
  diagnostic structure without promoting its exposed-image outcomes.
  `scripts/experiments/hier007_artifact_first_quadtree.py` writes a shared-base 2x2 report for
  selection and reconciliation scope, with active-frontier depth maps, full/worst-crop visuals,
  snapshot/stage/checkpoint/attempt curves, cold fields, and separate active/frontier/event byte
  ledgers. Its corrected packaging-only bundle preserves the original field/metric execution while
  restoring contextual HIER-005 rows omitted by the executed status filter.
  `scripts/experiments/hier008_overlap_elimination.py` writes the overlap-support x scheduler 2x2,
  including exact-prefit and Schur receipts, optimizer attribution, feature/centre/worst-error
  visuals, every numeric snapshot/checkpoint curve, cold fields, and separated native/evaluation
  byte ledgers.
  `scripts/experiments/core016_codec_native_field.py` is the bounded codec-native diagnostic. It
  writes exact packet/component ledgers, pixel/off-grid/structural/query metrics, generic numeric
  curves, worst crops, source snapshots, and contextual controls. Its custom manifest is internally
  hashed but is not a schema accepted by `check_report_bundle.py`; it must not be called a
  maintained portable report. `scripts/experiments/core016_multiview_downstream.py` is the bounded
  source-grounded follow-up: it separates reporting-only cameras, snapshots both repositories,
  builds complete candidate packet ledgers, drives the paired CPU-metadata/CUDA-query interface
  through CompactCarve and 3DGS, and emits checkpoint curves, models, visuals, and explicit scalar
  plus manual gates. Its `surface2x2` profile is CORE-017's one-shared-packet placement/covariance
  factorial. Its schema is likewise task-local and not accepted by the maintained report checker.
  `scripts/experiments/core018_ray_posterior_downstream.py` writes CORE-018's one-seed disjoint
  three-arm packet/feature/geometry diagnostic. Its reciprocal arm fails closed, its two rendered
  arms fail native review, and its custom partial-result schema is intentionally diagnostic rather
  than accepted by `check_report_bundle.py`.
  `scripts/experiments/core019_coherent_depth_downstream.py` writes CORE-019's one-seed four-arm
  packet/coherent-field/geometry diagnostic with immutable JSON/JSONL/CSV metrics, native visuals,
  models, a replayable scalar decision, and an explicit `claim_ready=false` manifest. Its custom
  schema is accepted by `check_report_bundle.py --allow-dirty` for portable diagnostic handoff only;
  the checker cannot convert its exposed-scene visual failure or dirty external dependency into a
  scientific claim.
- **agent workflow:** `tasks/INDEX.md` and task files are the work authority, while
  `tasks/SESSION-BRIEF.md` is a deterministic derived view. `scripts/check_task_policy.py`
  validates dependency and review state; `scripts/check_agent_workflow.py` checks agreement among
  the guides, harness configuration, skill mirrors, verification/CI spine, generated brief, and
  PR contract (ADR-0031, C61).
- **entrypoint:** `pipeline` (CORE-012/ADR-0025/0028) owns the maintained best-pipeline recipe and
  the masked/full-frame arm selection; it composes `init`, `fit`'s mask constraint, and
  `safe_schedule`, and holds no fitting mechanism of its own. `safe_schedule` (FIT-023/024/025)
  owns the phase order, the topology auction, and the Pareto-safe commit gate; ADR-0020..0023 own
  its storage policies. FIT-031/ADR-0029 adds the default-off terminal error-only tail: foreground
  MAE effective support estimates demand, half is requested as small isotropic residual-ranked
  rows in bounded batches no smaller than `event_min_count`, and the expanded field converges
  under the unchanged Pareto gate.
  FIT-039/040/ADR-0030 add a mutually exclusive, default-off masked pursuit tail: deep high-pass
  residual sites receive ordinary 0.35-pixel rows, all accumulated tail colors are jointly solved
  while inherited rows stay frozen, and the stage stops at explicit high-pass/Laplacian targets
  under the same protected gate.
- **entry:** `cli` (`structsplat fit` /
  `image-to-gaussians2d`, `render` /
  `gaussians2d-to-image`, `batch-fit`, `ablation`, `stage-search`); `render` cold-loads a native
  full-precision NPZ or self-describing SSPL1 stream and can emit display-referred error/metrics
  plus a read-only fitted-field ellipse overlay. `batch` (PORT-005) runs the `fit` option surface
  across worker processes with device round-robin and a resumable `metrics.jsonl`. The optional
  `fit --live` path remains diagnostic-only.

- **decision benchmark:** `benchmarks.actual_rate_phase_diagram` owns frozen actual-rate manifests,
  SSPL1 cold scoring, exact-cap RDO/statistics, and result figures for BENCH-007. Its manifest
  distinguishes the normalized weighted-sum equation from the selected implementation; native
  scientific runs may freeze the parity-checked owned exact-CUDA implementation explicitly.
  Persisted-stream parity is checked on decoded field state before a single cold render; two CUDA
  renders are not used as an equality oracle because atomic accumulation is not bit-reproducible.
  Result-figure stream replay uses the validated analysis device, so CUDA-frozen semantics are not
  silently forced through CPU tensors. The completed Stage-1 gate is negative; this substrate is
  reusable, but Stage 2 is not authorized for the current tensor-WSE claim.
- **cross-repository objective gate:** `benchmarks.stage1_downstream_objective` (BENCH-019) is a
  passive, hash-bound adapter around realtime-gs execution. It freezes the exact field equations,
  field/source/camera manifests, clean commits and environments, splits, seeds, downstream
  schedule, metrics, missing policy, and decision thresholds before outcomes exist. It never
  converts a normalized field into additive arrays. Exported cells must preserve the field
  semantic digest and one family-independent downstream-factor digest; an A/A replay must pass
  before frame-ranked, capture-clustered correlations can influence Field V2. Its portable report
  is a second schema accepted by `scripts/check_report_bundle.py`; this does not make a diagnostic
  or underscoped run claim-ready. The realtime-gs side now has a driver-handoff passive exporter
  checkpoint: it derives the shared factor, extracts untransformed finite metrics from sealed JSON
  pointers, binds all six cell artifacts, and requires each assembled row's provenance receipt.
  That external checkpoint remains pending distinct review. Its source-only 3+3 portfolio records
  acquired groups and closed gates; it is not matched field data or a frozen execution protocol.
- **task-scoped contraction diagnostics:** the HIER-005 contraction/bounded-repair and HIER-008
  overlap-elimination reports use explicit non-claim schemas accepted by
  `scripts/check_report_bundle.py`. Their gate checks the
  exact file manifest, source snapshots, JSON/JSONL/CSV row agreement, finite metrics, displayed
  artifact-gate arithmetic, field hashes, cold-render parity, curve inventory, and portable links
  to every field/history/image/row artifact. Structural acceptance preserves the diagnostic scope;
  it does not turn an exposed-image, dirty-source run into claim-ready evidence.
  HIER-009 through HIER-013 use the same dynamic-diagnostic artifact contract for neighborhood
  recovery, residual anchors, exact-count column exchange, global appearance projection, and its
  repository-image transfer screen; schema registration validates handoff structure only and does
  not broaden their evidence class.

## Stage-search (ABL-002, protocol in ADR-0010)
`benchmarks/stage_search.py` sweeps configurations across every swappable stage — tensor operator,
tensor color space, density mode, sampling mode, orientation mode, init strategy, color mode,
scale mode, opacity, renderer, loss, optimizer, LR schedule, factored refinement
(`refine_site`, `refine_primitive`, `refine_nms`, sampled-add score, plus
the opt-in normalized-responsibility mass exponent and color/prune/relocate flags), pyramid — in
two modes:
**factorial** (full product, ranked, for the best complete config) and **influence**
(one-factor-at-a-time paired deltas vs the baseline = first value of each axis; emits
`influence.md` with ΔPSNR/ΔMS-SSIM/ΔAUC/Δiters-to-target/Δseconds per stage option). Configs
whose differing stage is provably inert are canonicalized and deduplicated. Every row records
quality (PSNR/MS-SSIM/LPIPS), convergence (iters-to-target, PSNR-AUC), and speed (init/fit
seconds) so max-quality, max-convergence-rate, and max-speed candidates can be read from the same
run. The shipped defaults (ADR-0009 plus ADR-0013's init-default update) are one named cell in
that space; everything else is a candidate the screening can promote. `benchmarks/ablation.py`
(ABL-001) stays the focused
init-strategy × budget sweep.

## Performance notes (reference is the oracle; these keep it usable at N~20k on CPU)
- `pixel_contraction` stores at most one float32 atom slot per active source pixel and reuses slots
  after every contraction. Quadtree cells hold at most two resolved output ids; a ready cell has at
  most eight active atoms. A cheap RGB proxy keeps the image-sized frontier lightweight, exact
  Gaussian-product options are cached only after shortlist entry, and exact discrete fits are
  invalidated only by overlapping accepted support boxes. Support-disjoint actions can commit in
  one batch. This is a CPU reference design, not a PORT-006 acceleration result.
- `progressive_residual_quadtree` never renders an accepted prefix during a child optimizer block;
  it caches that image and renders at most `max_rows_per_stage` new rows per step, then performs one
  cold joint render for acceptance. The HIER-006 C0001 diagnostic completed its hierarchy build in
  about 7.3 seconds on the recorded RTX 3050, but that speed is not a competitive result because
  the artifact gate and quality control failed.
- `artifact_first_quadtree` caches the current full render and differentiates only a trial's local
  RGB block, but its current reference still rebuilds/cold-renders the complete canonical frontier
  for every transaction and recomputes mask-moment support metadata during overlap discovery. The
  C0001 arms take 357--1,133 seconds; this is a correctness/mechanism oracle, not a production-speed
  path. Cache/fusion work is justified only after a revised policy passes its quality gate.
- `overlap_elimination` applies each compact isotropic stencil by image shifts and uses the same
  operator for the transpose, so PCG never forms a pixel-by-row matrix. The C0001 overlap prefit
  converges in 22 iterations/1.29 seconds and feature elimination in about 1.56 seconds, but those
  dirty single-image timings are reference telemetry rather than a PORT-006 result.
- `sampling.eliminate` builds the WSE conflict graph vectorized over grid-cell offsets (only the
  greedy heap removal stays in Python); the anisotropic search reach is bounded per receiver by the
  metric's minimum eigenvalue, so no long-range along-edge conflict is missed. ~30x faster than the
  original per-pair Python loops at N=20k.
- `render` evaluates each Gaussian on the axis-aligned bounding box of its `sigma_cutoff` ellipse
  (per-axis radii `(rx, ry)`), laid out as one ragged flat tensor — no padding to a shared square
  tile. Elongated anisotropic Gaussians get a tight rectangle instead of a square sized by the major
  axis (~3x forward speedup on a flanking init). Still fully differentiable; radii stay detached.
- `render`/`conics` take an optional EWA-style `aa_dilation` (Sigma + d·I) low-pass for sub-pixel
  Gaussians — off by default; exact under RS since it only shifts the per-axis variances.
- `renderer=cuda` and `renderer=cuda_additive` call StructSplat's owned exact CUDA extension for
  the same clipped-support equations. The internal `cuda_block_reduce` selector preserves the
  exact forward equation and replaces only the untiled backward reduction; PORT-004 keeps it
  benchmark-only after the frozen all-grid/stability gate failed. `renderer=gsplat` is kept as a
  separate alpha/sum comparator because it is not numerically equivalent to the normalized
  reference.
- `renderer=cuda_tiled` (opt-in, PORT-002/003, locally parity-validated but performance-unmeasured)
  builds its tile index inside the
  extension (CUB radix sort over packed 32-bit keys; stable, so the index is deterministic),
  stages Gaussians through shared memory in both tiled kernels, warp-reduces backward gradients
  before atomics, and — under `support_fade` only — exactly culls (tile, Gaussian) pairs whose
  weight is provably zero via a closed-form conic-over-rectangle minimum. Semantics are
  unchanged; `benchmarks/tiled_render_profile.py` owns the preregistered acceleration gate, and
  `cuda` remains the shipped GPU default until that gate passes on hardware.
- `scale_cap_mode=feature` gives each Gaussian a local support ceiling from the structure tensor's
  feature run length. `scale_cap_mode=feature_rel` instead derives the cap from local density
  radius / quadtree leaf side with separate along/across multipliers. The fitter clamps optimized
  scales to the field-owned cap, preventing long edge spikes without changing the renderer
  equation. Both cap modes are searchable and default off after INIT-008's fair-density negative.

## Extension seams
- Init strategies: `init.STRATEGIES` (the ablation variables).
- Renderer variants (e.g. additive for AIR-style residuals): behind ADR-0006, keep reference oracle.
- Performance: `PORT-001` CUDA tile rasterizer → IntrinsicEngine RHI pass; reference stays the oracle.
- Feed-forward init predictor (`FF-001`) and compression codec (`COMP-001`) attach after the fitter.
