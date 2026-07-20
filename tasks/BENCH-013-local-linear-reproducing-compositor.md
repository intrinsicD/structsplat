# BENCH-013: Local-linear reproducing Gaussian compositor

## Status

Stage-0 protocol frozen before scientific outcome inspection (2026-07-16). A read-only
implementation preflight found that the first written version ambiguously passed float32-rounded
colors to the `1e-10` exact-double affine theorem check, guaranteeing a false failure from storage
rounding. Before the 108-cell early screen, the protocol was clarified to use exact analytic colors
for mathematical reproduction and a separate promoted-float32 same-state parity oracle; no gate,
threshold, geometry, target, count, or seed changed. Benchmark-only. Natural-image Stage 1, a
production renderer, CUDA implementation, fit integration, codec syntax, defaults, and method
claims remain locked unless every Stage-0 gate passes unchanged.

The complete early-screen artifact passed before any downstream cell was evaluated. A second
read-only hostile preflight then bound the downstream path roles, permutation semantics, gradient
construction, and no-short-circuit rule below. These clarifications change no scientific data,
seed, threshold, or gate; they remove implementation degrees of freedom before outcomes exist.
The full downstream artifact must bind both this clarified task and the immutable early-screen
parent, whose task SHA-256 is
`cc5506f1536771d77a8ade9e8c076520d772d984287124db70d0c5c3ae8aeb11` and whose binding SHA-256
is `394323650251ba129eab8b5e03ac2139ab2e348e1571fc4863646b3c777c13ab`.

A third read-only preflight found one analytically degenerate gradient comparison before any
downstream outcome existed. Exact analytic affine colors make the log-scale and rotation reference
gradients structurally zero because those blocks change weights but cannot change an exactly
reproduced affine field; the registered promoted-float32 colors make these blocks only near-null
after per-sample storage rounding. The original full-block relative rule divided by
`max(norm(g64),1e-12)`, so its
`2e-3` threshold demanded about `2e-15` absolute float32 agreement on the null blocks, contradicting
the already frozen directional tolerance's `2e-5` absolute term. The full-block gate below was
therefore repaired pre-outcome to use that same mixed absolute/relative form; the old relative-L2
quantity remains a diagnostic. No cell, permutation, or selected gradient outcome had been run or
inspected when this validity repair was frozen.

### Post-execution lifecycle amendment for the v3 rerun

The first full downstream attempt completed the registered raw forward, effective-weight, and
permutation matrices, then failed its artifact lifecycle before it could write a completion marker,
analysis, or manifest. Its built-in replay recorded 13/14 integrity checks passing; the only false
check was the aggregate forward raw replay. Final-summary serialization then failed because a
closed NumPy archive handle had shadowed the source-archive metadata record. The immutable failed
attempt is `results/bench013_local_linear_stage0_full_2026-07-16`, binding
`6817380d2599ff2d5d8a0a22d244b36fb5e9cb76ae4b220777d40d87b6a5e7d4`. Its inventory digest is
`111c0680df375e753ef6578847e1b80769e4e5f1a9bed09ebb9e5c9f05231ab9`. Compute it by sorting the
descendant files using Python `Path` component ordering, then serializing each as the UTF-8 record
`relative_posix_path<TAB>decimal_size<TAB>sha256`, joined by line feeds with no trailing line feed.
That attempt is unavailable evidence and must never be completed, patched in place, or used to
authorize Stage 1.

Lifecycle diagnosis did not inspect aggregate quality metrics or interpret the scientific gate
ledger, but it unavoidably exposed that the preregistered gradient audit was not reached because its
base forward cell failed. The v3 repair therefore changes no target, field, count, seed, order,
path role, renderer calculation, raw science-row schema, gate, threshold, conditional gradient
rule, or stop rule. It only (1) prevents the source-metadata name collision, (2) binds float32
condition numbers to covariance-verified persisted eigenvalues without amplifying independent
eigensolver rounding, (3) checks the two persisted 2 x 2 solves with the fixed componentwise bound
`|Ax-b| <= 128*eps32*(|A||x|+|b|)`, and (4) advances top-level lifecycle schemas to v3. The prior
clarified task SHA-256 was
`9d74511c95786592b5071b5f9b61434983b95bb0a28a97377903b45327cebb09`; the v3 runner must bind
this amended task, preserve the failed attempt byte-for-byte, run into a new directory and binding,
and prove science-bearing NPZ and binding-normalized ledger equivalence before interpreting the
rerun.

## Question and claim boundary

StructSplat's normalized compositor is a local-constant Nadaraya--Watson estimator. BENCH-013 asks
whether the same Gaussian centers, compact supports, and weights can instead form a local-affine
reproducing operator without storing any extra per-Gaussian state. For a pixel `x`, define

```text
L   = max(H - 1, W - 1, 1)
d_i = (mu_i - x) / L
z_i = [1, d_ix, d_iy]
M   = sum_i w_i z_i z_i^T
B   = sum_i w_i z_i c_i^T
y   = e_0^T M^-1 B
```

Equivalently, with normalized weights `p`, weighted offset mean `m`, covariance `C`, color mean
`cbar`, and cross-covariance `G`, `y = cbar - m^T C^-1 G`. A full-rank solve reproduces affine
sample fields and removes the local-constant boundary bias. It also introduces signed effective
weights, ill-conditioned moments, 15 accumulation channels, and a per-pixel solve.

Local-linear regression and moving least squares are prior art. The only potentially new evidence
is the recipient-specific state/work question: whether this order-elevated compositor is useful in
a learned 2-D Gaussian image representation. Stage 0 is an availability/robustness screen, not
quality, convergence, performance, compression, expressiveness, or publication evidence.

## Literature boundary

- Fan, [*Design-Adaptive Nonparametric Regression*](https://doi.org/10.1080/01621459.1992.10476255),
  establishes the local-linear regression and boundary-adaptation precedent.
- Lancaster and Salkauskas,
  [*Surfaces Generated by Moving Least Squares Methods*](https://www.ams.org/mcom/1981-37-155/S0025-5718-1981-0616367-1/S0025-5718-1981-0616367-1.pdf),
  analyze moving least squares as a projection method.
- Liu, Li, and Belytschko,
  [*Moving Least-Square Reproducing Kernel Methods (I)*](https://www.sciencedirect.com/science/article/pii/S0045782596011322),
  provide the polynomial-reproduction/RKPM precedent.
- Positive local maximum-entropy coordinates are not the selected mechanism: affine-exact positive
  coordinates require the query inside the active centers' convex hull. Current compact supports
  violate that assumption near boundaries unless ghosts or a new support grammar are added.

The novelty class is therefore `known components, possibly new recipient relationship`, with no
claim that the mathematical operator itself is new.

## Frozen Stage-0 renderer semantics

Use the exact current clipped AABB contributor rule: rounded float32 means, production integer
radii, `sigma_cutoff=3`, `support_fade=false`, `opacity=None`, and `aa_dilation=0`. The weight is
`exp(-q/2)` at every pixel in the AABB; do not add a `q <= 9` ellipse mask and do not drop small
positive weights. All `64 x 64` pixels are scored.

The solve has no denominator epsilon, ridge, pseudoinverse, determinant floor, clipping, fallback,
rank-adaptive order, ghost sample, enlarged support, added center, invalid-pixel exclusion, or
alternate boundary rule.

## Frozen targets

Let `u=x/63`, `v=y/63`. Evaluate the formulas analytically both at pixels and at continuous
Gaussian means. No antialiasing or output clipping is legal.

1. `constant`: `(0.25, 0.50, 0.75)`.
2. `affine_xy`: `(0.10+0.35u+0.25v, 0.80-0.30u-0.20v, 0.20+0.20u+0.40v)`.
3. `quadratic`: `(0.15+0.45u^2+0.15v, 0.75-0.30v^2-0.15u, 0.20+0.40uv)`.
4. `vertical_step`: use `(0.15,0.30,0.75)` for `u<0.5`, else `(0.85,0.70,0.20)`.
5. `diagonal_step`: use `(0.10,0.65,0.25)` for `u+v<1`, else `(0.90,0.20,0.75)`.
6. `checker8`: parity is `(floor(x/8)+floor(y/8)) mod 2`, using the vertical-step colors.

Evaluate every `64 x 64` pixel formula in float64 first, persist/hash that exact-double array, then
cast once to a C-contiguous float32 array for production `build_field` and hash it separately. At
continuous promoted-float32 Gaussian means, likewise evaluate sample colors in float64 once; cast
that array once for the stored production field and retain the exact-double array only for the
mathematical oracle. Do not rely on NumPy expression dtype inference or evaluate the formula
independently in float32.

## Frozen geometry matrix

Counts are `{28,56,112}` and seeds are `{0,1,2}`. These are the `64^2` area-scaled counterparts
of the locked Stage-1 counts `{175,350,700}` at `160^2`.

Use production `build_field` and an explicit `quadtree_wse` configuration:

```text
candidate_oversample=6, density_base=0.05, density_power=1,
density_mode=structure, sampling_mode=wse, wse_progressive_order=false,
max_axis_ratio=6, coherence_power=1, orientation_mode=tensor,
scale_mode=spacing, init_scale_mult=1, scale_cap_mode=none,
background_fraction=0, background_grid=0, flank_offset_frac=0,
color_mode=bilinear, opacity_mode=none
```

Use the explicit default structure-tensor settings `grad_sigma=1`, `tensor_sigma=2`, central
gradient, luma, `flat_frac=0.02`, and `corner_frac=0.15`. Overwrite initialized colors with the
analytic target values at continuous means.

Both cohorts are conjunctive:

1. `target_conditioned`: build geometry for every target/count/seed.
2. `shared_constant`: build geometry from `constant` once per count/seed and reuse it for all six
   targets.

Persist the production float32 field, its float32-derived conics, and integer radii once. Never
regenerate geometry or radii from the target. The mathematical float64 reproduction oracle
promotes stored float32 means/log-scales/rotations and recomputes conics in float64, while using
exact-double analytic sample colors. The distinct same-state arithmetic oracle promotes the exact
stored float32 means, float32-derived conics, and float32 colors to float64. This separation stops
float32 color-storage rounding from contaminating the `1e-10` mathematical reproduction theorem,
while the parity test still charges every rounded float32 input. The shared cohort prevents
target-conditioned geometry from acting as a favorable leaked preconditioner.

The exact matrix is 108 cells: 54 target-conditioned (`6 targets x 3 counts x 3 seeds`) plus 54
shared-geometry evaluations (`9` constant-derived base fields times six targets), for 63 logical
field records and 442,368 base pixel records. Because initialization is deterministic, the nine
shared constant-derived geometries are expected to be byte-identical to the corresponding nine
target-conditioned `constant` geometries; therefore only 54 distinct geometry content hashes are
expected, not 63. Persist every logical record and every target-specific float32 color-array hash.
The shared cohort changes only analytic colors; its means/log-scales/rotations/radii must be
byte-identical across targets. `flank_offset_frac=0` does not remove the initializer's internal
`edge_w` floor, so actual generated geometry—not an assumed lattice—must be persisted.

## Numerical paths and per-pixel diagnostics

Require weight mass `s0 >= 1e-5`, at least three active contributors, and reference rank three,
where `sigma_3 > max(n_active,3) * eps64 * sigma_1` for
`A_i=sqrt(p_i)[1,d_ix,d_iy]`. Separately require `kappa_2(C) <= 8192`. Store mass, active count,
all design singular values, all `C` eigenvalues, condition number, and worst-pixel coordinates.
`n_active` is exact AABB membership from `_tile_coords`, even when a float32 Gaussian weight
underflows to zero at a rotated AABB corner; mass and the weighted design still use the numerical
weights.

The mathematical float64 oracle uses reduced Householder QR on `A` and
`Y_i=sqrt(p_i)c_i`, with exact-double analytic colors, followed by a triangular solve. The
same-state float64 oracle repeats QR using the promoted float32-derived conics and colors. SVD is
rank diagnostics only. Effective weights are independently computed via `R^T R h=e_0` and
`phi_i=p_i z_i^T h`; record `sum(phi)`, `sum(phi d)`, and `A1=sum(abs(phi))`.

The three downstream paths have fixed, noninterchangeable roles:

- `math64` promotes stored float32 means/log-scales/rotations, recomputes conics in float64, and
  uses exact analytic float64 colors. It owns the exact reproduction theorem and its own
  partition, first-moment, and A1 gates.
- `same64` promotes the exact stored float32 means, persisted float32-derived conics, and stored
  float32 colors verbatim. It owns same-state parity and a registered float64 permutation gate;
  its A1 is diagnostic.
- `cand32` uses the stored float32 state and the packed 15-channel `index_add`/centered-Cholesky
  path. It owns float32 output, moment, A1, overshoot, and permutation gates.

For shared-constant geometry cells, every path uses the current evaluation target's exact or
stored color array as specified above; the constant field's initialized colors are never reused
for a nonconstant evaluation target.

The float32 candidate mirrors renderer enumeration and `index_add`s exactly 15 values per pixel:
`s0`, `sd(2)`, symmetric `sdd(3)`, `sc(3)`, and `sdc(6)`. Form centered `C/G`, symmetrize `C`,
and solve the 2-by-2 system with `cholesky_ex` plus `cholesky_solve`. Nonzero solve info,
nonpositive diagonal, or nonfinite state kills the branch. Compare it with a float64 QR oracle
using the same rounded float32 geometry and colors.

Call the repository's `_tile_bounds` and `_tile_coords` directly, including its `torch.round`
semantics; a separately rewritten membership test is not the scientific implementation. Compute
float32 diagnostic effective weights from the same 3-by-3 moment system implied by the Cholesky
solve, not from output perturbations. Define p99 as `numpy.quantile(values, 0.99,
method="linear")`; cell RMSE is over all `64 x 64 x 3` output entries.

## Frozen forward gates

Every applicable gate is conjunctive per cell:

- every pixel passes mass, contributor, rank, condition, and Cholesky checks;
- float64 constant maximum error is at most `1e-12`;
- float64 affine maximum error is at most `1e-10`;
- float32 affine maximum error is at most `2e-5`;
- float32 versus same-state float64 has max absolute error at most `2e-5` and RMSE at most `2e-6`;
- `math64` partition and first-moment residuals are at most `1e-10`, and `cand32` versions at most
  `2e-5`;
- for every logical field separately, both `math64` and `cand32` effective-weight `A1` have p99
  at most `1.5` and max at most `4.0`; `same64` A1 is diagnostic; and
- every step/checker cell's unclipped float32 output excursion beyond the analytic per-channel
  sample extrema is at most `2/255`.

The float32 affine error is `cand32` minus the exact analytic float64 pixel target, with the
subtraction evaluated in float64. Do not pool A1 quantiles across logical fields. For overshoot,
the reference extrema are the global per-cell, per-channel extrema of the exact analytic float64
sample colors at all Gaussian means, not active-per-pixel extrema and not rounded stored colors;
promote `cand32` output to float64 for the comparison and do not clip it first.

## Frozen permutation and gradient gates

For every cell, apply identity, reverse, and two PCG64 permutations to every row-aligned tensor.
Initialize one global PCG64 stream with seed `20260716013`, traverse cells in lexical cell-ID
order, and draw the two random permutations sequentially per cell without resetting; identity and
reverse consume no RNG. A permutation is a length-`N` Gaussian-row permutation applied
consistently to means, log-scales, rotations, persisted conics, radii, and that cell's exact and
stored target-color arrays before calling `_tile_bounds`/`_tile_coords`; it is not a shuffle of an
already-enumerated contribution table. Persist every pre-drawn permutation. Both `math64` and
`same64` outputs must differ from their corresponding identities by at most `1e-12`, and
`cand32` by at most `2e-5`; rank, failure, and gate decisions must agree exactly for every path.

Run the gradient gate only on target-conditioned `affine_xy`, `N=56`, seed 0 after its forward
gates pass. Freeze its contributor `(gid,px,py)` list and radii, matching the current renderer's
detached support derivative. The scalar loss is the rendered-image inner product with an
L2-normalized Gaussian cotangent from one PCG64 stream seeded `20260716014`: draw the float64 HWC
cotangent first and normalize it, then draw and normalize two float64 directions for each block in
the order means, log-scales, rotations, colors. Cast that same cotangent once to float32 for
`cand32`; do not draw a second one. Parameter blocks are `means/L`, `log_scales`, `rotations/pi`,
and colors. Use centered finite differences with `h=2^-12` and the exact float64 directions for
both directional dot products.

The gradient `math64` path starts from all stored float32 parameters and colors promoted to
float64 but recomputes conics differentiably from means/log-scales/rotations; it renders through
the distinct differentiable float64 QR path. The `cand32` path uses the differentiable packed
15-channel renderer. Freeze only the contributor list and radii: never reround parameters,
recompute support, change contributor order, or retarget colors during AD or finite differences.
All unperturbed parameter blocks remain fixed.

For directional derivatives, require
`abs(AD64-FD64) <= 1e-8 + 1e-4*max(abs(AD64),abs(FD64))` and
`abs(AD32-AD64) <= 2e-5 + 2e-3*abs(AD64)`. All full block gradients must be finite with relative
L2 error reported using denominator `max(norm(g64),1e-12)`. The registered full-block gate is
`norm(g32-g64) <= 2e-5 + 2e-3*norm(g64)`; this mixed norm is necessary for the theorem-implied
null and storage-rounded near-null blocks and uses no dimension- or outcome-dependent scaling. Do
not select a new cell, direction, seed, or step size after failure.

Transform the full parameter gradients into the declared dimensionless coordinates before the
relative-L2 comparison: `g_(means/L)=L*g_means`, `g_logscale=g_logscale_raw`,
`g_(rotation/pi)=pi*g_rotation`, and `g_color=g_color_raw`. Do not compare gradients in raw
pixel/radian units.

## Plumbing controls

Use `H=W=9`, float64 geometry, no opacity, and analytic `affine_xy` colors for all controls. The
positive field has means `[(0,0),(8,0),(0,8),(8,8)]`, conics `(1/64,0,1/64)` and radii `(8,8)` for
every row; it must be accepted and reproduce affine RGB. The collinear negative has means
`[(1,4),(4,4),(7,4)]` with the same conics/radii, and the one-node negative has mean `(4,4)` with
the same conic/radius. Both negatives must be rejected by the rank/contributor gates. Accepting
either negative control or rejecting the positive control makes the assay invalid/unavailable
rather than a scientific failure.

## Artifacts and audit

Persist immutable protocol/config/environment/source manifests; analytic formula and target-array
hashes; every float32 field and frozen radii; an incremental per-pixel diagnostic table; gradient
and permutation JSONL; a complete gate and error ledger; and an executed-source archive. Write a
completion marker only when the exact cell/row grid exists. An independent replay must reconstruct
formula/geometry hashes and every gate from the raw rows.

Any missing cell/artifact, binding mismatch, unexpected control result, pixel exclusion, or
scientific gate failure kills Stage 0. Timing is diagnostic only because the reference is unfused.

Stage 0 is sequential. First run the plumbing controls and the complete 108-cell, 442,368-pixel
mass/contributor/float64-rank/condition screen. Only if every one of those early gates passes is the
float64 forward oracle, float32 15-channel path, permutation matrix, and gradient gate authorized.
A complete early-screen scientific failure is a decision-ready `kill`; downstream rows are then
recorded as `not_reached_preregistered_short_circuit`, not as missing artifacts. This short circuit
does not permit stopping after a favorable or unfavorable single cell: every early-screen cell and
pixel must be persisted and independently auditable.

Because the early screen passed, the authorized downstream phase has no outcome-dependent short
circuit: complete all three forward paths for all 108 cells, both independently gated A1 ledgers,
and all four permutations even after any downstream failure. The gradient gate runs if and only if
its one preregistered base cell passes its own forward gates; otherwise record exactly
`not_reached_preregistered_base_forward_failure` and do not substitute another cell. No other
downstream row may be omitted because an earlier row failed.

## Locked Stage 1

Only a clean Stage-0 pass authorizes freezing and implementing the already separated natural-image
assay: first 12 official TESTIMAGES SAMPLING images at center-aligned `160 x 160`, counts
`{175,350,700}`, current `NW_N`, same-state `LL_N`, byte-matched affine-color `AFF_4N/7`, and
work-spending `NW_4N`. Remaining TESTIMAGES IDs stay sealed. Complete bytes, active
Gaussian--pixel pairs, 15 accumulator channels, per-pixel solves, cold decode, solve time, and peak
memory must be charged. Stage 1 is not yet authorized and its decision thresholds may not be
adjusted using Stage-0 outcomes.

## Stop rule

Any Stage-0 failure closes the no-ghost, zero-extra-state, first-moment-corrected compositor under
the current compact-support grammar. No ridge, pseudoinverse, support enlargement, added centers,
opacity, alternate counts/seeds/targets, clipping, or threshold relaxation may rescue it. Reopening
requires a materially new boundary/support grammar and a new disjoint protocol.
