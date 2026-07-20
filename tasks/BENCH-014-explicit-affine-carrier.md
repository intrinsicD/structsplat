# BENCH-014: Explicit gauge-fixed affine carrier

## Status

Protocol frozen before any canonical BENCH-014 outcome inspection (2026-07-16). Benchmark-only.
This task may write benchmark code and immutable result artifacts, but it does not authorize a
production renderer, fitter, CUDA kernel, SSPL1/SSPL2 syntax, default, natural-image experiment,
or method claim.

An adversarial source preflight amended this text before target-array generation or outcome
inspection. It made the formula-binding byte strings and the already-required target-plane OLS
diagnostic explicit; no target, gate, threshold, seed, count, codec, or aggregation changed. This
revision supersedes the unexecuted draft SHA-256
`612a2d45a5ead36628ecceb39bb932795d0738ccf95d5d4dce31bbf855db8841`.

During a later read-only preflight of unexecuted revision SHA-256
`540d146df3e403d0956559b1f333d877f62c397ed7094da8609bb4564dabd97a`, the adversarial reviewer
accidentally evaluated the eight public formulas on the canonical pixel grid and printed only each
array's global minimum and maximum. No geometry, field, fit, tail, stream, render, trajectory,
timing, permutation, gradient, or method-dependent quantity was constructed or inspected. The
formulas are explicitly not a target holdout, and their global ranges are analytically available
from the frozen definitions, so the matrix and gates remain unchanged. This disclosure narrows the
pre-binding seal below: the reported formula-range exposure is discovery metadata, while every
method-bearing object and the persisted full target arrays remain protected by the binding.

A final conceptual audit amended only the claim interpretation before any method-bearing outcome:
it made the analytic-sampling encoder asymmetry, iteration-indexed variable-projection meaning,
and correlated designed-cell status explicit. It supersedes unexecuted revision SHA-256
`15817a199e5d588b8b5d4270b9756c09da200cf099bc2215cbc0066c8e95e245`; no matrix, gate,
threshold, target, seed, codec, implementation result, or aggregation changed.

The first proposed carrier was `q=[1,x_n,y_n]`, with nine binary16 coefficients. Adversarial
review removed the constant term before execution: normalized splats already reproduce constants,
up to the renderer's fixed denominator-epsilon tolerance, so an explicit intercept is a gauge
duplicate that adds six bytes and an avoidable near-null direction. The frozen carrier is
therefore the gauge-fixed linear tail `q=[x_n,y_n]`, with six
binary16 coefficients. A second disposable probe showed that an unconstrained encoder-side
least-squares tail can overshoot discontinuities. The protocol therefore freezes the same
deterministic range-safe ray projection for every target. This is an encoder constraint, not
decoder clipping and not a theorem that arbitrary affine tails are range preserving.

Except for the disclosed formula-grid global ranges above, no canonical target array may be
persisted and no canonical geometry, field, fit, tail, stream, render, optimization trajectory,
timing row, permutation, gradient row, or other method-bearing quantity may be evaluated before
the task SHA-256 and source manifest are written into the new result directory.

### Discovery exposure and canonical replacement

A disposable pre-freeze probe executed `H=71`, `W=83`, `N=81`, seeds `{211,223,227}`, and target
formulas 1--6 below while checking whether unconstrained defect least squares could overshoot;
`N=79` was proposed in that discovery design but was not executed as canonical evidence. Every
probe geometry cell is discovery-only and ineligible for BENCH-014 evidence. The canonical matrix
instead uses untouched `H=73`, `W=89`, counts `81/83`, and seeds `{307,311,313}`. The smooth
formulas are deliberately retained to test whether the frozen
thresholds generalize to disjoint geometry; they are **not** a target-formula holdout. The runner
must reject the exposed `H=71/W=83/seeds={211,223,227}` tuple and count `79` in a canonical output;
count `81` is legal only on the new canonical geometry. Discovery observations justify the frozen
range-safe encoder but must never be pooled with, compared as if held out from, or presented as
scientific evidence for the canonical result.

## Prior negative evidence and question

BENCH-013 tested a zero-extra-state local-linear/moving-least-squares compositor. Its complete v3
artifact is a decision-ready Stage-0 kill:

- task SHA-256: `b40b9075262c8c2dc07212490a1178b0168b6da9e433cb4ca23a10957bb1d0ad`;
- binding SHA-256: `2ff71a04472d061532f2b18dbf8619bf92cbc523aff9b50c96a3a6240be74adf`;
- `analysis.json`: `68db5bc6686f2b3dda430abaa0df6ffdd95698551a99cd0ae5a305cead258791`;
- `completion.json`: `cf2adf32b848e1a661c864ae6e6dab0e2a363416a26411c6374cafd9031d4e4e`;
- `replay.json`: `709d9b55f6535bbb55d61e926706b501dfe7faafd35b320d14778b38dee74ce8`;
- `artifact_manifest.json`:
  `1ef0823777ff70723603ca40d4d9ba1dabdfb266deee145e19f4ba7ba14ffc1f`;
- `executed_sources.tar`:
  `38569701e8a1eca60be1e47cad9a90e54e4e756765f5843953d18b5f7f1f21f3`.

The artifact completed all `108` forward cells and `432` permutation rows. It reported `82`
forward-cell failures, `49/63` effective-weight field failures, `27` permutation failures, and a
preregistered gradient cell not reached because its base forward gate failed. The failure was not
failure of affine reproduction: it was the signed-weight stability/ringing burden of enforcing
first moments independently at every pixel under compact support. BENCH-013's no-ghost local
linear formulation is closed; its thresholds, supports, counts, or ridge policy may not be
retuned here.

BENCH-014 asks a deliberately different question: can six explicitly transmitted global linear
RGB coefficients carry the broad first-order trend while the existing positive normalized
Gaussian compositor carries an independent residual, improving smooth-field quality, boundary
behavior, color-only convergence, and fixed-stream expressiveness without a local solve or signed
weights?

## Literature and novelty boundary

- Shepard, [*A Two-Dimensional Interpolation Function for Irregularly-Spaced
  Data*](https://doi.org/10.1145/800186.810616), establishes normalized positive scattered-data
  interpolation.
- Franke and Nielson, [*Smooth Interpolation of Large Sets of Scattered
  Data*](https://doi.org/10.1002/nme.1620151110), establish local interpolation/blending
  precedents.
- Hengl, Heuvelink, and Rossiter,
  [*About Regression-Kriging: From Equations to Case Studies*](https://doi.org/10.1016/j.cageo.2007.05.001),
  exemplify the much older deterministic-trend-plus-interpolated-residual decomposition.
- Golub and Pereyra,
  [*The Differentiation of Pseudo-Inverses and Nonlinear Least Squares Problems Whose Variables
  Separate*](https://doi.org/10.1137/0710036), establish the variable-projection foundation used by
  the convergence diagnostic.
- BENCH-013 already binds the local-linear regression, moving-least-squares, reproducing-kernel,
  and maximum-entropy-coordinate threats.

Neither a linear trend, residualization, positive normalized interpolation, least squares, nor
their combination is claimed as new. The novelty class is `known components, possibly new
recipient relationship`. The only potentially new evidence is whether this unusually small,
gauge-fixed, explicitly priced decomposition is useful for StructSplat under the frozen assay.
The strongest prior-art threat is generic regression-kriging/detrending; the strongest recipient
threat is simply spending the bytes on more standard Gaussians.

## Frozen representation

For `H=73`, `W=89`, pixel `(x,y)`, and a decoded Gaussian mean `mu_i=(mu_ix,mu_iy)`, define

```text
x_n(x)    = 2*x/88 - 1
y_n(y)    = 2*y/72 - 1
q(x,y)    = [x_n(x), y_n(y)]
q_mu(i)   = [2*mu_ix/88 - 1, 2*mu_iy/72 - 1]
w_i(x,y)  = the current positive compact-AABB Gaussian weight
p_i       = w_i / (sum_j w_j + 1e-8)
y_AC      = q(x,y) beta + sum_i p_i r_i
```

`beta` has shape `(2,3)` and row-major order
`[x_R,x_G,x_B,y_R,y_G,y_B]`. It has no intercept. The `r_i` are independent stored residual RGB
parameters, not a decoder-derived view of ordinary colors. The decoder receives only the stream;
it never receives the source target, solves a regression, reconstructs an intercept, clips output,
or modifies `beta` or `r`.

The renderer uses the current clipped AABB contributor rule, production integer radii,
`sigma_cutoff=3`, `support_fade=false`, `opacity=None`, and `aa_dilation=0`. It evaluates
`exp(-q_i/2)` at every pixel in the AABB without an ellipse mask or small-weight drop. It uses the
same four accumulated scalar channels as ordinary normalized RGB (`mass` plus three residual
numerators), followed only by the six multiplications and six additions needed to evaluate/add the
two-term RGB tail. It has no per-pixel matrix, factorization, solve, pseudoinverse, ridge,
denominator epsilon beyond the current renderer's fixed semantics, fallback, or signed residual
weight.

## Frozen targets

Evaluate every formula in float64 at all integer pixels and at continuous float32 Gaussian means.
Persist/hash the exact float64 pixel array and the single C-contiguous float32 cast. Do not clip or
antialias. Let

```text
u = x/88                  v = y/72
q_x = 2*u - 1             q_y = 2*v - 1
```

The exact eight targets are:

1. `constant`: `(0.25,0.50,0.75)`.
2. `affine`:
   `(0.45+0.18*q_x+0.12*q_y, 0.55-0.16*q_x-0.10*q_y,
   0.45+0.10*q_x+0.20*q_y)`.
3. `affine_sin`:
   `(0.45+0.16*q_x+0.10*q_y+0.07*sin(2*pi*u)*sin(pi*v),
   0.55-0.14*q_x-0.08*q_y+0.06*sin(2*pi*v),
   0.45+0.08*q_x+0.17*q_y+0.06*sin(2*pi*u)*sin(2*pi*v))`.
4. `affine_bump`: with
   `g=exp(-((u-0.68)^2/(2*0.11^2)+(v-0.35)^2/(2*0.14^2)))`,
   `(0.38+0.13*q_x+0.09*q_y+0.18*g,
   0.56-0.12*q_x-0.08*q_y-0.12*g,
   0.43+0.08*q_x+0.14*q_y+0.15*g)`.
5. `saddle`:
   `(0.48+0.10*q_x+0.07*q_y+0.10*q_x*q_y,
   0.52-0.08*q_x+0.06*q_y+0.09*(q_x^2-q_y^2),
   0.45+0.06*q_x+0.12*q_y-0.08*q_x*q_y)`.
6. `zero_linear`:
   `(0.45+0.12*cos(2*pi*u)*cos(2*pi*v),
   0.55+0.10*cos(2*pi*u)+0.06*cos(2*pi*v),
   0.45+0.10*sin(2*pi*u)*sin(2*pi*v))`.
7. `vertical_step`: `(0.1875,0.3125,0.75)` for `u<0.5`, else
   `(0.8125,0.6875,0.25)`.
8. `checker9x7`: parity is `(floor(9*u)+floor(7*v)) mod 2`, using the two
   `vertical_step` colors in parity order zero/one.

The three smooth-benefit targets are items 3--5. `zero_linear` is the registered no-linear-trend
control, and the discontinuity targets are items 7--8. `constant` and `affine` own
theorem/control gates and do not enter smooth-benefit aggregates.

To remove any whitespace or line-joining ambiguity, the exact UTF-8 strings that own the formula
SHA-256 bindings are listed below. The numbered prose above owns the mathematical definition;
these byte strings own only its persisted identifier.

```text
constant=(0.25,0.50,0.75)
affine=(0.45+0.18*q_x+0.12*q_y, 0.55-0.16*q_x-0.10*q_y, 0.45+0.10*q_x+0.20*q_y)
affine_sin=(0.45+0.16*q_x+0.10*q_y+0.07*sin(2*pi*u)*sin(pi*v), 0.55-0.14*q_x-0.08*q_y+0.06*sin(2*pi*v), 0.45+0.08*q_x+0.17*q_y+0.06*sin(2*pi*u)*sin(2*pi*v))
affine_bump=g=exp(-((u-0.68)^2/(2*0.11^2)+(v-0.35)^2/(2*0.14^2))); (0.38+0.13*q_x+0.09*q_y+0.18*g, 0.56-0.12*q_x-0.08*q_y-0.12*g, 0.43+0.08*q_x+0.14*q_y+0.15*g)
saddle=(0.48+0.10*q_x+0.07*q_y+0.10*q_x*q_y, 0.52-0.08*q_x+0.06*q_y+0.09*(q_x^2-q_y^2), 0.45+0.06*q_x+0.12*q_y-0.08*q_x*q_y)
zero_linear=(0.45+0.12*cos(2*pi*u)*cos(2*pi*v), 0.55+0.10*cos(2*pi*u)+0.06*cos(2*pi*v), 0.45+0.10*sin(2*pi*u)*sin(2*pi*v))
vertical_step=(0.1875,0.3125,0.75) for u<0.5, else (0.8125,0.6875,0.25)
checker9x7=parity is (floor(9*u)+floor(7*v)) mod 2, using the two vertical_step colors in parity order zero/one
```

Persist these strings and bind their SHA-256 values before target-array generation.

## Frozen geometry and cohorts

Counts are `{81,83}` and seeds are `{307,311,313}`. Use production `build_field` and this explicit
`quadtree_wse` configuration:

```text
candidate_oversample=6, density_base=0.05, density_power=1,
density_mode=structure, sampling_mode=wse, wse_progressive_order=false,
max_axis_ratio=6, coherence_power=1, orientation_mode=tensor,
scale_mode=spacing, init_scale_mult=1, scale_cap_mode=none,
background_fraction=0, background_grid=0, flank_offset_frac=0,
color_mode=bilinear, opacity_mode=none,
grad_sigma=1, tensor_sigma=2, gradient=central, structure_channel=luma,
flat_frac=0.02, corner_frac=0.15
```

Both cohorts are required:

1. `target_conditioned`: build independent `N=81` and `N=83` fields for every
   `target x seed`.
2. `shared_constant`: build `constant` geometry once for every `count x seed` and reuse that
   geometry byte-for-byte for all eight evaluation targets.

Overwrite initialized colors with the analytic target evaluated at the original continuous means,
cast once to float32. Encode that ordinary field, cold-decode it, and use the cold field as the
state from which both the NW baseline and the affine carrier are constructed. Do not reevaluate
the target at quantized means. The shared cohort changes target-specific colors only; means,
log-scales, rotations, and radii are identical across its targets.

The matrix has `48` comparison units (`8 targets x 2 cohorts x 3 seeds`), `54` logical geometry
fields (`48` target-conditioned plus `6` shared-constant fields), and three required render arms per
unit:

- `NW81`: same-count native mechanism, quality, convergence, and performance control;
- `AC81`: six-scalar candidate using the exact same decoded `N=81` geometry; and
- `NW83`: two-row-larger complete-byte/RD challenger.

Thus the static ledger has `144` render rows and `935,568` pixel rows
(`144 x 73 x 89`). `AC81/NW81` isolates the mechanism at equal count; only a complete cold-stream
comparison may establish that `AC81` fits within the `NW83` rate envelope.

## Frozen benchmark stream

The primary rate lane reuses the audited benchmark-only `GFCOV01` codec in its ordinary
`current_rs` arm with this single configuration:

```text
chart=current_rs, bits_means=12, geometry_bits=(6,6,6), bits_colors=8,
predictor=absolute, coder=zlib9
```

No predictor, chart, bit allocation, range, or coder search is legal. This codec and its ordinary
cold-decoded re-encode audit are bound by COMP-007's `analysis.json`
`115c2e272a406b1d85313496a94c76e6a4f47c59e41b79e13f951f0ff464ea27` and
`artifact_audit.json`
`ad1ec6c889e818e4b1af4cc63a4f99959453534951f480554471bf14fc621aa5`.
The exact codec source used by BENCH-014 must also be included in its own source archive; historical
audit does not waive current-source replay.

Wrap every inner GFCOV blob in the same benchmark-only `AFCR014` framing:

```text
fixed header          20 bytes: little-endian struct <8sBBHII
  magic                8s = ASCII "AFCR014" followed by one zero byte
  version              u8 = 1
  variant              u8 = 0 (NW) or 1 (AC)
  reserved             u16 = 0
  inner_nbytes         u32
  tail_nbytes          u32 = 0 (NW) or 12 (AC)
inner                 exactly inner_nbytes bytes, one GFCOV01 blob
tail                  AC only: six row-major little-endian IEEE-754 binary16 values
crc32                  4 bytes: little-endian CRC-32 of every preceding byte
```

No dummy tail is added to NW. Reject every other version, variant/tail combination, reserved value,
length, nonfinite decoded tail, CRC, trailing byte, malformed inner stream, or AC inner field whose
decoded count is not `81`; `AC83` is not a legal arm. Binary16 conversion is round-to-nearest,
ties-to-even; decode once to a C-contiguous float32 `(2,3)` array. The wrapper overhead common to
all arms is exactly `24` bytes. The complete rate is `len(AFCR014_blob)`, including both containers,
every range/tag/frame, the beta tail, and both CRCs. The uncompressed carrier state is exactly six
additional scalars/`12` bytes; compressed inner size may move in either direction because AC stores
residual rather than ordinary colors.

For every stream require deterministic encode bytes, deterministic cold decode, exact component
accounting, and `decoded state -> ordinary inner encode` plus decoded-beta re-encode equality with
the original complete bytes. Cold decoded geometry/support hashes must agree exactly within
`AC81/NW81`. The decoder API accepts the blob and device only. Supplying a source target,
recomputing `beta`, or performing target-plane OLS in the decoder invalidates the assay.

## Frozen encoder-side tail construction

For each comparison unit, begin from the cold-decoded `NW81` state. In float64, promote that exact
decoded state, enumerate its frozen production supports, and form its positive normalized weight
matrix `W`, ordinary rendering `y0=W c0`, pixel coordinate matrix `Q`, decoded-mean coordinate
matrix `Q_mu`, and defect design

```text
D = Q - W Q_mu                   # shape (73*89, 2)
E = target - y0                  # shape (73*89, 3)
```

Here and everywhere below, `W` uses the renderer's exact normalization
`w/(sum(w)+1e-8)`; it is positive but only approximately row-stochastic. Require `sum(W)` per
pixel within `2e-5` of one, finite mass at least `1e-5`, at least one active
contributor, and design rank two, where
`sigma_2 > max(73*89,2)*eps64*sigma_1`. Require `kappa_2(D)<=64`. Compute the unconstrained
`beta_ls` by reduced Householder QR of `D`, followed by one triangular solve with all three RGB
right-hand sides. SVD is rank/condition diagnostics only. Require the exact-float64 projection
invariant

```text
SSE(y0 + D beta_ls) <= SSE(y0) + max(1e-15, 1e-12*SSE(y0)).
```

Do not substitute a global target-plane OLS. The required diagnostic uses reduced Householder QR
to solve `beta_TP=argmin_beta ||target-Q beta||_F^2` with the same gauge-fixed two-column pixel
matrix `Q`. (`Q` has exactly zero discrete column means, so a fitted intercept would not change
these slopes.) Report `beta_TP`, its direct plane SSE, and `SSE(y0+D beta_TP)`; exclude all three
values from every gate and stream.

The range-safe ray is channel independent and identical for every target. For channel `j`, let
`z=D beta_ls[:,j]`, `lo=min(target[:,j])`, `hi=max(target[:,j])`, where the extrema are over the
exact float64 **pixel target**, not Gaussian samples. Analytically intersect the inequalities

```text
lo - 1.75/255 <= y0[p,j] + alpha*z[p] <= hi + 1.75/255
0 <= alpha <= 1
```

for all pixels and call the largest feasible value `alpha_cap`; clamp roundoff to `[0,1]`. An empty
intersection is invalid because `alpha=0` must be feasible. Evaluate, in descending order,

```text
alpha_k = alpha_cap * (1 - k/4096),  k=0,...,4095,
then the explicit terminal candidate alpha=0.
```

For each candidate, round `alpha*beta_ls[:,j]` once to little-endian binary16 and decode it to
float32. Construct that residual channel from the decoded coefficient,
`r[:,j]=c0[:,j]-Q_mu beta_decoded[:,j]`, in float64 and cast once to float32. Encode the complete
residual field through the frozen GFCOV configuration, wrap it with the complete six-channel beta
candidate (already accepted channels retain their selected coefficients; not-yet-selected channels
use zero), cold-decode, and render without clipping. Accept the first candidate for which

```text
candidate_min >= lo - 2/255
candidate_max <= hi + 2/255
candidate_channel_SSE <= NW81_channel_SSE
                         + max(1e-15, 1e-10*NW81_channel_SSE).
```

Channel selection order is `R,G,B`; after each channel is accepted, rebuild the complete residual
field from the current three decoded beta columns before selecting the next. Because channels do
not mix in the renderer or scalar color quantizer, previous-channel render bytes and decisions
must remain unchanged. Persist `beta_ls`, singular values, `alpha_cap`, selected `k` (use `4096`
for the explicit zero), selected alpha, pre/post-binary16 beta, every tried candidate's terminal
decision, and the final stream hash. If alpha zero fails, the plumbing is invalid: it must reproduce
the cold `NW81` appearance state. Never refine the grid, bisect between candidates, switch logic by
target class, clip output, or retune the two range margins after a result.

This construction guarantees only a tested encoder-side ray safeguard under the frozen quantizer.
It does not make the linear-tail decoder intrinsically convex-hull preserving.

## Numerical and stability gates

Use independent float64 and production-like float32 renderer paths. The float64 path promotes the
cold-decoded float32 field and beta and recomputes conics from decoded log-scales/rotations. The
float32 path uses the exact cold-decoded state and current contribution enumeration/accumulation.
All `73 x 89` pixels are scored; any nonfinite state, missing pixel, mass below `1e-5`, or empty
support fails.

For the residual weights in every static arm and permutation require:

- maximum partition residual `abs(sum_i p_i-1) <= 2e-5` in both float32 and float64;
- minimum float32 effective weight at least `-2e-7`;
- per-pixel float32 `A1=sum_i abs(p_i) <= 1+2e-5`;
- finite nonnegative source weights and identical contributor/pair hashes within `AC81/NW81`.

The small negative tolerance diagnoses arithmetic only; an implementation that intentionally
creates signed weights fails the mechanism.

For each decoded `N=81` field, independently form dense `W` and `[W,D]`. Define one common absolute
threshold `tau=max(73*89,83)*eps64*sigma_1([W,D])`, then count singular values of both matrices
strictly above that same `tau`. Require
`rank([W,D]) == rank(W)+2`. The same two added spatial columns apply independently to RGB, so this
is exactly six additional linear appearance degrees of freedom. A rank increase is an
expressiveness diagnostic under the frozen geometry, not evidence that its directions are useful;
the quality gates test usefulness.

## Frozen quality gates

MSE is the float64 mean of all unclipped HWC squared errors against the exact float64 pixel target.
Channel SSE is an unnormalized pixel sum. PSNR is `10*log10(1/MSE)` with peak one and is diagnostic.
For ratios use denominator `max(baseline_mse,1e-20)`. The outer-nine-pixel mask is
`x<9 or x>=80 or y<9 or y>=64`. A win is a strict `AC81_mse < NW81_mse`; no tolerance is used to
turn a tie into a win. Median is `numpy.quantile(values,0.5,method="linear")`.

Every gate is conjunctive:

1. On `constant`, every arm has maximum absolute error at most `1e-6`.
2. On `affine`, every `AC81` cold render has maximum absolute error at most `1e-3`.
3. For each of `affine_sin`, `affine_bump`, and `saddle` separately, across its six
   `cohort x seed` comparisons, `AC81` beats same-count `NW81` in at least `5/6`, has median
   full-image MSE ratio at most `0.85`, and has median outer-nine-pixel MSE ratio at most `0.75`.
4. On `constant` and `zero_linear`, every same-count `AC81/NW81` MSE ratio is at most `1.001`.
5. For every discontinuity comparison and channel, the unclipped output of `AC81` remains
   within the exact pixel-target `[min-2/255,max+2/255]`.
6. Each final cold `AC81` channel SSE satisfies its encoder-side nonregression constraint against
   `NW81` in every target/cohort/seed cell.

Report unconstrained exact-float64 projection, range-safe pre-stream, and target-plane OLS
diagnostics, but none may replace the three registered arms in these gates.

## Frozen actual-byte gates

For each target separately across its six `cohort x seed` comparison units:

- `AC81` complete bytes are at most `NW83` complete bytes in at least `5/6` units;
- the median `len(AC81_complete_AFCR014)/len(NW83_complete_AFCR014)` is at most `1.00`;
- the worst such complete-byte ratio is at most `1.02`;
- for each of `affine_sin`, `affine_bump`, and `saddle`, the median cold-decoded
  `AC81/NW83` MSE ratio is at most `0.95`; and
- for `zero_linear`, `vertical_step`, and `checker9x7`, every cold-decoded `AC81/NW83` MSE ratio is
  at most `1.05`.

In addition, for every one of the `48` comparison units:

- all three `NW81`, `AC81`, and `NW83` complete streams pass cold decode, exact accounting,
  deterministic encode, and decoded
  re-encode;
- the carrier tail is exactly `12` bytes and no coefficient, alpha, range constraint, target
  statistic, or regression state appears elsewhere; and
- decoded `AC81` count is `81`, decoded `NW83` count is `83`, and their variant tags are correct.

Only after this gate passes may the result call AC81 `within the frozen NW83 complete-byte
envelope`. It must not be called generally byte-matched or an SSPL1 compression win. Report the
same-count `AC81-NW81` total-byte delta and exact component differences as diagnostics.

## Frozen color-only convergence lane

Run this lane for all `48` comparison units on the cold-decoded `N=81` geometry. It has two
trajectories per unit:

- `NW81_opt`: train centered node appearance `a` from `a0=c0`;
- `AC81_varpro`: train the same centered node appearance `a` from `a0=c0` while eliminating beta by
  a fixed-design
  Golub--Pereyra variable-projection solve.

At logical step zero, their fields, contributor rows, rendered output, and loss must be bitwise
identical: both use `a=c0`, and AC uses the explicit `beta_0=0`. The candidate's corresponding
decoder residual is `r_0=a0`. Geometry, support memberships, radii, means, log-scales, rotations,
`W`, `Q_mu`, and `D` remain fixed.

For each candidate update, promote the current float32 centered appearance to float64 and solve

```text
beta*(a) = argmin_beta ||target - W*a - D*beta||_F^2
```

with the one preregistered reduced Householder QR of `D` and a triangular solve for all RGB
right-hand sides. Treat the exact solve as detached in the subsequent node-color gradient; by the
variable-projection envelope condition this is the reduced-objective gradient. After the Adam node
update, solve beta again for the logged state and materialize the decoder residual only as
`r=a-Q_mu*beta`, verifying `q*beta+W*r == W*a+D*beta` to `1e-10`. The baseline uses `W*a` and the
same centered-appearance dtype. Both losses and beta solves are evaluated in float64; Adam owns
only C-contiguous float32 `a`. This centered parameterization matches the defect mechanism and
removes avoidable slope correlation from the trainable node variables. The binary16 tail, range
ray, GFCOV quantizer, and codec are
absent from this convergence lane. A joint-Adam residual/beta trajectory may be reported only as a
complete diagnostic and cannot enter the gate.

Run exactly `100` Adam updates with `lr=0.03`, `betas=(0.9,0.999)`, `eps=1e-8`, no weight decay, no
schedule, no clamp, no early stopping, and fresh optimizer state per trajectory. Log every logical
step `0,...,100`. Both arms optimize the same all-pixel HWC RGB MSE; there is no target-class branch
or coefficient solve in the NW arm.

For each trajectory define normalized loss AUC by the trapezoid rule over the `101` logged losses:

```text
AUC = (0.5*loss_0 + sum_{s=1}^{99} loss_s + 0.5*loss_100) / 100.
```

Across the `18` `affine_sin/affine_bump/saddle x cohort x seed` units, require median
`AUC_AC81_varpro/AUC_NW81 <= 0.90`, using denominator `max(value,1e-20)` and the frozen linear
median. No one of those `18` units may have
`loss_AC81_varpro_step100/loss_NW81_step100 > 1.01`. All constant, affine, zero-linear, and
discontinuity trajectories remain required diagnostics; none may be omitted after a failure.

This lane tests only color-subspace convergence from a shared state. It cannot support a full
fitter-convergence claim.

## Frozen permutation gate

For every static renderer row (`48 units x 3 arms`), apply identity, reverse, and two PCG64 row
permutations consistently to all row-aligned decoded tensors before contribution enumeration.
Initialize one global NumPy PCG64 stream with seed `20260716015`, traverse renderer IDs lexically,
and draw the two random permutations sequentially; identity and reverse consume no RNG. The global
beta is unchanged and is never re-solved. Persist every permutation.

Relative to identity, require maximum output difference at most `1e-12` for float64 and `2e-5` for
float32. Contributor counts, support decisions, finite/failure decisions, weight-stability gates,
and range/excursion decisions must agree exactly. The exact ledger is `576` rows
(`144 renderer rows x 4 orders`). Run every order even after a failure.

## Frozen gradient gate

Run one gradient cell only: `target_conditioned/affine_sin/N81/seed307/AC81`, if and
only if its static forward, stability, stream, and permutation gates pass. Otherwise write one
terminal `not_reached_preregistered_base_failure` row; do not substitute another cell.

Start from its exact cold-decoded residual field and beta, but treat residual RGB and beta as
independent continuous decoder parameters. The encoder QR, alpha search, residual construction,
binary16 quantizer, GFCOV codec, and target are outside the differentiated graph. Freeze the
identity contributor `(gid,x,y)` list and radii; never reround parameters or recompute support under
perturbation. The float64 path promotes decoded state, recomputes conics differentiably, and uses
the frozen-list normalized renderer. The float32 path uses the corresponding packed four-channel
renderer plus the explicit tail.

Use an L2-normalized HWC float64 Gaussian cotangent from a NumPy PCG64 stream seeded
`20260716016`; cast it once to float32. Then draw and normalize two float64 directions per block in
this order:

1. means in coordinates `(mu_x/88,mu_y/72)`;
2. log-scales;
3. rotations divided by `pi`;
4. residual RGB;
5. beta.

Use centered finite differences with `h=2^-12`. Transform full raw gradients before comparison:
`g_(mu_x/88)=88*g_mu_x`, `g_(mu_y/72)=72*g_mu_y`,
`g_(rotation/pi)=pi*g_rotation`; other blocks are unchanged. Require all values finite and, for
each direction,

```text
abs(AD64-FD64) <= 1e-8 + 1e-4*max(abs(AD64),abs(FD64))
abs(AD32-AD64) <= 2e-5 + 2e-3*abs(AD64).
```

For each full dimensionless block require

```text
norm(g32-g64) <= 2e-5 + 2e-3*norm(g64).
```

Also report relative L2 with denominator `max(norm(g64),1e-12)`. These are BENCH-013's repaired
mixed absolute/relative gates; do not replace them with a pure relative check on near-null blocks.

## Frozen performance gate

The operation ledger is primary and wall time is a separately required reference-implementation
check:

- `AC81` and `NW81` must have exactly the same contributor-triplet hash and number of
  Gaussian--pixel weight evaluations in every unit;
- both use exactly four accumulated scalar channels per pixel contribution;
- `AC81` adds exactly six pixel-tail multiplications and six additions per output pixel, with no
  per-pixel solve/factorization and no extra weight channel.

Time CPU float32 cold render and cold `decode+render` separately. Pin PyTorch intra-op and inter-op
threads to one, disable gradients, perform `20` unrecorded warmups and `100` recorded repetitions per
arm/unit, synchronize if a device ever requires it, and use `time.perf_counter_ns`. A global PCG64
stream seeded `20260716017` predraws an independent arm order for each repetition while cells are
traversed lexically; persist the schedules and all samples. Decode begins from immutable bytes;
render timing begins from already decoded state. Require across the `48` units:

- median of `median_render_ns(AC81)/median_render_ns(NW81) <= 1.15`.

Record cold `decode+render`, peak RSS, and encoder QR/ray-search work diagnostically. A timing
failure closes the current reference realization but does not negate a separately reported
mathematical quality result.

## Plumbing controls

Run and persist controls before the scientific matrix:

1. An independently constructed exactly row-stochastic, reflection-symmetric `W` and symmetric
   `Q/Q_mu` constant fixture verifies that the no-intercept optimum is beta zero within `1e-12`.
   A separate positive renderer-normalized `W` using the frozen `1e-8` denominator epsilon and a
   full-rank `D` verifies the general defect identity
   `q beta + W(c-Q_mu beta) == y0+D beta` to `1e-12` without requiring beta zero.
2. A `9 x 11` broad-support affine field with binary16-representable slopes passes rank, decoded
   beta, range-ray, stream, and `1e-3` reproduction checks.
3. A production-semantics constant field passes the `1e-6` output and `1.001` AC/NW-ratio gates;
   its small EPS-induced fitted beta is diagnostic and is not required to be exactly zero.
4. An explicitly supplied finite `17 x 2` defect design with
   `D[:,1]=2*D[:,0]` is diagnosed as rank one and rejected before QR/triangular solve. It is a
   linear-algebra plumbing control, not a renderer field; renderer controls retain legal
   `H,W>=2`.
5. Alpha zero rebuilds the exact decoded NW appearance and passes the channel SSE safeguard.
6. AFCR014 wrong version, nonzero reserved field, wrong tail length, CRC corruption, trailing
   bytes, nonfinite beta, and malformed inner stream are each rejected.
7. Decoding the same valid blob in two calls with different unavailable dummy source arrays is
   impossible by API construction; a source-target argument or decoder-side linear solve fails
   the control audit.

Rejecting a positive control or accepting a negative control makes the assay invalid/unavailable,
not a scientific failure.

## Artifact, replay, and execution contract

Before science, persist immutable protocol/config/environment/source manifests, this task hash,
the bound BENCH-013 and COMP-007 hashes, formula strings/hashes, exact target arrays, expected stable
keys/counts, and an executed-source archive. Persist:

- all initial, ordinary cold-decoded, residual, and final cold-decoded fields with state/support
  hashes;
- every complete AFCR014 stream and exact byte-component ledger;
- raw per-pixel weights/stability/output/error diagnostics in incrementally written NPZ files;
- the `144` static-render, `576` permutation, `96` optimization-trajectory, `9,696` optimization-
  checkpoint, conditional gradient, timing, beta-search, and gate ledgers;
- every RNG permutation, direction, cotangent, and timing schedule; and
- analysis, independent replay, artifact manifest, and a completion marker written last.

Every expected key has one terminal `ok`, `error`, or explicitly preregistered `not_reached` row.
Missing, duplicate, nonfinite, stale-binding, hash-mismatched, or silently excluded rows invalidate
the relevant assay. Controls run first. Once they pass, there is no outcome-dependent scientific
short circuit: complete all `48` units, all alpha searches, all streams, all static arms, all four
permutations, all optimization trajectories/checkpoints, and all timing rows even after a quality,
rate, convergence, stability, or performance failure. Only the one gradient cell has its explicit
base-failure condition.

Independent replay must reconstruct targets and shared geometries, cold-decode and ordinarily
re-encode every stream, rebuild beta candidates and alpha decisions from raw state, rerender all
static/permutation outputs, recompute convergence AUCs and every aggregate/gate, verify exact row
counts, audit the decoder call graph for a target/solve dependency, and validate the executed-source
archive. Analysis may inspect scientific outcomes only after this replay passes.

## Claim boundary and decision

This is a synthetic analytic Stage-0 assay. A full pass establishes only:

- **quality:** improvement on the three registered smooth-benefit analytic targets at the fixed counts,
  seeds, cohorts, codec, and range safeguard;
- **stability:** positive normalized residual weights and bounded registered outputs in this matrix;
- **convergence:** faster color-only optimization from one shared initialization policy;
- **performance:** the frozen CPU reference and counted decoder work meet the stated bounds;
- **compression:** the complete benchmark AFCR014/GFCOV streams fit within the registered `NW83`
  byte envelope; and
- **expressiveness:** six global linear RGB scalars add the registered rank and improve over the
  same-count `NW81` control, while the `AC81` stream competes with the registered `NW83` rate/RD
  challenger on these targets.

The static encoder effort is intentionally asymmetric: all ordinary arms use only analytic target
samples at their original means, whereas AC additionally receives the frozen target-wide QR and
codec-in-loop ray search. Consequently, a pass measures the marginal value of this six-scalar
extension over the frozen sampling initializer; it is not superiority to color-optimized or
fully-fitted NW baselines. Likewise, the convergence lane indexes 100 Adam updates while AC uses
exact variable projection between updates. Its AUC is reduced-objective loss versus update index,
not wall-clock speed or a claim that its optimizer is intrinsically better. The six cells per
target are correlated designed cohort/seed conditions and their predicates are descriptive, not
inferential samples.

It cannot establish natural-image quality, full fitting convergence, GPU/CUDA performance,
production SSPL rate, dataset generalization, broad robustness, publication novelty, or superiority
to learned context/VQ codecs. Even a complete pass only authorizes freezing a new disjoint
natural-image and production-codec Stage 1.

## Stop rule

Every gate above is conjunctive. Any failure closes this exact gauge-fixed two-coordinate,
range-ray, same-count `AC81/NW81` mechanism and `AC81/NW83` complete-rate formulation without
changing targets, seeds, counts, margins, alpha grid, codec bits/predictor/coder, optimizer,
thresholds, timing procedure, or
aggregation. Do not rescue it with an intercept, decoder OLS, local-linear weights, ridge,
pseudoinverse, clipping, target-class switch, more tail bits, more Gaussians, a different color
quantizer, or a post-outcome alpha refinement. Reopening requires a materially new mechanism and a
new disjoint protocol.
