# BENCH-015: Decoder-synchronized robust affine lift

## Status

Protocol design frozen before canonical BENCH-015 target, geometry, stream, render, trajectory, or
timing generation (2026-07-16). Benchmark-only. Preflight may amend this document only before a
canonical binding is written. Once bound, any amendment requires a fresh result directory and is
ineligible to rescue a failed gate.

Before binding, tests may execute only target-free plumbing controls, explicitly noncanonical small
fixtures, and the target-independent direction/synthetic-fixture/region-mask binding arrays named
below. They may not evaluate a default `83 x 107` target, build a canonical-seed field, or score any
registered comparison.

This task authorizes benchmark code, tests, immutable result artifacts, and one canonical disjoint
Stage-0 run. It does not authorize production renderer/codec changes, a new SSPL syntax or default,
natural-image execution, or a publication claim.

## Prior evidence and branch selection

BENCH-014's canonical artifact
`results/bench014_affine_carrier_stage0_v1_2026-07-16` is a complete, independently replayed
scientific kill of the transmitted gauge-fixed affine tail:

- task SHA-256: `e2b7661c8a856cb2b560bd36e34c0d89f4962f06cf850d94f73afeb98623dbc7`;
- binding SHA-256: `0a4febdbd9570bcb216a8f6ec1f3461af3577553401415900be4583e5f34825c`;
- analysis SHA-256: `cb33e79204e7e1eca8be80fe8cb8591c968d59afeb0f7b5f3a1df9af86a0a6c3`;
- replay SHA-256: `f93c571bb554eccd94fde7b347751d99316526a9d6bb753c5da6b6f992f4e024`;
- artifact-manifest SHA-256:
  `3a49fb7cab3071d95e2943337c21c28d4c2601b61fcce8853a98f544015c511a`;
- source-archive SHA-256:
  `1c0d9cceb7f87ce2db74592d4f6f8c4dbeb019d3217b36f96ea2f5867beee8e3`.

All 18 stored replay checks and a fresh external 19-check replay passed. At equal count, the
carrier's median MSE ratios were `0.433251`, `0.668130`, and `0.404654` on its three smooth target
families, with `6/6` wins each and exact rank gain `+2` in `48/48` cells. Its CPU render ratio was
`1.005603`. It nevertheless failed every target's actual-byte gate: the fixed 12-byte tail plus
less-compressible residual colors produced same-count byte deltas from `-17` to `+196` bytes. Its
median convergence AUC ratio passed (`0.668853`), but three shared-geometry bump cells failed the
terminal guard, with worst final ratio `1.417939`.

The evidence therefore closes BENCH-014's transmitted-beta/range-ray representation but does not
close global first-order structure. BENCH-015 tests the one preregistered successor: derive a
first-order trend solely from the ordinary decoded field, transmit no coefficient or residual
colors, and use a target-blind smooth confidence gate to fall back to ordinary normalized
Gaussians on discontinuities.

## Literature and novelty boundary

The mathematics is not claimed as new. Regression/universal kriging already decomposes a
deterministic trend plus an interpolated residual (Hengl et al.,
[DOI 10.1016/j.cageo.2007.05.001](https://doi.org/10.1016/j.cageo.2007.05.001)), and moving least
squares supplies polynomial reproduction (Lancaster and Salkauskas,
[DOI 10.1090/S0025-5718-1981-0616367-1](https://doi.org/10.1090/S0025-5718-1981-0616367-1)).
Backward-adaptive image prediction already refits linear coefficients from decoded causal samples
without predictor side information (Motta, Storer, and Carpentieri,
[DOI 10.1109/5.892714](https://doi.org/10.1109/5.892714); Ulacha et al.,
[DOI 10.3390/e22090919](https://doi.org/10.3390/e22090919)). Cauchy IRLS is standard robust
estimation, and fixed iterations do not imply a global optimum. Data-dependent MLS uses WENO-style
smoothness weights near discontinuities ([arXiv:2412.02304](https://arxiv.org/abs/2412.02304)); a
2026 data-dependent Shepard method adapts positive scattered-data kernels from an LS
discontinuity indicator ([arXiv:2606.20332](https://arxiv.org/abs/2606.20332)). Log-mean-exp soft
morphology and quintic smootherstep are also established primitives.

The novelty class is `known components, possibly new recipient/evidence relationship`. The only
potential contribution is deriving the robust trend and confidence from the same cold-decoded
StructSplat rows, combining it with the existing compositor, and measuring same-inner-stream RD,
robustness, convergence, gradients, and cost under a source-bound assay. No new robust estimator,
soft maximum, polynomial primitive, or decoder-side predictor is claimed.

### Discovery exposure

After BENCH-014 was complete, its already-scored `H=73`, `W=89`, `N=81`, seeds
`{307,311,313}`, and eight target families were used only as a mechanism-discovery set. Plain
decoded-color ordinary least squares produced same-count median MSE ratios `0.5788`, `0.8843`, and
`0.5270` on the prior smooth families, but a `1.2876` median ratio and `0.16394` worst range
excursion on the prior vertical step. A fixed eight-step Cauchy IRLS plane improved the prior bump
median to `0.7855`. The first raw-color global gate was rejected during adversarial preflight: it
could suppress a legitimate steep affine trend and acted as a scene classifier. The corrected
gate scores neighbor jumps of the robust-plane residual and attenuates the complete affine
correction, making exact affine data active and confidence zero an exact NW fallback. On the prior
data, residual log-mean-exp score maxima were `1.0350` on smooth-benefit rows and minima were
`1.5535` on steps. One corrected `1.20--1.50` C2 band was examined: it retained the prior smooth
signal and exactly fell back on the prior discontinuities.

Those calculations are exploratory, multiply tried, and categorically ineligible as BENCH-015
evidence. In particular, OLS, Huber, pseudo-Huber, and Cauchy fits, four raw-color transition
bands, and the single corrected residual-score band were examined. The canonical replacement
below changes dimensions, counts, seeds, and every nontrivial formula. No canonical target,
field/geometry, stream, render, trajectory, or timing array may be generated before binding. The
target-independent direction, synthetic-fixture, and region-mask prebinding arrays explicitly
required below are the sole exception: generate and hash them after task/source hashing and include
their records in the canonical binding before any target or field.

A pre-binding numerical control on a noncanonical `N=32` fixture found that the initially proposed
finite-difference step `2^-12` crossed an IRLS median-order knot: its AD/FD discrepancy was
`2.1e-4`, while `2^-16` reduced the discrepancy to `6.28e-10` and remained stable through `2^-22`.
The gradient step below is therefore frozen at `2^-16` before any BENCH-015 target generation.
This is not a claim that median/MAD IRLS is globally smooth: the gradient ledger records the two
central order-statistic row identities at the base and both finite-difference endpoints so any
piecewise-differentiable kink crossing remains visible and cannot waive a failed gate.

Before binding, assay unit/smoke tests then materialized a full `H=79,W=101` constant array, three
affine/occlusion point evaluations, and one full `H=79,W=101` affine-chirp target. They created no
Gaussian geometry, stream, cold state, render, error metric, or scientific score. This nevertheless
violated the literal target-array seal above. Rather than grant a retrospective exception, the
canonical matrix below was moved before binding to previously unevaluated `H=83,W=107`; every
dimension-dependent pixel grid, region mask, line distance, and synthetic gradient fixture changed
with it. The exposed `79 x 101` arrays are ineligible for evidence, and the replacement dimensions
may not be changed again after a result.

## Question and null

Can an ordinary same-byte Gaussian stream support a useful first-order reproduction basis when
the decoder deterministically factors a robust affine trend from its decoded node colors, while a
smooth, target-blind local-jump score disables the trend on discontinuities?

The primary null is the exact same decoded `N=78` field rendered by ordinary positive normalized
Gaussian interpolation (`NW78`). `DSL78` changes only the decoder interpretation of that state.
It has the same Gaussian rows, quantized attributes, inner stream bytes, and parameter count. A
complete `NW80` stream is a stronger two-standard-row actual-rate/RD control. Failure of the
disjoint same-byte quality, no-harm, convergence, or cost gates closes this global first-order
family; no threshold, neighbor count, robust loss, iteration count, or formula may be retuned.

## Frozen representation

For decoded means `mu_i=(mu_ix,mu_iy)`, decoded ordinary colors `c_i`, and pixel `(x,y)`, define

```text
q_mu(i) = [1, 2*mu_ix/(W-1)-1, 2*mu_iy/(H-1)-1]
q(x,y)  = [1, 2*x/(W-1)-1,     2*y/(H-1)-1]
X_mu    = rows q_mu(i)                         # N x 3
X       = rows q(x,y)                          # HW x 3
p_i     = w_i / (sum_j w_j + 1e-8)
W_G     = the HW x N matrix of p_i
```

The current positive clipped-AABB Gaussian support, integer radii, `sigma_cutoff=3`, no ellipse
mask, no weight drop, `support_fade=false`, `opacity=None`, and `aa_dilation=0` are unchanged.

### Robust decoded-color plane

All derivation is cold-decoder-side and float64. Require `rank(X_mu)=3` under
`tau=max(N,3)*eps64*sigma_1` and `kappa_2(X_mu)<=8`. For each RGB channel independently:

1. initialize `b` by reduced Householder QR least squares of `X_mu b ~= c`;
2. repeat exactly eight updates:

```text
e_i       = c_i - q_mu(i) b
m         = linear-quantile(0.5, {e_i})
s         = max(1/255, 1.4826 * linear-quantile(0.5, {abs(e_i-m)}))
omega_i   = 1 / (1 + (e_i/s)^2)
b         = argmin_z sum_i omega_i * (c_i-q_mu(i)z)^2
```

The weighted solve is reduced Householder QR of `sqrt(omega) X_mu` followed by one triangular
solve. Its rank test is the same strict singular-value rule
`sigma_3 > max(N,3)*eps64*sigma_1`. Record all `8 x 3` weighted condition numbers. Reject a
nonfinite or rank-deficient intermediate,
but impose no hidden weighted-condition threshold; do not ridge, pseudoinvert, change the scale
floor, converge early, or add iterations. This is fixed Cauchy IRLS, used as a robust trend factor
rather than claimed as new robust regression. The two quantiles use the average of the two central
order statistics for even `N`, matching NumPy/PyTorch linear interpolation.

### Smooth residual-discontinuity confidence

After the eighth IRLS update, let `e=c-X_mu b` for the unconstrained robust plane `b`. Build a
directed graph from decoded normalized mean coordinates. Each row selects its four nearest other
rows by squared Euclidean distance; decoded `(x_n,y_n)` lexicographic order is the exact,
permutation-invariant distance-tie break. Exact duplicate decoded coordinates are invalid. For
channel `j`, over its `4N` directed edges, define

```text
d_ej       = sqrt((e_i-e_k)^2 + (1/4096)^2)
sigma_cj   = sqrt(mean_i((c_i-mean_i(c_i))^2) + (1/255)^2)
z_ej       = d_ej / sigma_cj
score_j    = 0.1 * log(mean_e(exp(z_ej/0.1)))
t_j        = clip((score_j-1.20)/(1.50-1.20), 0, 1)
S2(t)      = 6*t^5 - 15*t^4 + 10*t^3
alpha_j    = 1 - S2(t_j)
```

Compute `score` with a subtract-maximum log-sum-exp identity. Underflow of negligible exponential
terms is legal; a nonfinite maximum, nonfinite/zero sum, or nonfinite result is invalid.
The log-mean-exp is a smooth maximum, while `S2` is the C2 smootherstep transition. There is no
target class, target range, pixel error, encoder search, learned threshold, transmitted gate, hard
clip of the rendered image, or post-outcome adjustment.

Multiply all three fitted coefficients in channel `j` by `alpha_j`. Call the resulting `3 x 3`
matrix `beta`. Then

```text
r        = c - X_mu beta
y_DSL    = X beta + W_G r
         = W_G c + (X-W_G X_mu) beta.
```

For a fixed unconstrained `b`, this is the per-channel convex blend
`y_DSL=(1-alpha)*y_NW+alpha*y_full`; therefore `alpha=0` is an exact NW fallback, including the
renderer denominator epsilon. Exact affine decoded colors have zero robust residual, hence
`alpha=1` by construction and are reproduced up to declared arithmetic tolerance. The confidence
remains a global per-channel switch. A localized-occlusion target below is explicitly allowed to
kill the mechanism if that global scope prevents simultaneous smooth-region benefit and edge
safety.

`NW78` is `W_G c`. The DSL decoder receives only complete stream bytes and device; it may not
receive a target, source pixels, alpha, beta, residual, class, threshold override, or regression
state. The robust plane and confidence are recomputed from the cold-decoded means/colors. The
reference path retains the float64 derived state; the measured renderer casts beta and residuals
once to contiguous float32.

At render time DSL uses the same mass plus three RGB numerator accumulators as NW and adds one
three-term RGB plane: six spatial multiplications and nine additions per pixel. There is no per-pixel
solve, factorization, extra weight channel, signed Gaussian weight, or output clip. QR, graph, IRLS,
beta, and residual formation occur once after decode and are measured separately.

## Frozen disjoint matrix

Use `H=83`, `W=107`, counts `{78,80}`, and seeds `{401,409,419}`. These dimensions, counts, and
seeds were not used in BENCH-014 discovery or evidence. Count 79 is deliberately excluded because
it appeared as an unexecuted BENCH-014 discovery proposal. Use both cohorts:

1. `target_conditioned`: independent fields for every `target x count x seed`;
2. `shared_constant`: build constant geometry once for every `count x seed`, then replace colors
   with each target evaluated at the original continuous means.

The frozen field construction is the same production `quadtree_wse` family as BENCH-014:

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

Overwrite initialized colors with the analytic target at original continuous means, cast once to
float32. Encode normally, cold-decode, and never reevaluate target colors at quantized means.

There are nine targets, `54` comparison units, `60` logical fields, and three arms per unit:
`NW78`, `DSL78`, and `NW80`. Static evidence contains `162` rows. `DSL78/NW78` must share cold
geometry and ordinary colors byte-for-byte.

The exact terminal inventory is: `60` field rows; `162` complete streams and static rows; `54`
DSL attribution records, each containing both OLS and ungated-Cauchy outputs; `486` permutation
rows; `108` convergence trajectory rows containing `61` losses/output hashes each (`6,588` logical
step records total) plus eight arrays at steps `{0,20,40,60}` per trajectory (`864` checkpoint
arrays); `6` gradient rows; and `162` timing rows. Timing contains `6,480` recorded render samples,
`6,480` recorded cold samples, and `6,480` separately recorded derive diagnostics. The two
decision-bearing timing kinds therefore contain `324` arm/kind cells and `12,960` samples. Exact
expected-ID set equality, not only these counts, is bound before target creation. These sample and
checkpoint-array counts apply when every registered method is available; a terminal
`method_unavailable` sentinel replaces, rather than silently omits, each dependent sample/checkpoint
payload and is itself a scientific failure.

## Frozen targets

Let `u=x/106`, `v=y/82`, `q_x=2*u-1`, and `q_y=2*v-1`. Evaluate exact pixel arrays and continuous
mean samples in float64, then cast once where float32 is required. Do not clip or antialias.

1. `constant`: `(0.30,0.52,0.68)`.
2. `affine`:
   `(0.46+0.15*q_x-0.11*q_y, 0.54-0.13*q_x+0.14*q_y,
   0.42+0.09*q_x+0.18*q_y)`.
3. `affine_chirp`: with `p=2*pi*(0.70*u+1.10*u^2+0.35*v)`,
   `(0.46+0.13*q_x-0.08*q_y+0.045*sin(p),
   0.54-0.11*q_x+0.12*q_y+0.040*cos(p+0.5*pi*v),
   0.43+0.07*q_x+0.15*q_y+0.040*sin(p-0.7*pi*v))`.
4. `affine_soft_crease`: with
   `t=(u-0.52)+0.45*(v-0.50)` and `h=sqrt(t^2+0.035^2)`,
   `(0.45+0.12*q_x-0.07*q_y+0.075*h,
   0.55-0.10*q_x+0.11*q_y-0.065*h,
   0.42+0.06*q_x+0.14*q_y+0.055*h)`.
   This is the registered steep-but-smooth false-positive control.
5. `affine_ring`: with
   `g=exp(-(((u-0.68)^2+(v-0.31)^2-0.18^2)^2)/(2*0.012^2))`,
   `(0.45+0.12*q_x-0.08*q_y+0.11*g,
   0.55-0.10*q_x+0.11*q_y-0.09*g,
   0.42+0.07*q_x+0.14*q_y+0.10*g)`.
6. `zero_linear_quartic`: with `r2=q_x^2+0.7*q_y^2`,
   `(0.46+0.045*(q_x^4-0.2),
   0.53+0.040*(q_y^4-0.2),
   0.44+0.035*(r2^2-0.35))`.
7. `affine_occlusion`: define
   `rho=((u-0.66)/0.12)^2+((v-0.36)/0.16)^2`, use the item-2 affine background, and add
   `(0.16,-0.14,0.12)` where `rho<1`. Its registered edge band is
   `abs(sqrt(rho)-1)<=0.18`; its away region is `sqrt(rho)>=1.35`.
8. `continuous_crease`: with `t=u+0.35*v-0.68` and `k=max(t,0)`,
   `(0.38+0.18*u+0.12*v+0.20*k,
   0.58-0.14*u+0.10*v-0.16*k,
   0.36+0.10*u+0.16*v+0.18*k)`. Values are continuous and first derivatives jump at `t=0`.
9. `isoluminant_step`: color A `(0.20,0.60,0.40)` when `u+0.60*v<0.82`, otherwise color B
   `(0.75,0.45164988814317675,0.25)`. Both colors have Rec.709 luma `0.50052` before rounding.

The smooth-benefit families are items 3--5. Item 6 is the no-linear-trend control. Item 7 is the
decisive mixed-content control; items 8--9 are hard discontinuity controls. Formula strings,
region-mask definitions, and hashes are part of the pre-target binding.

## Frozen stream and rate accounting

Use only the audited benchmark `GFCOV01` configuration:

```text
chart=current_rs, bits_means=12, geometry_bits=(6,6,6), bits_colors=8,
predictor=absolute, coder=zlib9
```

Wrap it in `DSLR015` with the little-endian 20-byte header `<8sBBHIHH`: eight-byte magic
`b"DSLR015\0"`, one-byte version `1`, one-byte mode (`0=NW`, `1=DSL`), zero `uint16` reserved
field, `uint32` inner length, `uint16` height, and `uint16` width. The header is followed by exactly
that many inner bytes and a little-endian `uint32` zlib CRC32 over the complete header plus inner
payload. Reject a wrong magic/version/mode/reserved value, inconsistent inner length or extent,
CRC mismatch, truncation, trailing byte, invalid mode/count pair, or noncanonical inner stream.
The header mode byte is always present and priced. Both variants have no tail. `NW78` and
`DSL78` must contain the exact same inner bytes; their complete blobs differ only in the mode byte
and resulting CRC, and their complete lengths must be exactly equal. This is equal rate inside the
frozen common benchmark syntax, not proof that an existing SSPL header has a free signaling bit.

Every stream requires deterministic encode, cold decode, exact component accounting, canonical
inner re-encode, and decoded-state re-encode. `NW80` is encoded independently at count 80. No
predictor, bit allocation, chart, range, or coder search is legal.

## Frozen numerical, quality, and no-harm gates

Use independent float64 dense-weight and production-order float32 renderers. Production integer
radii and the ordered contributor identities are computed once from the cold decoded float32
state. The float64 reference promotes that exact state, recomputes conics and weights on the same
semantic contributor list, and accumulates in float64; it may not recompute AABB, radius, support,
or contributor membership in a different precision. All pixels enter MSE.
Require every row finite, active count at least one, mass at least `1e-5`, partition error at most
`2e-5`, nonnegative effective weights to `-2e-7`, float32/reference maximum difference at most
`2e-5`, and unchanged contributor hashes inside every DSL78/NW78 pair.
Active-count, mass, and partition predicates apply to every pixel in both arithmetic paths.
Per pixel/path, `active_count=#{i on the frozen contributor list: w_i>0}`,
`mass=sum_i w_i` before the denominator epsilon, and
`partition_error=abs(sum_i p_i-1)` with `p_i=w_i/(mass+1e-8)`; take the worst predicate over all
pixels.
"Nonnegative effective weights" means the normalized Gaussian coefficients `p_i`; it does not
apply to the signed overall OLS diagnostic operator or reinterpret DSL as a signed Gaussian
compositor.

All decision MSE, maximum-error, target-range, excursion, and win values use the unclipped float64
reference output against the exact float64 pixel target. The float32 path is used for
candidate/reference parity, performance, and the mandatory shadow-decision guard below, not to
select the primary numeric values. A win is a strict smaller-MSE comparison with no tie
tolerance. Every median is `numpy.quantile(values,0.5,method="linear")`.

As an actual-decoder agreement guard, recompute every atomic quality/no-harm/range predicate,
strict win sign, win count, aggregate-threshold predicate, and final static quality decision using
the cold float32 candidate outputs (promoted to float64 only for scoring against the exact target).
Persist both numeric metric ledgers; their ratios, medians, and worst values need not be bitwise
equal. Require exact equality only of Boolean atomic predicates, strict win signs, integer win
counts, Boolean aggregate predicates, and the final decision. Any disagreement is a conjunctive
scientific numerical-robustness failure/kill, not invalid plumbing, and may not be resolved by
choosing the favorable arithmetic path. It is unavailable only if an independently preregistered
plumbing control fails.

For attribution only, render two nondecision diagnostics from the same decoded N=78 state:
ungated ordinary-OLS lift and ungated eight-step Cauchy lift. They cannot rescue DSL, enter any
aggregate, or change the decoder. For the OLS diagnostic, report `rank(A)` versus `rank(W)`, minimum
coefficient, and maximum row L1 amplification for
`A=W+(X-WX_mu)(X_mu^T X_mu)^{-1}X_mu^T`. This exposes that the overall color-to-output map can be
signed even though its Gaussian residual weights are positive. Robust DSL adds no rank/DOF.

For a named baseline `B` and registered mask `M`, define
`R(DSL,B,M)=MSE_M(DSL)/max(MSE_M(B),1e-20)`; every gate below names `B` as `NW78` or `NW80`, and
`full`, `outer`, `edge`, or `away` identifies `M`. The outer-ten mask is
`x<10 or x>=97 or y<10 or y>=73`. Medians use NumPy's linear quantile. Every gate is conjunctive:

1. `constant`: DSL maximum absolute error `<=1e-6` in all six units.
2. `affine`: DSL maximum absolute error `<=1e-3` in all six units.
3. For each smooth-benefit family separately, DSL78 beats NW78 in at least `5/6` units; median
   full-image ratio `<=0.90`; median outer-ten ratio `<=0.80`.
4. For each smooth-benefit family, DSL78 beats NW80 in at least `4/6` units and its median
   DSL78/NW80 MSE ratio is `<=0.98`.
5. `zero_linear_quartic`: every DSL78/NW78 ratio `<=1.02`.
6. `affine_occlusion`: in every unit, DSL's away-region MSE is strictly below NW78 and the median
   away-region ratio is `<=0.90`; its edge-band MSE ratio is `<=1.01`; its full-image ratio is
   `<=1.01`; and its output remains in the exact target channel range enlarged by `2/255`.
7. `continuous_crease` and `isoluminant_step`: every DSL78/NW78 full-image ratio is `<=1.01`,
   edge-band ratio is `<=1.02`, and output remains inside the exact target channel range enlarged
   by `2/255`. The edge band is Euclidean pixel distance at most ten from the analytic line. For
   a boundary `u+a*v-b=0`, it is exactly
   `abs(x/106+a*y/82-b)/sqrt((1/106)^2+(a/82)^2)<=10`, using `(a,b)=(0.35,0.68)` for
   `continuous_crease` and `(0.60,0.82)` for `isoluminant_step`.
8. Every DSL output remains in `[-2/255,1+2/255]`; exact target-range excursions are reported for
   every family even where they are not a gate.
9. The DSL/NW complete lengths are equal, their inner streams identical, and all component,
   re-encode, count, and mode gates pass in all `54` units.
10. For every target separately, DSL78 complete bytes are no greater than NW80 in at least `5/6`
    units, with median byte ratio `<=1.00` and worst byte ratio `<=1.02`.

The `NW80` comparison is an actual-rate recipient control, not an assertion that two rows always
cost the nine robust-plane scalars; DSL transmits no such scalars.

## Frozen convergence lane

For all `54` units, freeze cold `N=78` geometry/support and optimize the same decoded float32 node
color tensor from an exact all-zero start. `NW78_opt` renders ordinary colors. `DSL78_opt`
recomputes the exact robust plane, residual graph score, confidence, beta, and residual from its
current colors at every logical step, in float64, and renders through the same dense frozen `W_G`.
Both step-zero outputs must be bitwise-identical all-zero arrays; their losses against the target
must be bitwise identical and are generally nonzero. The analytic sampled colors used by the static
lane do not initialize this lane.

For this lane only, cast `W_G`, the pixel design, the derived DSL beta/residual, and the exact target
once to float32 before rendering and loss evaluation; the robust derivation itself remains
float64. Adam owns the float32 color tensor and both arms' HWC mean-square losses are float32. This
is a production-arithmetic color-subspace lane, distinct from the float64 static decision metrics.
Each arm owns a fresh contiguous all-zero color tensor and fresh Adam state. Compute the common
float64 dense `W_G` once from the cold reference support, cast it once to float32, and do not
recompute conics, radii, support, or contributors inside optimization. Log step `0` before any
update and step `s` immediately after update `s` for `s=1..60`.

Run exactly `60` Adam updates with `lr=0.03`, `betas=(0.9,0.999)`, `eps=1e-8`, no schedule,
weight decay, clamp, early stop, or codec. Instantiate
`torch.optim.Adam(...,foreach=False,fused=False,amsgrad=False,maximize=False)` with float32 state.
At every update call `zero_grad(set_to_none=True)`, backpropagate the currently logged loss, then
call `step()`. Optimize all-pixel RGB MSE. Log steps `0..60`. Define
normalized AUC exactly as
`(0.5*loss_0 + sum_{s=1}^{59} loss_s + 0.5*loss_60)/60`.

Every convergence ratio uses denominator `max(NW_value,1e-20)` and every convergence median uses
the frozen linear median. For the `18` smooth-benefit units require median DSL/NW AUC ratio `<=0.90`, median final-loss ratio
`<=0.90`, and no final ratio `>1.05`. For the `18` mixed/hard units require no final ratio `>1.02`.
Constant, affine, and zero-linear trajectories are required diagnostics. This is update-indexed
color-only convergence from an exact shared output, not wall-clock convergence, geometry fitting,
quantized training, or full StructSplat training evidence.

## Frozen permutation, gradient, and performance gates

Traverse static renderer IDs in lexical order. For every static row, run identity, reverse, and one
PCG64 permutation from global seed `20260716025`; identity and reverse consume no random draw, and
exactly one `permutation(N)` draw is consumed per renderer. Apply an order to every row-aligned
tensor and rerender. In addition, derive the graph/IRLS/confidence nondecision diagnostic from each
arm's decoded ordinary colors, including NW78/NW80, so its invariance is checked without changing
those arms' output. Relative to identity, require float64 output difference `<=1e-11`, float32 output
difference `<=2e-5`, identical four-neighbor edge sets as coordinate pairs, confidence difference
`<=1e-12`, beta difference `<=1e-11`, and identical pass/fail decisions.

Run directional color-gradient checks on exactly two DSL78 cells:
`target_conditioned/affine_chirp/seed401` and
`target_conditioned/affine_occlusion/seed401`. Freeze geometry, graph neighbor identities, and dense
weights. Then append the synthetic fixture below. Traverse those three fixtures in that exact order
and draw two L2-normalized float64 color directions per fixture from one continuous PCG64 stream
seeded `20260716026`. For each fixture/direction call
`rng.normal(0,1,size=(N,3))` in C order, retain float64, and divide by the flattened Frobenius/L2
norm. Do not renormalize after the AD32 cast.

The synthetic fixture has the registered `H=83,W=107,N=49` row-major `7 x 7` Cartesian grid with
`x=linspace(0,106,7)` nested inside `y=linspace(0,82,7)`. At each mean define
`p=2*pi*x/106`, `w=0.02*sin(p)*cos(pi*y/82)`, and colors
`(0.5+w,0.5+0.8*w,0.5-0.6*w)`. Its exact float64 target is the registered `affine_chirp` pixel
formula. Its dense weight from pixel `q=(q_x,q_y)` to mean `q_i` is
`exp(-5*||q-q_i||^2)/(sum_k exp(-5*||q-q_k||^2)+1e-8)`. Before any canonical target or field is
created, persist and bind all six direction arrays, this fixture's means/colors/weights/target
formula, every registered region-mask C-contiguous bool payload and nonzero pixel count, and their
hashes. A mask with zero pixels is a validity failure.

For scalar all-pixel MSE and centered finite differences with `h=2^-16`, require

```text
abs(AD64-FD64) <= 1e-8 + 2e-4*max(abs(AD64),abs(FD64))
abs(AD32-AD64) <= 2e-5 + 3e-3*abs(AD64).
```

`AD64` differentiates the full float64 dense-reference MSE from the promoted cold colors, frozen
float64 `W_G`, and exact float64 target; `FD64` perturbs that exact state. `AD32` casts the same cold
colors and direction to float32, performs the decoder derivation in float64, casts beta/residual and
`W_G` once to float32, and differentiates the float32 dense production-arithmetic HWC MSE against
the once-cast float32 target. It is not a separately tuned gradient path.

The check includes all eight IRLS updates and the smooth confidence graph with respect to colors;
it does not differentiate neighbor selection or support membership with respect to geometry.
For every direction, persist the lower/upper quantile-support row identities used by the error
median and absolute-deviation median in every IRLS iteration/channel at the base, `+h`, and `-h`;
for odd `N` the lower/upper identities are the same central row. Report any identity change as a
quantile-order kink crossing without changing the registered numerical decision.
On the synthetic fixture, at least one channel must satisfy `0.1 < alpha_j < 0.9`. Also compute
the three-channel directional derivative of `alpha` itself by autograd and the same centered finite
difference. Each channel must meet
`abs(AD_alpha-FD_alpha)<=1e-8+2e-4*max(abs(AD_alpha),abs(FD_alpha))`, and each of the two directions
must have `max_j abs(AD_alpha_j)>=1e-4`. Thus a nonzero smootherstep derivative, rather than only a
flat confidence endpoint, is audited.

Time single-thread CPU render and cold `decode+derive+render` under `torch.no_grad`, with Torch
intra-op and inter-op threads pinned to one and `perf_counter_ns`. Traverse cells lexically. For
the whole timing assay instantiate exactly one
`numpy.random.Generator(numpy.random.PCG64(20260716027))`; never reseed it. Traverse each cell in
phase order
`warmup_render`, `record_render`, `warmup_cold`, `record_cold`, drawing one `permutation(ARMS)` per
round. Use 10 warmups and 40 recorded repetitions per arm/unit and persist every schedule and
recorded sample.

For render-only timing, prepare outside the clock the cold decoded arrays, common conics/radii,
DSL-derived beta/residual, and shared normalized pixel coordinates. Start immediately before fresh
contributor enumeration and end after the output is materialized; do not cache the contributor
list. NW performs only the ordinary normalized RGB render. DSL performs the identical residual
render plus its six-multiply/nine-add plane. For cold timing, start from immutable wrapper bytes,
exclude canonical re-encode/audit work, and include parsing, field decode, DSL derivation where
applicable, conics/radii, fresh contributor enumeration, rendering, and the plane. The convergence
lane uses cached dense `W_G` and is not a timing path. Persist 40 derivation-only diagnostic samples
per arm/unit outside both decision timings.

Require median over units of DSL78/NW78 median render time `<=1.15` and median cold
decode+derive+render ratio `<=1.50`. Operation accounting must show identical Gaussian-pixel weight
evaluations, four accumulated scalar channels, exactly six plane multiplications and nine plane
additions per pixel for DSL, and no per-pixel solve/factorization. Report peak RSS and color-fit
step time diagnostically.

## Controls, artifacts, and replay

The source-bound inventory is exactly:

```text
benchmarks/__init__.py
benchmarks/affine_carrier_core.py
benchmarks/decoder_synchronized_lift_core.py
benchmarks/decoder_synchronized_lift_assay.py
benchmarks/gauge_free_covariance_codec.py
benchmarks/gauge_free_covariance_core.py
src/structsplat/__init__.py
src/structsplat/config.py
src/structsplat/density.py
src/structsplat/gaussians.py
src/structsplat/init.py
src/structsplat/render.py
src/structsplat/sampling.py
src/structsplat/structural_controls.py
src/structsplat/structure_tensor.py
tasks/BENCH-015-decoder-synchronized-affine-lift.md
pyproject.toml
```

Hash all of those paths into the canonical binding and archive them, plus
`tests/test_decoder_synchronized_lift_core.py` and
`tests/test_decoder_synchronized_lift_assay.py`, before target evaluation. The same binding also
contains the canonical formula/mask/direction records, exact expected-ID inventory, stable
environment fields, and one binding SHA-256. No unlisted local source may implement the assay.

Before science, controls must establish:

1. exact constant and exact affine decoded-color fixtures reproduce their functions;
2. a smooth fixture has finite nonzero confidence and a step fixture reaches `alpha<=0.01`;
3. row permutation leaves graph coordinate edges, beta, confidence, and output invariant;
4. rank-deficient/duplicate-mean/nonfinite inputs are rejected and distance ties follow the frozen
   decoded-coordinate rule;
5. malformed magic/version/mode/reserved/length/CRC/trailing bytes and noncanonical inner streams
   are rejected;
6. the decoder signature accepts blob and device only, and source-target/decoder-OLS injection is
   impossible by API construction.

Rejecting a positive or accepting a negative control makes the assay invalid, not a scientific
failure.

Before target creation, persist task/source hashes, environment, exact config, formula strings and
hashes, expected stable keys/counts, and an executed-source archive. Persist fields, complete
streams, cold states, target arrays, robust-fit/confidence ledgers, static outputs/errors, stream
components, permutations, convergence rows/checkpoints, gradients/directions, timing
schedules/samples, controls, analysis, artifact manifest, independent replay, and a completion
marker written last.

After valid plumbing controls, do not short-circuit on an unfavorable scientific predicate: execute
the complete matrix and all independent cells. A valid decoded field on which the registered DSL
preconditions fail is a scientific `method_unavailable` failure, not invalid plumbing. Write the
terminal static row and prescribed `not_run_due_to_method_unavailable` rows for every dependent
permutation/trajectory/gradient/timing key, continue all independent cells, and count those rows as
scientific failures. Missing expected keys remain artifact invalidity.

Write raw arrays and append-only ledgers first. Then write an artifact manifest that excludes
`artifact_manifest.json`, `replay.json`, `analysis.json`, and `completion.json`; validate it; run
independent replay; and only if replay passes interpret scientific aggregates into `analysis.json`.
On replay/control failure an `analysis.json` marker may contain only an `unavailable` classification
and replay/control diagnostics, never an outcome aggregate. Write `completion.json` last.

Every expected key has exactly one terminal row. Missing, duplicate, nonfinite, stale-binding,
source-mismatched, silently excluded, or wrong-count rows invalidate the assay. Independent replay
must rebuild all `60` fields (`54` target-conditioned plus `6` shared) and byte/hash-compare them,
independently reapply every target-at-original-mean color overwrite, deterministically reconstruct
every wrapper, and cold-decode/re-encode every stream. It must recompute every
robust fit/graph/confidence/static render/permutation/convergence/gradient and aggregate, validate
timing schedules and arithmetic, audit the decoder call graph for target dependence, and validate
the source archive and artifact manifest before analysis is readable. Exact expected-ID set equality
is required before aggregation.

## Claim boundary and stop rule

A full pass establishes only that, on this synthetic disjoint matrix and benchmark codec:

- decoder-derived robust first-order reproduction improves registered smooth targets at exactly
  equal complete byte length and competes with the registered two-row control;
- the target-blind smooth discontinuity confidence meets the registered no-harm/range guards;
- its color-only fitting curves, CPU reference cost, gradients, and row-order behavior meet the
  frozen gates; and
- the same `N x 3` transmitted colors can express a more useful first-order inductive basis on
  these targets.

DSL adds no transmitted parameter and no linear appearance dimension. It cannot inherit
BENCH-014's `+6 DOF` claim; "expressiveness" here means reproduction behavior at fixed parameter
count, not rank expansion. Equal length is inside `DSLR015`, whose mode byte is explicitly priced;
it is not a production SSPL compression claim.

The assay cannot establish natural-image value, production/GPU performance, broad robustness,
full-fit convergence, an available signaling bit, learned-codec superiority, or publication
novelty. A full pass only authorizes freezing a disjoint natural-image/production-codec Stage 1.

Any scientific failure closes this exact robust-IRLS, residual four-neighbor log-mean-exp,
`1.20--1.50`-smootherstep decoder-synchronized global first-order family. Do not rescue it by
retuning the robust loss, scale, iterations, neighbor count, temperature, thresholds, formulas,
counts, codec, optimizer, or gates. A plumbing-invalid run may be replaced only after repairing
the validity defect under a new source binding and untouched scientific protocol.
