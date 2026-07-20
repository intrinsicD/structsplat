# BENCH-009: Rate/DOF-priced residual tangent-space auction

**Status: completed negative/unavailable (2026-07-16).** Stage 1 and its complete recovery ledger
ran on the frozen development split. The causal-validity gate failed, the literal incremental
projector statistic violated its non-negativity invariant, and neither candidate passed the frozen
utility decision. No production method, expressiveness classification, or actual-rate result is
authorized. The exact audit is in
[`docs/research/2026-07-16-bench009-results-audit.md`](../docs/research/2026-07-16-bench009-results-audit.md).

## Decision question

After the FIT-018--020 and COMP-006 negatives, is StructSplat's remaining error primarily:

1. reachable by better optimization of the existing constant-color Gaussian grammar;
2. outside that grammar but reachable by a local affine appearance jet;
3. outside it but reachable by a localized oscillatory carrier; or
4. best handled by allocating another standard Gaussian?

The experiment must answer this before a CUDA/codec implementation of a richer primitive is
authorized.

## Central claim and null

**Claim.** At a near-plateau fitted field, at least one six-appearance-scalar extension family
captures residual energy outside the scaled tangent space of the current grammar and realizes at
least `+0.15 dB` more PSNR than both the existing-tangent and generous standard-birth controls.

**Null.** After removing residual directions already reachable by the current grammar, no richer
family yields a stable, realized, recovery-persistent advantage over standard births at matched
appearance DOF and explicitly reported selection/payload cost.

## Core diagnostic

For a frozen parent field `theta`, target `t`, render `R(theta)`, residual `r`, and scaled Jacobian
`J = dR / dtheta`, define the incremental explanatory power of a **zero-offset linear** extension
family `A_k` as

```text
Delta_k = ||P_[J,A_k] r||_2^2 - ||P_J r||_2^2.
```

Every extension column is residualized against `J` before it is scored. Raw correlation with the
residual is not sufficient: it confounds a representation deficit with a direction that a better
optimizer could already reach.

This is an evidence program built from known variable-projection, Gauss--Newton, and matching-
pursuit tools. It is not claimed as a new optimizer or new linear-algebra method.

A finite normalized birth is an affine action with a nonzero denominator-induced offset even when
its color is zero. Its exact finite gain is reported separately and must not be labeled this
orthogonal-projector `Delta_k`.

Keep two non-interchangeable ledgers:

1. **Causal-capacity ledger:** the unpriced full-`J` baseline versus jointly realized
   `J + affine`, `J + carrier`, and `J + two-birth` actions. Literal unregularized projector
   `Delta_k` is available only for the zero-offset affine/carrier columns; birth uses its separately
   named finite affine-action gain.
2. **Matched-six-DOF ledger:** `tangent6`, `affine6`, `carrier6`, and two-birth `RGB6`, without an
   extra full-`J` update in any arm. This ledger owns equal-appearance-DOF comparisons and the
   representation-survival gate.

Do not select a winner by mixing the unpriced full tangent, a residualized extension, and a
six-coefficient action in one scalar table.

## Frozen action families

The primary action block has six appearance coefficients.

1. **Current tangent.** Report the full damped current-grammar projection as an unpriced
   optimization-capability upper bound. Also report a six-direction realized trust-region control;
   never compare the full tangent's DOF to a six-scalar packet as if they were equal-rate methods.
2. **Affine jet.** Upgrade one existing Gaussian with local scale-normalized `x/y` RGB slopes
   (`2 x RGB = 6` coefficients). Search only the frozen parent rows.
3. **Carrier.** Add scale-normalized Gaussian-windowed sine/cosine RGB appearance to one existing
   envelope (`sin/cos x RGB = 6` coefficients). Frequencies/orientations come from a preregistered
   finite bank; the selected bank index is priced and bank-size sensitivity is reported.
4. **Standard birth.** Add two fixed-geometry constant-RGB Gaussians (`2 x RGB = 6` fitted
   coefficients). Candidate geometry is selected from the same frozen discovery data and is
   deliberately treated as free in the primary DOF comparison, making this a generous native
   control. Geometry/type/index payload is still counted in the provisional packet view.

The standard-birth arm must use the exact finite normalized-renderer solve, not the degenerate
zero-opacity tangent. For one candidate raw weight field `u`, base denominator `D`, and base render
`y`, let

```text
a = u / (D + u + eps)
y_new = (1 - a) y + a c
c* = sum_p a_p [t_p - (1 - a_p)y_p] / sum_p a_p^2
```

per color channel. For two births, use their joint finite denominator and solve the resulting
two-column RGB least-squares problem exactly. Birth identity and coefficients are solved on the
discovery data under the parent-frozen prediction model. After one joint uniform trust rescale,
the step-0 coefficient vector is immutable. Rebuild native post-update parent/birth supports,
weights, and denominator only to realize those coefficients and verify affine-formula/direct-render
parity; native support changes must never cause a second solve.

A local maximum-entropy/barycentric compositor is a high-risk follow-up candidate, not a primary
arm. It may enter only as a separately labeled forward diagnostic after its local convex-hull and
boundary feasibility are measured without ghost sites.

## Parameter scaling, gauges, and projection

- Freeze one parent-local coordinate chart before looking at action outcomes. One dimensionless
  mean unit is a displacement along a parent's principal Gaussian axis,
  `R(theta_parent) diag(sx_parent, sy_parent) z_mean`; one log-scale unit is `1`, one rotation unit
  is one radian, and one RGB unit is one display-linear unit. Affine slopes (per local sigma),
  carrier amplitudes, birth RGB, and current-grammar RGB therefore share the same appearance unit.
  This chart is frozen at the parent, and a joint action receives one uniform infinity-norm scale
  over its entire concatenated coefficient vector. Never scale global `x/y` elementwise by
  `sx/sy` for a rotated row, and never clip coordinates independently. Opacity, if later enabled,
  uses probability rather than raw-logit units. Damping acts in this same raw dimensionless parent
  chart. Column-norm scaling may stabilize rank discovery, but a separately labeled
  image-sensitivity-normalized ridge is not interchangeable with the frozen damping sweep.
- The primary parent has no learned opacity, avoiding the normalized compositing global-opacity
  gauge. An opacity sensitivity arm may be added only after an explicit gauge constraint is fixed.
- The small Stage 0 reference deliberately includes finite opacity solely to verify that truncated
  image-space SVD removes a real quotient/gauge nullity. This is a calibration control, not the
  Stage 1 primary-parent parameterization.
- Persist each parent's exact support membership `M_parent[N,H,W]` and its SHA-256. J/JVP,
  projectors, candidate scoring, and **linear predictions** use `M_parent`. Finite realization
  applies the already-frozen coefficient vector once through the native normalized renderer,
  recomputing compact AABBs from updated means/scales/rotations; support changes never trigger
  coefficient reselection or refitting. Record `added_memberships=sum(M_native & ~M_parent)`,
  `removed_memberships=sum(M_parent & ~M_native)`, their XOR sum, and both membership hashes.
  Birth memberships are reported separately rather than counted as parent changes. Recovery uses
  native dynamic support at every optimizer step and additionally records changes relative to the
  previous checkpoint. Frozen-support finite renders are diagnostic only and cannot enter gates.
- Validate JVPs against centered finite differences before interpreting a projection.
- Stage 1 uses matrix-free JVP/VJP range discovery and damped reduced solves. Relative rank
  thresholds apply to the seeded, fixed-probe randomized range's reduced scaled SVD; the full
  scientific Jacobian is never materialized. Materialized truncated SVD is reserved for small
  preflight/Stage-0 oracles. Repeat the decision across the frozen `10x` damping/rank-threshold
  range.
- Candidate selection **and all six fitted coefficients** use only the discovery tiles for a
  fold; the resulting frozen action is scored only on the held-out tiles. Fitting coefficients on
  the held-out tiles and then scoring those same tiles is leakage, not cross-fitting. The no-fold
  oracle independently selects both identity and coefficients on all pixels and is named
  `all_pixel_identity_and_coefficient_oracle`; it is explicitly excluded from every gate. It must
  not be described as reusing a discovery identity (doing so for both fold identities would double
  the oracle ledger). This prevents a larger carrier bank from winning only through search
  multiplicity.
- Report candidate type/index/frequency choices and canonical fixed-width trial-packet bytes. Those
  bytes are diagnostic marginal pricing, not SSPL1/SSPL2 actual rate.

## Stage 0: diagnostic validity

Before fitting natural or mixed targets, construct deterministic small fields with one known source
of error:

1. a finite current-parameter perturbation;
2. one affine local RGB jet;
3. one fixed-envelope sine/cosine RGB carrier;
4. one or two finite normalized standard births;
5. a null target already represented by the parent.

The assay must recover the generating family, produce no extension gain on the null within numeric
tolerance, and validate the finite birth solve against a direct render. Stage 0 is allowed to use a
materialized CPU Jacobian and does not support a quality, convergence, or compression claim.

### Stage 0 result (2026-07-15)

The source-bound `13 x 13`, three-Gaussian CPU control passed all frozen instrument checks:

- all five known-source targets were identified correctly;
- centered finite-difference/JVP relative L2 error was `7.4676e-11` and maximum absolute error was
  `2.2066e-12`;
- maximum packet realization error was `3.3307e-16`;
- image-space tangent nullity was five and residualized-column orthogonality error was
  `5.6066e-15`;
- `7` targeted tests passed.

The control targets are self-generated by the fitted packet definitions. It tests one fixed
three-RGB birth, not the primary joint two-birth solve, and it checks the base and packet finite
renders separately rather than a joint nonlinear update. Its regularized/clipped step score must
not be called the literal projector statistic. Therefore this result confirms only numerical
plumbing/identifiability and authorizes implementing Stage 1. The development run remains locked
until the no-opacity, joint-realization, two-birth, cross-fit, damping, and two-radius paths pass
their own preflight; method promotion remains false. Reproduce with:

```bash
PYTHONPATH=src:. python -m benchmarks.residual_tangent_auction \
  --outdir results/bench009_tangent_auction_stage0
```

## Stage 1: disjoint development screen

Freeze hashes before running for new `64 x 64` targets disjoint from FIT-020 and COMP-006:

- ramps / smooth shading;
- steps and junctions;
- thin curves;
- stationary sinusoids;
- chirps / changing frequency;
- natural texture patches from a development-only source.

Use `N=64`, seeds `{101,211}`, and fixed parent checkpoints `{24,160}` from **one continuous
160-step Adam trajectory per target/seed**: one deliberately underfit and one near-plateau
candidate (the name does not authorize adaptive plateau selection). The step-24 optimizer state is
continued, not reset, on the way to step 160. Keep
initialization, normalized renderer, support policy, parent count, fitter,
metric convention, and recovery optimizer identical across action families. Do not tune these
targets after observing the gate.

The two spatial folds are `8x8`-pixel checkerboard tiles. The frozen numerical sensitivity grid is
the full cross-product of dampings `{1e-3,1e-2}`, reduced-SVD relative rank thresholds
`{1e-5,1e-4}`, and dimensionless trust radii `{0.25,0.75}`. Trust clipping uniformly rescales a
coefficient vector to the radius in infinity norm; it never clips coordinates independently. The
positive predicted-gain floor is `max(1e-12, parent_mse * 1e-5)` and all nonpositive/near-zero rows
remain in the raw output.

For every parent/action, save:

- base and post-action PSNR/MSE;
- projected residual-energy reduction and predicted PSNR;
- finite realized gain at two frozen dimensionless trust radii;
- action/search identity, effective rank, damping, coefficient norm, and condition estimate;
- short and long matched recovery (`20` and `100` steps) from an explicitly identical optimizer-
  state policy;
- six coefficient DOF, candidate-bank size, provisional packet bytes, support evaluations, and
  decode arithmetic;
- wall time and peak memory for the diagnostic itself.

Every row declares `causal_capacity` or `matched_six_dof`. Both ledgers use a joint finite render of
all updates claimed by that row. The raw output must include the expected cell count for
`family x seed x parent_horizon x fold x action x damping x rank_threshold x trust_radius`;
incomplete or duplicate cells fail closed rather than being averaged away.

The frozen minimum workload ledger has named, non-interchangeable components:

- `3,072` immediate cross-fit action cells (`24` parents x `2` folds x `8` actions x `2`
  dampings x `2` rank thresholds x `2` trust radii);
- `1,536` all-pixel-oracle cells (`24 x 8 x 2 x 2 x 2`), excluded from scientific gates; and
- `816` recovery trajectories (`24 x 2` folds x `2` trust radii x `8` actions, plus `24 x 2`
  no-action controls). Each trajectory links its step-0 state to an immutable immediate-action (or
  parent-baseline) row and writes new step-20 and step-100 checkpoints. This is `1,632` newly
  rendered recovery records and `2,448` logical step `0/20/100` records after following the
  required links; step 0 must not be silently omitted or redundantly rerendered.

Thus the minimum combined record/trajectory ledger is `5,424`, not an action-cell count. The three
components must be validated independently; their sum must never be presented as though all rows
had the same schema or statistical role.

The `1,152` separate linear/finite diagnostic-score rows also have heterogeneous meanings. Of
these, `864` are unregularized non-birth linear scores (`576` cross-fit and `288` all-pixel
oracle), while `288` are finite birth affine-action scores (`192` cross-fit and `96` oracle).
Birth rows are never counted or named as projector evaluations.

### Frozen recovery and evidence contract

Each required recovery trajectory starts from the exact cross-fit immediate action at
`damping=1e-3`, `rcond=1e-5`, and its stated trust radius. The discrete identity and the complete
step-0 coefficient vector are immutable; selection is never rerun. The no-action control starts
from the same parent. Use a fresh Adam state in every arm, with no inherited moments, schedule, or
weight decay, and the following fixed groups: means `5e-2`, log-scales `3e-2`, rotations `1e-2`,
base RGB `3e-2`, and any affine/carrier/birth RGB packet coefficients `3e-2`. Adam uses
`betas=(0.9,0.999)` and `eps=1e-8`.

Optimize discovery-tile RGB MSE for one continuous `100`-step trajectory and score held-out tiles
without refitting at logical steps `0`, `20`, and `100`. The all-pixel oracle, if recovered as an
optional complete component, optimizes and scores all-pixel MSE and is excluded from every gate.
Clamp base log-scales after every update to `[log(0.35), log(64)]`. Disable topology events,
opacity, QAT, color solves, early stopping, pruning, relocation, and packet reselection. Birth
geometry is fixed; selected affine row and carrier row/frequency are fixed. Continuous base and
packet coefficients may keep optimizing after step 0.

The parent support hash is frozen for the Jacobian, linear prediction, and immediate surrogate.
The step-0 finite action and every recovery step use the native renderer with recentered/recomputed
AABBs, exactly as specified above, and record support-membership changes relative to the parent.
Exact step-0 linkage means reconstructing the same finite state and render as the referenced
immediate row, not reusing frozen support during nonlinear recovery. Each trajectory stores a
`step0_source_cell_id` and copies the source state, packet, native-support, native-render, and
metric hashes. Recovery setup rerenders the state once and requires exact CPU state/support/render
hash agreement; that verification creates no second logical checkpoint row. Baselines link to the
parent's native-render row.

The no-fold oracle selects both identity and coefficients on all pixels and is labeled
`all_pixel_identity_and_coefficient_oracle`; it is not a post-selection refit. Required cross-fit
recovery remains `816` trajectories. Optional oracle recovery is interpretable only if complete:
`24 x (2 radii x 8 actions + 1 baseline) = 408` trajectories and `1,224` logical checkpoints.

For every scoring mask, define

```text
gain_floor = max(1e-12, parent_mse * 1e-5).
```

Here `parent_mse` is measured on that same scoring mask. Rows at or below the floor remain in raw
output and count as nonpositive. Exclude them only from
realized/predicted ratios and Spearman calculations; fewer than three eligible non-tied rows makes
that validity statistic fail. For survival statistics, pair an affine/carrier result with the
stronger tangent6/birth control at the same family, seed, fold, radius, and checkpoint. Average the
four seed-by-fold pairs within each of the six families, then run `100,000` paired bootstrap
resamples of the six family means with NumPy PCG64 seed `20260715`; use the `2.5` percentile as the
95% lower endpoint. Reuse the same sampled family indices across actions/checkpoints and evaluate
the two radii separately.

Canonical little-endian diagnostic marginal packets are fixed as follows: tangent6 is type `u8`,
six `u16` parameter indices, and six `f32` coefficients (`37` bytes); affine is type `u8`, row
`u16`, and six `f32` coefficients (`27` bytes); carrier adds bank `u8` and frequency `u16`
(`30` bytes total); birth is type `u8`, two geometries as ten `f32`, and six RGB `f32` values
(`65` bytes). The full-`J` part of causal joint actions remains unpriced, so none of these is a
self-contained stream. Freeze dictionary sizes at tangent `512`, affine `64`, full carrier
`64 x 32 = 2,048`, small carrier `64 x 8 = 512`, and birth `C(8,2) = 28`.

Every expected cell has a stable key and a terminal `ok` or `error` record. Missing, duplicate,
stale-binding, nonfinite, hash-mismatched, or error rows invalidate that component rather than
being dropped. Record search work separately from selected-action decode work using a hashed,
benchmark-local counting specification; include candidate fits/linear solves, base and birth
support-weight evaluations, affine/carrier basis samples, carrier sine/cosine calls, RGB FMAs,
normalized divisions, rendered pixels, wall time, and peak RSS.

## Frozen validity gates

The assay is invalid and stops if any condition fails:

- centered finite-difference/JVP relative error exceeds `1e-3`;
- predicted-versus-realized action gains have Spearman correlation below `0.8`, either globally
  or within any `action x parent_horizon` stratum;
- median realized/predicted gain falls outside `[0.5, 2.0]`, either globally or within any
  `action x parent_horizon` stratum;
- the winning family changes across the frozen `10x` damping range, `10x` rank-threshold range, or
  two trust radii in more than one of six target families;
- the known-source Stage 0 fixtures are not identified correctly;
- the exact finite birth solve does not match a direct normalized render within the renderer's
  established tolerance.

## Frozen representation-survival gate

An affine or carrier grammar survives only in the matched-six-DOF ledger and, on near-plateau
parents, it:

- removes at least `25%` more residual energy than the stronger of the six-direction tangent and
  finite-birth controls;
- realizes at least `+0.15 dB` paired PSNR over that control;
- retains a family-bootstrap lower confidence bound above zero;
- is positive in at least four of the six target families; and
- remains positive after both `20`- and `100`-step matched recovery.

Failure closes that richer-grammar family on this screen. It cannot be rescued by changing the
candidate bank, damping, trust radius, parent horizons, targets, or recovery horizon.

### Adversarial interpretation boundary

Frozen before the Stage-1 development run: passing the matched-six-DOF survival gate is a utility
screen, not by itself proof of new representational expressiveness. A six-column affine or carrier
block can lie entirely inside `span(J)` and still beat six selected coordinate columns because it
is a better six-dimensional basis. The machine-readable analyzer must therefore keep
`scientific_interpretation_authorized=false` until the separate causal projector/joint-action
ledger demonstrates meaningful energy outside `J` and survives finite realization.

The causal selector searches identities by discovery-residual fit rather than directly maximizing
residualized-against-`J` novelty. A weak causal result can therefore reject the selected identity,
but cannot prove that every identity in the family lacks a novel direction. This asymmetry must be
reported; it cannot be repaired by reselection after seeing the run.

The parent trajectory optimizes the repository's frozen `0.7 L1 + 0.3 (1-SSIM)` objective, while
the auction and fresh recovery optimize RGB MSE. Consequently, checkpoint 160 is a near-plateau
under the parent objective, not an established MSE plateau. BENCH-009 may select a useful next
mechanism, but a representation-versus-optimization conclusion additionally requires an
objective-aligned parent or optimizer diagnostic on disjoint evidence.

## Interpretation

| Outcome | Next authorization |
|---|---|
| Full current tangent explains the residual and realized current-grammar recovery wins | Test reduced-manifold / variable-projection fitting; do not add syntax. |
| Affine survives all gates | Design the smallest codec-native color jet, with max-entropy linear-precision compositor as a no-extra-attribute challenger. |
| Carrier survives all gates | Build a WIPES-controlled frequency-bearing grammar and real versioned stream. |
| Finite births win | Keep the constant Gaussian grammar; investigate discrete allocation/search under a new hypothesis. |
| Projection does not predict realized/recovered gain | Abandon local tangent selection; investigate topology changes or optimizer dynamics. |
| No family survives | Stop richer-atom implementation until a materially new residual mechanism is identified. |

## Compression boundary

This task cannot demonstrate compression. SSPL1 rejects affine/carrier state, and provisional
packet bytes are not a self-contained codec. A surviving grammar authorizes a separate versioned
cold-decode task that counts base stream, type/index syntax, quantizers, entropy state, side
information, and decoder work against the strongest standard-Gaussian precision/birth control.

## Depends on

CORE-001, CORE-006, FIT-005, FIT-017, FIT-020, BENCH-002, COMP-006.

## Stage 1 result and closure (2026-07-16)

The frozen v3 workload completed with exact expected coverage:

- `4,608` immediate action cells (`3,072` cross-fit and `1,536` excluded oracle);
- `1,152` diagnostic-score rows and `288` carrier-bank sensitivity cells;
- `816` recovery trajectories and `2,448` logical step-`0/20/100` checkpoints;
- `1,632` newly rendered recovery checkpoints; and
- a complete `3,072`-row matched-evidence union.

The independent artifact audit found no missing cells, stale bindings, shard-union differences,
parent-file changes, or step-0 join errors. The governing scientific decision nevertheless fails:

1. global predicted-versus-realized Spearman correlation is `0.268549 < 0.8`, and all causal
   action-by-horizon strata fail;
2. independently truncated base/joint projector spaces yield negative incremental energies for
   affine and carrier at both rconds, so neither outside-`J` nor inside-`J` is established;
3. affine loses `-0.631/-0.779 dB` immediately and remains negative at step 20 against its
   stronger matched control;
4. carrier loses `-0.184/-0.315 dB` immediately and remains negative at step 20; its positive
   step-100 mean does not survive both radii and all earlier horizons; and
5. the L1+SSIM-parent versus MSE-assay mismatch prohibits an optimizer-deficit conclusion.

The exact causal audit returns expected exit code `2` and SHA-256
`b209cc57a865ce0c9cd28a9a9fd65a3d25ac0d86b0a6c2d03047b0842025ec72`. The executed pre-status-
update source bytes are preserved in the results directory as `executed_sources_v3.tar`, SHA-256
`20b3b08ec57f286e25249b76d927d525459bfdd62e6baf91899bc94fdceec4f3`.

BENCH-009 is closed. BENCH-011 may test the minimal nested-subspace measurement repair on the
already-spent parents, but cannot rescue this result. A failed BENCH-011 calibration closes that
formulation without retuning; a pass only permits preregistering a disjoint objective-aligned
assay.
