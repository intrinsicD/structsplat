# BENCH-011: Nested residual-extension spent-data diagnostic

**Status: v2 completed negative (2026-07-16); formulation closed without retuning.** V1 was
retired after an audit found a base-factorization seed mismatch. Corrected v2 reproduces
BENCH-009's exact cross-fit unit IDs and base ranks, passes every algebra/integrity/replay check,
and fails all four frozen calibration strata. This spent-data postmortem cannot rescue BENCH-009,
authorize a grammar change, or support a quality, convergence, performance, compression, or
novelty claim.

## Question

BENCH-009's literal statistic independently truncated the scaled base and joint matrices. The two
retained image subspaces were therefore not guaranteed to be nested, and the reported difference
`||P_[J,A] r||^2 - ||P_J r||^2` became negative in exact saved rows. Does the simplest algebraic
repair—a single immutable base projector followed by a directly residualized six-column
extension—produce a non-negative capacity statistic whose discovery-fitted physical update
calibrates to native held-out rendering?

This is Frisch--Waugh/partial-regression algebra applied to the existing assay, not a new method.

## Frozen scope

- Inputs: BENCH-009 Stage-1 `near_plateau` parents only, six development families, seeds
  `{101,211}`, and both checkerboard folds: `24` cross-fit units.
- Candidates: the already selected `affine6` and full-bank `carrier6` identity for each unit. No
  identity search, bank change, or reselection is permitted.
- Numerical setting: `rcond=1e-5`, dimensionless-chart damping `1e-3`, trust radii `{0.25,0.75}`.
- Work: `24 x 2 candidates x 2 radii = 96` native renders. There is no recovery phase.
- The development inputs are spent. Results are diagnostic even if every gate passes.

## Immutable nested statistic

For discovery rows, let `Q_J` be BENCH-009's already defined retained base left basis at the frozen
primary `rcond`. It is never recomputed after seeing the extension. For the selected physical
six-column design `A`, compute in float64

```text
r_perp = (I - Q_J Q_J^T)^2 r
B      = (I - Q_J Q_J^T)^2 A
B      = U S V^T
keep_i = S_i > 1e-5 * sigma_max(A)
Q_B    = U[:, keep]
Delta_nested = ||Q_B^T r_perp||_2^2.
```

There is no extension-column normalization and the threshold is relative to the original physical
`A`, not to residualized `B`. `Q_J` and `Q_B` must be mutually orthonormal within

```text
tol = 64 * eps_float64 * max(discovery_rows, parameter_count, 6).
```

`Delta_nested` must be non-negative up to this tolerance. A negative value beyond tolerance, a
rank increase outside `[0,6]`, non-finite state, or failed orthogonality makes the whole diagnostic
unavailable.

## Frozen physical solve and held-out score

Use the nested image basis `[Q_J,Q_B]`. The base physical coordinates may act only through `Q_J`;
the extension may act through both blocks. With the compact discovery design

```text
D = [[Q_J^T J, Q_J^T A],
     [       0, Q_B^T A]],
```

solve one `L2`-damped least-squares system in the raw dimensionless-parent-plus-RGB chart with
`lambda=1e-3`. Uniformly scale the complete `(base_step, extension_coefficients)` vector to each
frozen L-infinity radius. Do not refit either block after scaling.

On held-out rows, predict with the discovery-fitted physical coefficients only:

```text
prediction = J_heldout base_step + A_heldout extension_coefficients.
```

`Q_J`, `Q_B`, and any coefficients must never be fitted or recomputed on held-out rows. Native
rendering uses dynamic post-update support exactly once and is compared with the parent on the
same held-out mask.

## Required algebra controls

Before any repository parent is processed, deterministic float64 controls must establish:

1. invariance of `Delta_nested` to extension-column permutation and sign;
2. `A = Q_J C` gives extension rank zero and `Delta_nested = 0` within tolerance;
3. singular values just below/above `1e-5 * sigma_max(A)` are rejected/retained as specified;
4. nested projection agrees with an explicit concatenated orthonormal basis; and
5. the physical solver returns finite coefficients and obeys one joint trust scaling.

Failure stops the run before the 96 native renders.

## Frozen gates

For each `candidate x trust_radius` stratum across all `24` held-out rows:

- include a row only when `predicted_mse_gain > max(1e-12, base_mse * 1e-5)`;
- require at least three non-tied eligible predictions and realizations;
- require Spearman rank correlation `>= 0.8`; and
- require median `realized_mse_gain / predicted_mse_gain` in `[0.5,2.0]`.

All four strata must pass. Report excluded rows and all negative realized gains; never delete them.
Also report family means, but do not bootstrap or promote a utility result from this spent-data
diagnostic.

## No-rescue rule and next authorization

- If any algebra control or any calibration stratum fails, close this formulation. Do not retune
  `rcond`, damping, trust radii, bank, identity, eligibility floor, or correlation/ratio gate.
- Only if every frozen gate passes may a separate task preregister a disjoint,
  objective-aligned assay. That assay must use new targets and must compare the unchanged current
  identities before any residualized-bank reselection is considered.
- Even a pass does not authorize affine/carrier implementation. It says only that the repaired
  measurement is coherent enough to test on new data.

## Evidence boundary

Provisional coefficient counts or selector bytes are not actual rate. CPU matrix-free work is not
a performance measurement. Immediate held-out MSE is not a convergence result. BENCH-011 cannot
alter StructSplat defaults.

## v1 audit failure (2026-07-16)

The deterministic algebra controls all passed, including sign/permutation invariance, exact zero
for `A` contained in the fixed base span, the frozen singular-value threshold, explicit nested-
projector agreement, finite physical coefficients, and one joint L-infinity trust scale.

All four v1 source-bound shards completed with `24/24` terminal units, `96/96` unique native rows,
no row errors, and non-negative nested deltas. The canonical binding is
`2e59a751f1675538c9bfe0047ab4cf9b1875216c1257da7ff57ac8a673bac57c`. The frozen calibration
summary was:

| Candidate | Radius | Eligible / 24 | Spearman | Median realized/predicted | Negative realized / 24 | Gate |
|---|---:|---:|---:|---:|---:|---|
| affine | `0.25` | `16` | `0.400` | `0.206` | `14` | fail |
| affine | `0.75` | `14` | `-0.305` | `-21.263` | `24` | fail |
| carrier | `0.25` | `16` | `0.385` | `-0.172` | `16` | fail |
| carrier | `0.75` | `14` | `-0.204` | `-25.229` | `24` | fail |

These numbers are descriptive invalid-run diagnostics, not the BENCH-011 decision. BENCH-009
derived its factorization seed from the exact axis object
`{target_id, seed, parent_horizon, selection_scope, heldout_fold}`. V1 instead hashed
`{target_id, family, seed, parent_horizon, heldout_fold}`. Equal retained ranks do not repair the
changed subspace: an independent reconstruction found a minimum principal cosine of `0.9997002`
and projector Frobenius-squared difference `0.01056` on one unit.

The v1 executed bytes remain preserved under the canonical result as `executed_sources_v1.tar`,
SHA-256
`eaaba6d462321199fcd98d12e73ec60cde756a9b5298ab7cb69f1ddda17ac7d6`.
V2 must reconstruct and persist the exact BENCH-009 unit ID, rerun all `96` rows, and apply the
unchanged gates. Until then BENCH-011 is unavailable, not negative.

## Corrected v2 result and closure (2026-07-16)

V2 changed only the factorization-seed binding plus provenance/validator hardening. It binds every
unit to BENCH-009's exact `{target_id, seed, parent_horizon, selection_scope, heldout_fold}` hash,
validates that hash against BENCH-009's terminal completed-unit ledger, and reproduces all `24`
base ranks. The four shards contain `5/5/7/7` units and `20/20/28/28` rows; their canonical union
has `24` units, `96` unique rows, and zero errors.

Independent audits replayed every held-out linear prediction and native render bit-exactly. All
nested deltas are non-negative, all extension ranks are six, and maximum orthogonality error is
`2.4425e-15` against a minimum permitted tolerance of `8.7311e-11`. The frozen v2 decision is:

| Candidate | Radius | Eligible / 24 | Spearman | Median realized/predicted | Negative realized / 24 | Gate |
|---|---:|---:|---:|---:|---:|---|
| affine | `0.25` | `16` | `0.4000` | `0.20711` | `14` | fail |
| affine | `0.75` | `14` | `-0.30549` | `-21.54402` | `24` | fail |
| carrier | `0.25` | `16` | `0.38529` | `-0.17226` | `16` | fail |
| carrier | `0.75` | `14` | `-0.20440` | `-25.23032` | `24` | fail |

The nested statistic repairs BENCH-009's non-negativity defect, but capacity in that local image
subspace does not calibrate to discovery-fitted native held-out gain. Per the frozen rule, close
this current-identity local-linear formulation. Do not change its identity bank, `rcond`, damping,
radii, eligibility floor, or gates, and do not consume disjoint data for it.

Canonical v2 hashes are binding `3c16c92e1de8c2bc005533f1a59465f8cacf77d5c7255be52cc5abe1c3aa5f61`,
rows `2ae62fbb220542c2d39365758e0ac0524323f50c61eee885ee4a1cc0a579e2dd`, and analysis
`a8363ecd6bced14214dfcdf2641c0e4bdf616fdcd5321ab3591aa7e1451b497a`. The executed v2 source
archive has SHA-256 `65742c2adfc953cce8a3cab40c77864864282920b489de70db878eac1746c2e1`.
