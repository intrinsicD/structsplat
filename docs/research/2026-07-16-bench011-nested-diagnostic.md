# BENCH-011 nested residual-extension diagnostic

**Date:** 2026-07-16
**Status:** corrected v2 completed negative; close without retuning
**Evidence scope:** already-spent BENCH-009 development parents; no new-data or method claim

## Result

The minimal nested-subspace repair fixes BENCH-009's algebraic non-negativity problem, but it does
not make the selected local affine/carrier actions predict native held-out improvement.

Corrected v2 exactly reuses BENCH-009's cross-fit unit IDs and base-factorization seeds. It
completed `24` units and `96` native rows with no errors. Two independent audits replayed every
held-out prediction and native render bit-exactly and recovered the exact BENCH-009 base ranks and
one independently reconstructed basis hash.

| Candidate | Radius | Eligible / 24 | Spearman | Median realized/predicted | Negative native / 24 | Decision |
|---|---:|---:|---:|---:|---:|---|
| affine | `0.25` | `16` | `0.4000` | `0.20711` | `14` | fail |
| affine | `0.75` | `14` | `-0.30549` | `-21.54402` | `24` | fail |
| carrier | `0.25` | `16` | `0.38529` | `-0.17226` | `16` | fail |
| carrier | `0.75` | `14` | `-0.20440` | `-25.23032` | `24` | fail |

All four preregistered strata require Spearman `>=0.8` and median ratio in `[0.5,2.0]`. None is
close. The valid decision is `close_without_retuning`: do not change the identities, bank,
`rcond`, damping, radii, eligibility floor, or gates, and do not consume disjoint data for this
formulation.

## What was repaired

For one immutable base basis `Q_J`, v2 computes in float64

```text
r_perp = (I - Q_J Q_J^T)^2 r
B      = (I - Q_J Q_J^T)^2 A
Q_B    = left singular vectors of B with s_i > 1e-5 sigma_max(A)
Delta_nested = ||Q_B^T r_perp||^2.
```

There is no extension-column normalization and no joint rethresholding. All deterministic controls
pass; all `96` extension ranks are six; every `Delta_nested` is non-negative; maximum basis
orthogonality error is `2.4425e-15` versus minimum tolerance `8.7311e-11`.

The physical solve uses the frozen compact block

```text
[[Q_J^T J, Q_J^T A],
 [        0, Q_B^T A]]
```

with raw-chart damping `1e-3` and one uniform joint L-infinity scale. Coefficients are fitted on
discovery rows only. Held-out prediction uses the frozen physical coefficients, and native dynamic
support is rendered once.

## V1 invalidation and v2 correction

V1 derived its randomized factorization seed from
`{target_id, family, seed, parent_horizon, heldout_fold}`. BENCH-009 had used
`{target_id, seed, parent_horizon, selection_scope, heldout_fold}`. All `24` seeds differed. Equal
ranks did not restore the comparator: on one independently reconstructed unit, minimum principal
cosine was `0.9997002` and projector Frobenius-squared distance was `0.01056`.

V1 is therefore unavailable, not negative. Its complete source bytes are retained under its
canonical result, but none of its gate values support the decision.

V2 corrects only that seed binding and adds fail-closed provenance/coverage hardening:

- the BENCH-009 completed-unit ledger is an exact hashed input;
- each unit persists and validates its exact BENCH-009 ID;
- canonical analysis requires the exact six-family/two-seed/two-fold/four-cell grid;
- Python, Torch, NumPy, CPU, git commit/branch/dirty digest, sources, and inputs are bound; and
- canonical JSONL merge is overwrite-idempotent rather than append-based.

## Mechanistic postmortem

A descriptive frozen-support decomposition was run after the decision without refitting or gate
reinterpretation. At radius `0.25`, the mean linear prediction is slightly positive
(`~1.5--1.7e-6` MSE), but finite frozen-support curvature makes it negative; dynamic-support change
adds a smaller additional loss. At radius `0.75`, finite curvature dominates by roughly
`2.0--2.7e-4` MSE, while the extra dynamic-support gap is roughly `0.7--1.2e-5`.

This says the failure is not repaired by merely making the projector nested. Dense physical base
steps leave the useful local linear regime before support discontinuities become the dominant
problem. It does not prove that a particular trust-region optimizer would win; such an optimizer
would be a different experiment.

## Exact artifacts

- Binding: `3c16c92e1de8c2bc005533f1a59465f8cacf77d5c7255be52cc5abe1c3aa5f61`
- Config: `3c2c87b00acb7c259ed6ce6feb6dddfbc58124a28788583eb054228cb5028b8e`
- Rows: `2ae62fbb220542c2d39365758e0ac0524323f50c61eee885ee4a1cc0a579e2dd`
- Completed units: `0e519e0235c746ce3da726b841477ff477efed5ad73e843d2c7a406cf4ef8ba4`
- Analysis: `a8363ecd6bced14214dfcdf2641c0e4bdf616fdcd5321ab3591aa7e1451b497a`
- Algebra controls: `84b9fb6b3756e85ea0ea9defb13e4fafdaf1ec61069153598a85a7a925e582de`
- Executed-source archive:
  `65742c2adfc953cce8a3cab40c77864864282920b489de70db878eac1746c2e1`

Canonical analysis is scientifically order-invariant, but replaying from row-ID-sorted canonical
rows changes excluded-ID ordering and two pooled means by one floating-point ulp relative to the
original shard insertion order. All gate statistics and the decision are unchanged. Future
analyzers sort inputs before reduction.

## Claim boundary

| Axis | Conclusion |
|---|---|
| Quality | No improvement claim; this is spent development data and every calibration stratum fails. |
| Convergence | Not tested; there is no recovery phase. |
| Performance | Not tested; CPU diagnostic cost is not an implementation benchmark. |
| Compression | Not tested; equal coefficient count is not actual stream rate. |
| Expressiveness | Unavailable; coherent local capacity does not realize as calibrated native gain. |
| Production | No grammar, optimizer, renderer, codec, or default change authorized. |
