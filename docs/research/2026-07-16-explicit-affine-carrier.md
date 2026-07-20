# BENCH-014 explicit affine carrier

## Decision

**KILL the transmitted six-scalar affine-tail/range-ray representation, while retaining the
first-order signal as a decoder-synchronized research lead.** BENCH-014 passed synthetic quality,
stability, rank/expressiveness, gradients, permutations, and CPU render cost. It failed the frozen
complete-byte gate for every target and failed the terminal convergence guard.

Canonical artifact: `results/bench014_affine_carrier_stage0_v1_2026-07-16`.

## Independent integrity audit

- Task: `e2b7661c8a856cb2b560bd36e34c0d89f4962f06cf850d94f73afeb98623dbc7`
- Binding: `0a4febdbd9570bcb216a8f6ec1f3461af3577553401415900be4583e5f34825c`
- Analysis: `cb33e79204e7e1eca8be80fe8cb8591c968d59afeb0f7b5f3a1df9af86a0a6c3`
- Replay: `f93c571bb554eccd94fde7b347751d99316526a9d6bb753c5da6b6f992f4e024`
- Artifact manifest:
  `3a49fb7cab3071d95e2943337c21c28d4c2601b61fcce8853a98f544015c511a`
- Completion file:
  `525ecf9ecbc0192e14d65605f324d08024d52d079b8cc12d208e57ef91d4788a`
- Source archive:
  `1c0d9cceb7f87ce2db74592d4f6f8c4dbeb019d3217b36f96ea2f5867beee8e3`

All 1,179 sealed files passed independent hash/size verification. Exact ledgers contain 54 fields,
144 static rows and streams, 576 permutations, 96 trajectories, 9,696 checkpoints, 48 convergence
pairs, 288 timing rows, and one reached gradient cell. A fresh external replay passed all 19
checks.

## Axis results

| Axis | Result | Bound conclusion |
|---|---:|---|
| Quality | pass | Same-count median MSE ratios were 0.433251 (affine-sin), 0.668130 (affine-bump), and 0.404654 (saddle), with 6/6 wins each. |
| Boundary quality | pass | Outer-nine ratios were 0.211912, 0.201707, and 0.428136. |
| Expressiveness | pass | `[W,D]` gained exactly two spatial columns in all 48 cells, corresponding to six RGB coefficients. |
| Stability/gradient/permutation | pass | Positive residual weights, registered range guards, mixed-precision gradients, and row-order tolerances passed. |
| Prepared CPU render | pass | Median AC81/NW81 ratio was 1.005603; decode+render diagnostic was 1.006928. |
| Complete bytes | **fail** | Every target failed the AC81/NW83 gate; same-count AC81−NW81 was −17/12/196 bytes min/median/max. |
| Color convergence | **fail** | Median AUC ratio passed at 0.668853, but three shared-constant affine-bump terminal ratios failed; worst was 1.417939. |

The byte failure is mechanistically localized. It is not merely the nominal 12-byte binary16 tail:
turning ordinary colors into residual colors changes entropy. On discontinuities, residual-color
payloads added up to 169 bytes for the step and 184 bytes for the checker. Thus nominal scalar
pricing would have drawn the wrong compression conclusion.

## Research consequence

The result rejects explicit transmission, not first-order reproduction. The evidence-selected
successor is a decoder-synchronized lift using the ordinary cold-decoded colors:

```text
beta = robust_fit(X_mu, c)
r    = c - X_mu beta
y    = X beta + W r
```

This removes both measured delivery costs: it sends neither beta nor a residual-color stream. It
also adds no degrees of freedom, so it cannot inherit BENCH-014's `+6 DOF` claim. Its real risks are
signed effective color influence, range overshoot, a target-blind discontinuity decision, and
decoder/learning cost. BENCH-015 freezes disjoint mixed-content, steep-smooth, same-byte,
zero-start convergence, gradient, and timing tests for those risks.

The algebra is classical trend-plus-residual interpolation, closely threatened by
[regression kriging](https://doi.org/10.1016/j.cageo.2007.05.001),
[backward-adaptive prediction](https://doi.org/10.1109/5.892714), and recent
[data-dependent MLS](https://arxiv.org/abs/2412.02304). The defensible novelty boundary is a
possibly new StructSplat recipient/evidence relationship, not new mathematics.

## Claim boundary

BENCH-014 is a correlated synthetic Stage-0 matrix. It does not establish natural-image quality,
production SSPL rate, GPU performance, general fitter convergence, dataset generalization, or
publication novelty. Stage 1 is unauthorized because the conjunctive result is a kill.
