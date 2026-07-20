# BENCH-015 decoder-synchronized affine lift

## Decision

**KILL the frozen global decoder-synchronized lift and close its exact robust-IRLS,
four-neighbor log-mean-exp, and smootherstep-confidence lineage.** The method achieved large
same-stream static quality gains with no transmitted tail, but failed the registered no-harm,
color-convergence, and cold-decode performance gates. Stage 1 and the prewritten local-confidence
successor are not authorized.

Canonical artifact:
`results/bench015_decoder_synchronized_lift_stage0_v1_2026-07-16`.

## Independent integrity audit

- Task: `215bdc585944d2be17edf1d737fe958ff04e06e36980b375d26a4347dcde83fa`
- Binding: `be733865b77570d2cfb8244bde6805aee3020d6b92eaa1b406ae2ca3ed3b3c46`
- Analysis: `62f42f280efaa0e2d0857da55c3ea7f02b507a09ea3eaa95f1bf139c038a88be`
- Replay: `55d22739e29c28915ad253bfd1d65a4ef340f059c155beff0873203e143e571e`
- Artifact manifest:
  `01a4f19bd22d07a856f41e6ed9a4ecac747e9873e8adf3a4f817c91d8df4770c`
- Completion:
  `b1e34b0d6a2edbc7cc4a64c988eb66055d3954cb8a0b7bf7db46873d515a1d97`
- Executed-source archive:
  `11a578e3215ec0a0c641daa2e81bbf55491f0ce3a36e9a07486d8562feb3e97c`

The internal replay and a fresh external replay passed all `27` checks. The `677`-file artifact
contains exactly `60` fields, `162` streams and static rows, `486` permutations, `108`
trajectories with `6,588` logged losses/output hashes and `864` checkpoint arrays, `6` gradient
rows, and `162` timing rows. All terminal statuses are `ok`. Float32 shadow and float64 primary
decisions agree exactly.

## Axis results

| Axis | Result | Bound conclusion |
|---|---:|---|
| Plumbing/parity | pass | Constant and affine reproduction, stream identity, renderer parity, permutations, gradients, malformed-stream controls, and all replay inventories pass. |
| Smooth static quality | pass | Median DSL78/NW78 MSE ratios are `0.670143` for affine chirp, `0.109768` for soft crease, and `0.558233` for affine ring, with `6/6` wins for every family. |
| Boundary static quality | pass | Outer-ten ratios are `0.548929`, `0.082364`, and `0.244617`. |
| Complete rate | pass, narrow | DSL78 and NW78 use byte-identical inner streams and exactly equal complete lengths in all `54` pairs (`501--862` bytes). DSL78 beats the two-row NW80 control by median ratios near `0.98`; this is not compression relative to NW78. |
| No-harm | **fail** | One affine-occlusion edge cell is `1.011398` versus the `1.01` cap. Continuous crease exceeds its target-range guard in `5/6` cells, worst `0.029832` versus `2/255`. |
| Color convergence | **fail** | Smooth median AUC is `0.992516`, median final loss is `1.058438`, and worst final loss is `1.400686` relative to NW; `10/18` smooth cells exceed `1.05`. Hard worst is `1.090499`. |
| Prepared render | pass | Median DSL78/NW78 render ratio is `1.009728`. |
| Cold decode | **fail** | Median decode+derive+render ratio is `1.624762` versus the `1.50` cap. Median derivation alone is about `6.47 ms`; its operation ledger is exact. |
| Appearance rank | unchanged | The diagnostic base and lifted operators both have rank `78` in all `54` cells. No parameter or linear appearance dimension was added. |

The discontinuity detector is not softly discriminating on the canonical matrix. Every
affine-occlusion and continuous-crease channel receives `alpha=1`, while every isoluminant-step
channel receives `alpha=0`; only one quartic channel enters the `1.20--1.50` transition band.
Occlusion scores stay at `0.7016--1.1323`, continuous-crease scores at `0.0893--0.1132`, and
isoluminant-step scores at `1.5235--1.6489`. This explains why the detector completely suppresses
one easy step yet permits target-range excursions on a smooth crease and misses one occlusion edge
guard. The synthetic mid-transition derivative and all gradient gates pass, so this is a
spatial/statistical-scope failure, not a broken smootherstep derivative. It is evidence against
this frozen score/gate, not permission to tune its thresholds on the exposed matrix.

## What was learned

The useful positive is real but narrower than promotion: a decoder-derived trend-plus-residual
basis can rotate the same `N x 3` transmitted color coordinates toward smooth first-order image
structure. The static gains do not require extra bytes or dimensions. The negative is equally
mechanistic: recomputing a robust color-dependent basis during learning changes the optimization
geometry, its global confidence cannot localize mixed content, and eight robust updates dominate
cold decoding. Same parameter count therefore does not imply equivalent convergence or decode
cost.

The saved trajectories localize convergence further. All `18` smooth cells beat NW at step `20`
(median ratio `0.777526`), all `18` lose at step `35` (median `1.082952`), and only `8/18` win at
step `60`. A worst shared soft-crease cell keeps `alpha=(1,1,1)` throughout yet ends at
`1.162575`, so threshold switching is not the only cause; the changing robust reparameterization
itself creates oscillatory optimization geometry. OLS attribution reproduces essentially the same
no-harm failures, while Cauchy fitting reduces occlusion range excursion, so robust fitting remains
a useful analysis/initialization component even though it is not a viable decoder-time basis here.

This construction remains closely preceded by
[regression kriging](https://doi.org/10.1016/j.cageo.2007.05.001),
[backward-adaptive prediction](https://doi.org/10.1109/5.892714), and
[data-dependent moving least squares](https://arxiv.org/abs/2412.02304). BENCH-015 establishes a
StructSplat recipient/evidence result, not new robust-regression or approximation mathematics.

## Stop rule and claim boundary

Do not retune the robust loss, scale, iterations, graph, temperature, thresholds, formulas,
optimizer, or gates. Do not run the conditional local-confidence successor: its frozen
authorization required the smooth, convergence, and cost sections to pass, and the latter two
failed. A post-hoc non-rendering check using only the preregistered node score and persisted cold
states also finds that local confidence would attenuate none of the `1,404` continuous-crease
node-channel values, while weakening the isoluminant-step and quartic fallback. Any continuation
must pose a materially different mechanism and null on disjoint data.

This is a synthetic CPU Stage-0 result. It does not establish natural-image value, production
SSPL compression, GPU cost, general fitter behavior, broad robustness, added expressiveness,
publication novelty, or a default change.
