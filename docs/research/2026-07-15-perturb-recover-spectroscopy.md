# FIT-020 ranked deduplication perturb--recover assay

**Date:** 2026-07-15
**Decision:** stop the frozen response-bend and branch-selection lineage
**Task:** [`FIT-020`](../../tasks/FIT-020-perturb-recover-spectroscopy.md)
**Benchmark:** [`benchmarks/perturb_recover_spectroscopy.py`](../../benchmarks/perturb_recover_spectroscopy.py)
**Primary artifact:** `results/fit020_response_spectroscopy_v1_2026-07-15/`
**Replay artifact:** `results/fit020_response_spectroscopy_v1_replay_2026-07-15/`

## Executive result

FIT-020 found a large, non-inert late treatment signal within the frozen assay but no evidence that
the preregistered early response bend predicts it. The response model was slightly worse than the
early-level baseline on held-out target variants: `2.9641` versus `2.9616 dB` RMSE. Its sign
accuracy was the same
`69.44%` as both the early model and the training-majority comparator, it improved family RMSE in
only 2/6 families, and it retained `-1.0455 dB` mean bias. It also chose exactly the same C5 arm as
the early model on every held-out target and had worse regret than selecting by observed step 10.

This is a decisive negative result for the frozen one-bend operationalization. It is not evidence
that every learning-curve model, recovery diagnostic, or recovery-aware Gaussian method fails.
The result closes tuning of this bend, horizon, feature set, ridge grid, and target suite. No
production code changed.

## Question and identification boundary

The assay asks whether the shape of a short fresh-Adam response predicts the late utility of a
controlled structural allocation. Starting from the same fitted N=32 checkpoint, each arm applies
eight moment-preserving births and ends at N=40. C5 through C8 replace repeated high-ranked tickets
with progressively more distinct ranked sites. Site rank, ticket multiplicity, lineage depth, and
distinct-site coverage therefore co-vary.

The identified treatment is a **ranked deduplication path**, not pure coverage. All conclusions are
restricted to that path, the normalized renderer, procedural 48x48 targets, fresh Adam recovery,
and 200 updates. Because every arm has the same atom class, N, and nominal float payload, this
experiment cannot test compression or expressiveness. Its instrumented CPU timings are benchmark
costs, not a renderer or training-speed comparison.

## Mechanistic prior-art boundary

The relevant state of the art already contains the broad ingredients:

- [Recovery-Aware Pruning](https://openaccess.thecvf.com/content/CVPR2026F/html/Deng_Improving_Densification_in_3D_Gaussian_Splatting_for_High-Fidelity_Rendering_CVPRF_2026_paper.html)
  uses per-Gaussian opacity recovery after an opacity reset as a pruning signal. Edge-aware
  selection, long-axis splitting, and growth control are separate components of the same pipeline.
  FIT-020 neither reproduces nor refutes that coupled method.
- [SteepGS](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.html)
  uses a local constrained second-order formulation to derive a two-offspring rule, opposite
  minimum-eigenvector displacements, and half-magnitude opacity normalization.
  [Splitting Steepest Descent](https://proceedings.neurips.cc/paper_files/paper/2019/hash/3a01fc0853ebeba94fde4d1cc6fb842a-Abstract.html)
  supplies the earlier second-order functional-growth foundation. FIT-020 uses neither gradient nor
  curvature controls, so it cannot compare against those mechanisms.
- [GradMax](https://openreview.net/forum?id=qjN4h_wwUO) initializes new units by maximizing
  new-weight gradient signal. That is a different growth-initialization mechanism from FIT-020's
  frozen responsibility-ranked birth path.
- [Network Morphism](https://proceedings.mlr.press/v48/wei16.html) studies exactly
  function-preserving growth. [Staged Training](https://proceedings.mlr.press/v162/shen22f.html)
  transforms the whole training state and explicitly targets preservation of loss and training
  dynamics. FIT-020 restarts Adam, so it makes no optimizer-state-continuation claim.
- Prefix-based performance prediction is a mature field, from
  [parametric learning-curve extrapolation](https://www.ijcai.org/Proceedings/15/Papers/487.pdf)
  to modern [freeze--thaw prediction](https://proceedings.mlr.press/v235/rakotoarison24a.html).
  The negative result applies only to one preregistered bend added to preregistered static and
  step-10 controls.
- [Dynamic Mode Decomposition with control](https://epubs.siam.org/doi/10.1137/15M1013857) is a
  formal state/input identification framework. FIT-020 does not estimate a dynamical system and is
  therefore described as a perturb--recover **assay**, not technical system identification.

The search-bounded contribution is consequently an evidence program: a matched, seed-aggregated,
held-out killing test for one early-response descriptor under a specific discrete birth path. No
global novelty claim is made.

## Frozen experiment

The generated target suite has six procedural families with six fixed variants each. Variants
`{0,2,3,5}` train the models; `{1,4}` are evaluated once as held out. Seeds `{0,1,2}` are repeated
checkpoint measurements and are averaged before modeling. The ordered target manifest is bound by
SHA-256 `4e51518b9c2b1816311f1b2856c19ea77acaea913b534987fa776cf160740ebe`.

For each target and seed:

1. initialize N=32 with `quadtree_wse` and explicit opacity;
2. prefit for 40 deterministic CPU updates;
3. rank canonical rows once by stable alpha-1 responsibility;
4. apply one of C5, C6, C7, or C8's eight ordered births;
5. recover once for 200 fresh-Adam updates while recording the full 0--200 curve; and
6. extract steps `0,1,2,5,10,20,40,60,100,200`, including the terminal state in AUC.

For every C5/C6/C7 trajectory, C8 on the same target and seed is the paired reference. The primary
label is `y = PSNR_Ck(200) - PSNR_C8(200)`. Three ridge models use grouped leave-one-training-target-
variant-out CV and fold-local standardization:

| Model | Additional information |
|---|---|
| Intervention | Family, allocation strength, prefit quality, residual concentration, immediate paired perturbation, render RMS |
| Early | Intervention features plus paired step-10 delta and C8's 0--10 gain |
| Response | Early features plus only `((d10-d2)/8) - ((d2-d0)/2)` |

The response claim required at least a 15% held-out RMSE reduction, at least 70% sign accuracy and
a 10-point gain over both comparators, wins in 4/6 families, and absolute bias at most `0.10 dB`.
Screening additionally required lower selection regret than every frozen comparator.

## Integrity and signal

The pre-run adversarial audit changed no scientific choice. It made missing histories, malformed
counts, non-finite values, wrong action chains, ranking/hash mismatches, incomplete curves, missing
terminal AUC, source drift, and target reuse fail closed. Focused tests cover the target manifest,
action chain, trajectory semantics, seed aggregation, grouped CV, held-out isolation, gates, and
malformed tables.

All integrity checks passed:

| Check | Result |
|---|---:|
| Raw trajectories | 432/432 |
| Seed-averaged pairs | 108/108 |
| Train / held-out model rows | 72 / 36 |
| Complete histories, finite renders, N=40 | all pass |
| Frozen action, ranking, target, and source hashes | all pass |

The signal guard also passed strongly: held-out `SD(y)=3.252924 dB`, and 35/36 cells had
`|y| >= 0.10 dB`. Failure cannot be attributed to an inert intervention.

## Central held-out result

| Metric | Intervention | Early | Response | Frozen response requirement |
|---|---:|---:|---:|---:|
| RMSE (dB) | 2.958811 | 2.961624 | 2.964081 | response / early <= 0.85 |
| MAE (dB) | 2.269034 | 2.266802 | 2.269235 | descriptive |
| Bias, mean prediction minus observation (dB) | -1.041212 | -1.051579 | -1.045524 | abs(response) <= 0.10 |
| Sign accuracy | 69.44% | 69.44% | 69.44% | >=70% and +10 points |
| Spearman | 0.6018 | 0.6036 | 0.5951 | descriptive |
| Families with response RMSE lower than early | -- | -- | 2/6 | >=4/6 |

The response/early RMSE ratio is `1.000830`; its family-stratified 2,000-bootstrap 95% interval is
`[0.998582, 1.001890]`. Response bias has interval `[-1.907854, -0.219634]`. Grouped train-only CV
also preferred the simpler model: response `1.662894` versus early `1.661532`, with lambda 100
selected for all three models. The train-only CV likewise shows no response-model advantage and
provides no rationale for a rescue fit.

## Selection and convergence

| Selector | Mean held-out late regret (dB) | Mean selected late delta over C8 (dB) |
|---|---:|---:|
| Early model | 1.111568 | +1.875776 |
| Response model | 1.111568 | +1.875776 |
| Observed best at step 10 | 0.766896 | +2.220448 |
| Always C8 | 2.987344 | 0.000000 |

The early and response models selected C5 on all 12 held-out targets, so the bend changed no
decision. The observed best arm at step 10 still differed from the step-200 oracle on 9/12 targets.
Seed-averaged paired signs changed between step 10 and step 200 in 11/36 held-out cells and in 4/6
family means. Thus early/late instability was observed within this assay, but the frozen bend does
not explain it.

Held-out bend versus late utility had Pearson `-0.196` and Spearman `0.039`; step-10 delta versus
late utility had Pearson `0.106` and Spearman `0.117`. These are descriptive diagnostics, not new
models.

## Descriptive arm signal, not a promotion

| Arm | Step-0 PSNR | Step-10 PSNR | Step-200 PSNR | Terminal-inclusive AUC |
|---|---:|---:|---:|---:|
| C5 | 30.9398 | 31.2897 | 42.5331 | 39.9329 |
| C6 | 30.8522 | 31.1937 | 42.7164 | 39.8060 |
| C7 | 30.7188 | 31.1731 | 42.1797 | 39.6293 |
| C8 | 30.6490 | 31.1453 | 41.3119 | 38.9515 |

These arm means pool all 108 trajectories per arm across training and held-out variants.

On held-out targets, mean step-200 deltas over C8 were `+1.8758`, `+2.2487`, and `+1.6424 dB` for
C5, C6, and C7. C6 was also the best fixed arm on training targets and would have had `0.7387 dB`
held-out regret. This is scientifically interesting but not confirmatory: the preregistration did
not contain a fixed-C6 policy claim, and held-out outcomes were exposed before this observation was
made.

The effect is highly heterogeneous. C6's held-out family means range from `+7.8785 dB` on
sinusoids and `+3.8957 dB` on chirps to `-0.9396 dB` on opposing ramps and `-0.4071 dB` on soft
ridges. Across concentrated arms, excluding sinusoid and chirp reduces the mean held-out gain from
`+1.9223` to `+0.1858 dB`. Post-hoc, in-sample OLS on the 36 held-out paired rows had R² of about
0.71 for family only, 0.01 for coverage only, and 0.92 for target identity. These values are
descriptive rather than transferable variance explanations. The simple-arm signal therefore needs
new target breadth and a separately frozen hypothesis before it could be evaluated.

## Requested-axis verdict

| Axis | Evidence | Decision |
|---|---|---|
| Quality | Concentrated ranked paths can outperform C8, but effects are family-dominated and the confirmatory predictor fails. | No promoted quality method; C6 is exploratory only. |
| Convergence | Horizon rankings and signs reverse often; the bend does not predict late utility or improve selection. | Close the bend lineage; do not tune another horizon on these targets. |
| Performance | Dense CPU instrumentation differs by only a few percent and changes across replay. | No speed claim; evaluate kernel/backward work in a separate benchmark. |
| Compression | Every arm has N=40 and the same nominal payload; no stream was encoded. | No compression evidence. COMP-006 subsequently measured marginal cold-stream bytes and rejected its frozen standard-birth claim. |
| Expressiveness | Atom family, count, and trainable schema are fixed. | No expressiveness evidence. Use an equal-byte atom suite for that question. |

## Reproducibility and writer correction

The primary completed all 432 trajectories, then its paired-CSV writer failed because the first
training row did not contain prediction fields that occur only on held-out rows. `rows.json` and
the journal were already complete. Finalization loaded those immutable rows, aggregated with the
primary source snapshot, asserted exact equality with the repaired aggregator, and changed only
the CSV header to the sorted union of row fields. `finalization.json` records the correction and
the aggregate-with-paired-rows SHA-256
`b8848767ae746c19576bbbc546b66f5f809e318a745f0cf164b812c21687b2bf`.

The replay was then run from the repaired writer. It is measurement-equivalent, not byte-identical
source: the benchmark serializer and its regression test are the only two expected source-hash
differences. Across 432 keyed cells there are zero mismatches among the compared non-timing
measurement fields after timing and source provenance are excluded; paired rows, the normalized
aggregate, target manifest, and STOP decision are exact. Primary and replay row SHA-256 values are
`9e55ce6df746f44ffe9a0c437e700340b7f7ad7882f22cde2e0721a3335f12f6` and
`b33a4be5e8c35b47e987e8c4dbac2abb5b56746277c814d171117a175c24ac1b` because timing and source
fields are intentionally retained in the raw rows.

Independent numerical review reproduced pairing, fold-local standardization, grouped CV,
held-out predictions, gates, selection, manifests, source snapshots, finalization hashes, and the
primary/replay comparison. No blocking discrepancy was found. The descriptive
`first_crossover_step` uses 201 as a no-crossing sentinel, so its seed average need not be an
actual checkpoint; it is unused by every decision and should not be interpreted as event time.

## Decision and subsequent experiment

Do not add a response-bend feature, learned branch selector, ranked-deduplication default, or
lineage state to production. Do not rescue the result by changing bend definitions, checkpoints,
regularization, targets, or the already-exposed held-out split. A future recovery study would need
a materially different mechanism, disjoint targets, cross-intervention validation, and preserved
optimizer-state controls; FIT-020 does not authorize it.

E4 was the clean next axis and was executed as COMP-006. It encoded standard births, matched
count-neutral replacements, and an exhaustive precision envelope as complete SSPL1 streams on new
development targets. Exact bytes changed fine-grained control selection, but birth lost
`-1.0714 dB` to the strongest control with all family means negative. See
[`2026-07-15-marginal-cold-stream-rd.md`](2026-07-15-marginal-cold-stream-rd.md). This subsequent
result does not alter FIT-020's frozen decision; it closes the proposed standard-birth follow-up.
