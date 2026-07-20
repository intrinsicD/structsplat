# FIT-020: Ranked deduplication response spectroscopy

## Status

Implemented and screened negative on 2026-07-15. A pre-run adversarial audit made integrity
enforcement fail-closed and bound the generated suite by hash; it changed no target, split,
intervention, feature, model, threshold, or horizon. The signal guard passed, but the response-
prediction and screening-utility claims failed decisively. Close this one-bend response-model
lineage without retuning it on the exposed targets. No production allocator, default change,
natural-image evaluation, optimizer-state claim, compression claim, or novelty claim is
authorized.

## Context and prior-art boundary

FIT-019 proved that raw row allocation does not commute with an exact opacity refinement, but its
invariant quotient allocator failed the recovery-utility guard. Its quotient-minus-raw response
changed sign from step 20 to step 100 in 7/16 cells and was strongly target-dependent. The next
question is therefore whether the *shape* of an early perturb--recover trajectory predicts late
utility, rather than whether another endpoint rule wins.

Early learning-curve prediction, function-preserving growth, gradient-based growth, curvature-based
splitting, and recovery-aware Gaussian management are established. In particular, Recovery-Aware
Pruning uses opacity recovery after a reset to manage 3D Gaussians; GradMax uses gradient signal to
initialize new units; Splitting Steepest Descent and SteepGS use local splitting curvature; Network
Morphism and staged training study function-preserving growth; and learning-curve extrapolation and
freeze--thaw methods predict late performance from prefixes. FIT-020 does not claim those ideas.
Its search-bounded remainder is narrower: predict the held-out late utility of a controlled,
budget-matched structural birth allocation from a paired response shape beyond static and
early-endpoint baselines.

Primary references:

- Deng et al., *Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering*,
  CVPR Findings 2026.
- Wang et al., *Steepest Descent Density Control for Compact 3D Gaussian Splatting*, CVPR 2025.
- Wu, Wang, and Liu, *Splitting Steepest Descent for Growing Neural Architectures*, NeurIPS 2019.
- Evci et al., *GradMax: Growing Neural Networks using Gradient Information*, ICLR 2022.
- Wei et al., *Network Morphism*, ICML 2016.
- Shen et al., *Staged Training for Transformer Language Models*, ICML 2022.
- Domhan, Springenberg, and Hutter, *Speeding up Automatic Hyperparameter Optimization by
  Extrapolation of Learning Curves*, IJCAI 2015.
- Rakotoarison et al., *In-Context Freeze--Thaw Bayesian Optimization*, ICML 2024.

## Question and claim

**Central claim:** after controlling target family, allocation strength, checkpoint quality,
residual concentration, immediate perturbation, and the step-10 endpoint, one preregistered early
response-bend descriptor predicts the paired step-200 utility of concentrated versus dispersed
birth allocation on held-out procedural target variants.

**Null:** the response bend does not improve held-out prediction or action selection over those
controls.

The intervention is called a **ranked deduplication path**, not pure coverage. Site rank, repeated
lineage depth, and distinct-site count move together, so the experiment identifies only this
specific one-ticket replacement treatment.

## Frozen intervention

For each target/seed, build a `quadtree_wse` field with explicit opacity at N=32 and fit it for 40
deterministic CPU steps with the normalized renderer. Rank canonical rows once by stable alpha-1
responsibility score and name the top eight `g1..g8`. Apply exactly eight sequential
moment-preserving births to the same detached checkpoint:

| Arm | Ordered canonical-group tickets |
|---|---|
| C5 | `g1,g2,g3,g4,g5,g3,g2,g1` |
| C6 | `g1,g2,g3,g4,g5,g6,g2,g1` |
| C7 | `g1,g2,g3,g4,g5,g6,g7,g1` |
| C8 | `g1,g2,g3,g4,g5,g6,g7,g8` |

Each adjacent arm changes one ticket at the same sequence position, replacing a repeated
higher-ranked site with the next absent ranked site. Every arm has eight births, N=40, the same
primitive, target, checkpoint, total recovery steps, and fresh Adam state. Repeated tickets split
the highest-opacity descendant with stable row-index ties, exactly as in FIT-019.

Run one 200-step recovery per arm with `log_every=1`, `lr_schedule=none`, terminal checkpointing,
and no pruning, growth, relocation, color solve, curriculum, support-fade schedule, adaptive count,
or early stop. History entry `t` is the pre-update render after exactly `t` updates for `t=0..199`;
append the terminal render as step 200. Extract steps `0,1,2,5,10,20,40,60,100,200` and integrate
AUC only after appending the terminal point. This is a fresh-optimizer response assay; the pre-fit
Adam state is unavailable and no production continuation claim is allowed.

## Frozen procedural data and split

Use 48x48 float32 RGB targets, seeds 0/1/2 as repeated checkpoint measures, and six families with
six fixed variants. Variant indices `{0,2,3,5}` are model-training targets and `{1,4}` are held
out. Target hashes, parameters, and split labels must be written before the first fit. No protected
natural image or FIT-019 target hash may occur.

The ordered `target_id:pixel_sha256` manifest has frozen combined SHA-256
`4e51518b9c2b1816311f1b2856c19ea77acaea913b534987fa776cf160740ebe`.

Parameter tuples are interpreted by the deterministic generator named in parentheses:

- `sinusoid(cycles, angle_deg, phase)`:
  `(2.75,12,.25)`, `(3.80,38,1.10)`, `(4.90,71,2.05)`, `(6.10,109,2.75)`,
  `(5.55,143,.70)`, `(7.25,166,1.85)`.
- `chirp(base_cycles, sweep, angle_deg, phase)`:
  `(1.40,3.20,8,.30)`, `(2.10,4.60,34,1.20)`, `(1.80,6.10,67,2.10)`,
  `(2.80,3.80,101,2.85)`, `(2.45,5.35,137,.75)`, `(3.20,6.70,169,1.65)`.
- `oriented_ramp(angle_deg, slope, gamma, hue)`:
  `(5,.85,.70,.15)`, `(31,1.10,1.25,.80)`, `(63,1.35,1.70,1.45)`,
  `(97,.95,2.10,2.10)`, `(132,1.20,.90,2.75)`, `(164,1.45,1.40,3.40)`.
- `opposing_ramps(angle_deg, slope, gamma, hue)`:
  `(18,.90,.80,.25)`, `(45,1.15,1.30,.95)`, `(76,1.40,.65,1.55)`,
  `(112,1.05,1.80,2.20)`, `(146,1.30,1.10,2.85)`, `(173,1.50,1.55,3.50)`.
- `soft_ridge(angle_deg, width, offset, hue)`:
  `(10,.055,-.18,.10)`, `(42,.080,-.08,.70)`, `(74,.105,.02,1.30)`,
  `(108,.065,.12,1.90)`, `(139,.090,-.14,2.50)`, `(168,.120,.16,3.10)`.
- `lattice(angle_deg, freq_u, freq_v, softness, hue)`:
  `(7,2.5,3.5,1.2,.2)`, `(29,3.5,5.0,1.8,.8)`, `(58,4.5,6.0,2.4,1.4)`,
  `(93,5.5,3.0,3.0,2.0)`, `(127,6.5,4.5,2.0,2.6)`,
  `(161,7.0,6.5,2.8,3.2)`.

## Frozen measurements and models

Record artifact-level source/config/environment provenance and per-trajectory target, source,
canonical-field, ranking, and ordered-action hashes; family/variant/split/seed; pre-fit PSNR;
residual normalized entropy and top-decile mass;
image-space intervention RMS; exact count and finiteness; PSNR/loss at the frozen steps; terminal-
inclusive PSNR AUC; peak/crossover/late-asymptote descriptors; and action/recovery instrumentation
time. Peak, crossover, post-10, late-asymptote, and full-AUC values are descriptive only.

For C5/C6/C7, pair against the same target/seed C8 trajectory. Average all numeric predictors and
outcomes over the three seeds *before* model fitting, leaving one row per target variant x coverage.
The primary outcome is `y_k = PSNR_Ck(200) - PSNR_C8(200)`.

Fit three ridge models. Standardize on each training fold only. Select lambda independently per
model from the frozen grid `{1e-4,1e-3,1e-2,1e-1,1,10,100}` by leave-one-training-target-variant-
out CV, keeping all three coverages together; break ties toward the smaller lambda. Refit on all
training targets and evaluate held-out targets once.

1. **Intervention baseline:** family one-hot, coverage deficit, squared coverage deficit, pre-fit
   PSNR, residual entropy, residual top-decile mass, paired immediate PSNR delta `d0`, and
   intervention-render RMS.
2. **Early-level baseline:** all intervention features plus paired `d10` and C8's step-0-to-step-10
   gain.
3. **Response-shape model:** all early-level features plus only
   `bend = ((d10-d2)/8) - ((d2-d0)/2)`.

Report held-out RMSE, MAE, bias, sign accuracy, Spearman correlation, per-family RMSE, calibration,
and selection regret. The sign subset is frozen to `|y| >= 0.05 dB`; require at least 18/36
eligible held-out cells. The majority-sign comparator is learned from eligible training labels,
never held-out labels. Use a deterministic 2,000-replicate target-variant bootstrap stratified by
family for uncertainty only; it cannot change the decision.

For selection, set predicted C8 delta to zero, choose the predicted best C5..C8 (ties choose higher
coverage), and compare held-out late regret with the early-level model, choosing the observed best
step-10 arm, and always C8.

## Preregistered decisions

The **signal guard** passes only if held-out `SD(y) >= 0.10 dB` and at least 9/36 cells have
`|y| >= 0.10 dB`. If it fails, the ranked deduplication treatment is too weak in this regime and no
descriptor conclusion is permitted.

Conditional on signal and integrity, the **response-prediction claim** passes only if:

1. response-shape held-out RMSE is at most `0.85x` early-level RMSE;
2. response sign accuracy is at least 70% and at least 10 percentage points above both early-level
   accuracy and the frozen training-majority comparator on at least 18 eligible cells;
3. response RMSE is lower in at least 4/6 families; and
4. absolute response mean prediction bias is at most `0.10 dB`.

The **screening-utility guard** passes only if the prediction claim passes and the response
selector's mean held-out regret is strictly lower than the early-level selector, observed-step-10
selector, and always-C8 policy, while its mean selected late delta over C8 is positive. Lower RMSE
without lower selection regret does not authorize a method follow-up.

Failure means stop without retuning targets, splits, horizons, features, deadbands, lambda grid, or
thresholds. A full pass authorizes only a new task that tests the frozen descriptor across disjoint
intervention types and, separately, an optimizer-state-preserving implementation. It does not
authorize production use.

## Result and decision

The primary artifact contains all 432 frozen trajectories and 108 seed-averaged paired rows. All
integrity checks and the signal guard passed: held-out `SD(y)=3.2529 dB`, and 35/36 held-out cells
had `|y| >= 0.10 dB`. The experiment therefore had ample treatment variation to test the central
claim.

The response bend added no predictive value. Held-out RMSE was `2.9616 dB` for the early-level
baseline and `2.9641 dB` for the response model, a ratio of `1.00083` with stratified bootstrap
95% interval `[0.99858, 1.00189]`, rather than the required ratio at most `0.85`. Response sign
accuracy was 25/36 (`69.44%`), identical to the early model and training-majority comparator;
response RMSE improved in only 2/6 families; and mean bias was `-1.0455 dB` with bootstrap interval
`[-1.9079, -0.2196]`. Grouped training CV agreed: response RMSE `1.6629` versus early `1.6615`.

The bend also changed no held-out action. Both early and response selectors chose C5 for all 12
held-out targets and had mean late regret `1.1116 dB`; the observed-step-10 comparator had lower
regret `0.7669 dB`, while always C8 had `2.9873 dB`. The response-prediction and screening-utility
gates therefore fail and the frozen decision is **stop**.

C5/C6/C7 had descriptive step-200 means above C8, and C6 was the train-best fixed arm, but target
identity dominated: the largest gains came from sinusoid and chirp families, signs reversed across
horizons, and the held-out set was exposed before that simple-policy observation was made. C6 and
step 10 remain exploratory observations, not promoted methods or pass-authorized rescues.

Artifacts:

- primary: `results/fit020_response_spectroscopy_v1_2026-07-15/`;
- measurement-equivalent replay after the paired-CSV union-schema repair:
  `results/fit020_response_spectroscopy_v1_replay_2026-07-15/`;
- target-manifest SHA-256:
  `4e51518b9c2b1816311f1b2856c19ea77acaea913b534987fa776cf160740ebe`;
- primary/replay source combined SHA-256:
  `e1d64469d858c65d74aa6083f2845a1a421946eab646ede5bd0be6857ba9b383` /
  `75fbe6ac96af2d3b250e81c6157a0645402ebe2ed70e5fa29dd5ca2cff3b99ab`;
- primary aggregate-with-paired-rows SHA-256:
  `b8848767ae746c19576bbbc546b66f5f809e318a745f0cf164b812c21687b2bf`.

The primary computation completed before its CSV writer discovered that the first train row did
not contain held-out-only prediction columns. Finalization reloaded the immutable 432-row JSON,
used the primary source snapshot for frozen aggregation, proved the aggregate unchanged, and
changed only the paired-CSV header to the sorted union of row fields. The replay therefore differs
from primary source only in that serializer repair and its regression test. Across all 432 cells,
every compared non-timing measurement field matches after excluding timing and source-provenance
fields; all paired rows, the normalized aggregate, target manifest, and STOP decision also match
exactly. It is not described as a byte-identical-source replay.

## Acceptance criteria

- [x] Frozen target suite, action chain, trajectory extraction, terminal-inclusive AUC, aggregation,
      ridge/CV, gates, and malformed-table behavior have focused tests.
- [x] Primary and measurement-equivalent post-writer-fix replay agree exactly on compared non-
      timing measurement fields; timing/provenance exclusions, two expected source differences,
      and the output-only correction are disclosed above.
- [x] Results are reported whether positive, partial, or negative, without post-hoc rescue.
- [x] Task/index, benchmark docs, research report, and decision ADR agree.
- [x] Full tests, Ruff, source-snapshot verification, and diff hygiene pass.

## Interfaces allowed

New benchmark/test/task/research evidence only. FIT-020 does not authorize changes to production
`GaussianField`, fitter, renderer, config, CLI, codec, or defaults.

## Depends on

FIT-007, FIT-009, FIT-018, FIT-019, BENCH-002, E2/D3 in the 2026-07-15 research portfolio.
