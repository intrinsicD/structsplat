# ADR-0015: Keep the response-bend selector benchmark-only

## Context

FIT-020 tested whether one preregistered early trajectory bend predicts late utility along a
ranked deduplication path. Four equal-count arms applied eight moment-preserving births to the same
N=32 checkpoint, restarted Adam, and recovered for 200 steps. Models were trained on four variants
per procedural family and evaluated once on two held-out variants.

The treatment signal was strong (`SD(y)=3.2529 dB`; 35/36 held-out cells exceeded `0.10 dB` in
absolute value), but the response feature failed every material prediction gate. Response RMSE was
`2.9641 dB` versus `2.9616 dB` for the early baseline, sign accuracy was the same `69.44%`, response
improved only 2/6 family RMSEs, and bias, defined as mean prediction minus observation, was
`-1.0455 dB`. Both models chose C5 on every held-out target and had `1.1116 dB` late regret, worse
than the observed-step-10 comparator's `0.7669 dB`. A measurement-equivalent post-writer-fix replay
reproduced every compared non-timing measurement field under the normalized comparison, plus every
paired row, the normalized aggregate decision, and the target manifest.

Concentrated fixed arms have positive descriptive mean quality and AUC versus C8, with C6 strongest
at step 200, but those gains are dominated by sinusoid and chirp families. A fixed-C6 policy was
not preregistered, and the held-out targets were already exposed when that observation was made.

## Decision

Do not add the FIT-020 response bend, its ridge predictor, or its branch selector to production.
Close this exact model/feature/horizon/target lineage without post-hoc tuning. Keep the assay and
its fail-closed integrity machinery as benchmark infrastructure only.

## Consequences

+ `GaussianField`, the fitter, renderer, CLI, config, codec, and defaults remain unchanged.
+ The dense trajectory logger, target-grouped model evaluation, source snapshots, and malformed-
  table gates remain reusable for genuinely new, disjoint hypotheses.
+ C6 and observed-step-10 selection are exploratory observations, not promoted policies and not
  pass-authorized rescue experiments.
+ Any future recovery predictor must use a materially different mechanism, disjoint targets,
  cross-intervention validation, and explicit optimizer-state-preserving controls.
+ The next recommended research axis was marginal cold-stream rate--distortion attribution, which
  tests exact bytes rather than treating Gaussian count or structural regularity as compression;
  COMP-006 has since completed that negative screen, recorded in ADR-0016.
- FIT-020 supplies no natural-image, optimizer-continuation, speed, compression, expressiveness,
  or broad learning-curve-prediction claim.

## Links

Follows ADR-0014 and depends on FIT-019/FIT-020. It does not supersede a shipped architecture
decision because no response predictor or selector entered production.
