# FIT-042: Independent fine-detail pursuit confirmation

## Status

Todo. Preregister the source manifests, exact gates, and resource ceiling before opening any new
result-bearing target. This task may authorize a bounded masked-image opt-in recommendation, but
never a recipe/default change by itself.

## Context

FIT-040 is the strongest current candidate for deep fine detail. On the exposed masked Janelle
`frame_00008/C0001` trajectory it adds 768 ordinary rows and reduces deep sigma-1.5 high-pass and
Laplacian residual MSE by `25.926%/27.316%`, with a `10.46%` raw-LPIPS improvement and every
protected check passing. FIT-041's equal-base control shows that FIT-031 solves a different
problem: its 2,777-row error tail gains more global foreground PSNR, but places no rows in the
predeclared deep region and makes essentially no change to either fine-detail stop metric.

That is credible source-bound mechanism evidence, not independent confirmation. The method was
selected through FIT-033/037/038/039 on the same exposed image; FIT-040 and FIT-041 still leave
image generality, seed sensitivity, equal-row superiority, actual-rate efficiency, and
work-normalized efficiency unresolved. In particular, richer primitives and alternate solves
failed on that image while iterative remeasurement plus exact-site-only exclusion succeeded, but
one trajectory cannot establish allocation as the general bottleneck.

The 2026-07-28 evidence-completeness rerun under
`ara/evidence/fit031-new-method-stages-janelle-2026-07-28/` strengthens but does not broaden that
conclusion. FIT-032--038 reproduce their qualitative dispositions on a fresh RTX-4090 trajectory
with one common 10,816-row base, while FIT-039--041 remain the separately published exact-same-base
RTX-3050 tier. The rerun is still the same target, and exact-site pursuit was not rerun on its new
base, so it is replication of the precursor mechanism chain rather than independent-image or full
winning-method confirmation.

The subsequent user-requested
`ara/evidence/janelle-cross-view-tail-diagnostic-2026-07-28/` sweep adds useful but non-qualifying
transfer evidence. On all 51 remaining Janelle frame/view cells, pursuit reaches the protected
`25%/20%` detail target in `51/51` cells versus `7/51` for the natural FIT-031 tail, with median
added counts 384 versus 6,144; error-only still wins foreground-PSNR gain in every cell. These are
correlated views and adjacent frames from the same capture, one seed, archived mask-contained
bases, and natural unequal row counts. The sources were not the preregistered disjoint screen or
sealed confirmation set, and the run lacks FIT-042's equal-row controls, multi-seed analysis,
codec accounting, and work instrumentation, so none of its cells enter FIT-042's decision gate.

FIT-043 subsequently tests the distinct sequential-composition question on those same exposed
cells. Error→pursuit reaches the cumulative detail target in `51/51`, but its frozen exact-prefix
gate fails `43/44` because one inherited `scale_max` certificate refreshes at schedule entry; the
controller is rejected without retuning. That diagnostic neither weakens nor advances this task:
it reuses the exposed correlated cells, tests ordering rather than pursuit's equal-row generality,
and consumes none of the source manifests or confirmation authority below.

## Goal

Determine whether FIT-040's orthogonal pursuit mechanism reliably improves masked deep-interior
fine detail over equal-row generic residual births and static high-pass allocation on unseen
images, while preserving the accepted base field and protected mask metrics. Separately determine
whether its quality advantage survives actual-byte and measured-work accounting, and confirm that
FIT-031 remains the global/boundary objective control rather than a substitute.

## Frozen protocol

### Sources and base states

1. Commit two disjoint source manifests before fitting: a four-target killing screen and an
   eight-target sealed confirmation set, each with source/prepared image and mask hashes. Targets
   must span at least two frames or capture groups; report results by target and group.
2. Exclude Janelle `frame_00008/C0001`, every prepared source hash used by FIT-023--041, and any
   crop or transform derived from them. Selection may use source metadata and mask geometry only,
   never an arm's render or residual.
3. Require at least 4,096 pixels strictly deeper than `mask_margin + 6 px`; declare any other
   mask ineligible before generating a base field.
4. Prepare every target at max-side 1,200 through one version-pinned transform, materialize those
   exact target/mask bytes, and make every arm read the materialized files. Record Pillow and all
   other transform versions. A byte mismatch, including the Pillow 11 versus 12.3 Lanczos failure
   found by the stage rerun, quarantines the base and every dependent result.
5. Generate one `run_pipeline` base per `(target, seed)` with at least three seeds, persist it
   once, and bind every arm to identical field/target/mask/config hashes. Never regenerate a
   nominally same-seed base for another arm: CUDA may change the terminal row count and field
   trajectory across devices or environments. Device/environment changes are blocking factors,
   not pooled replicates. Record renderer, package versions, dirty state, and the CUDA
   nondeterminism caveat.

### Equal-row mechanism lane

Run nested 128-row waves and save snapshots at added-row budgets
`{128, 256, 512, 768, 1024, 2048}`. The experiment adapter must continue to the requested nested
budget rather than using FIT-040's Janelle-derived `25%/20%` early stop; production behavior stays
unchanged.

- `base`: no terminal tail.
- `orthogonal_pursuit`: FIT-040 unchanged -- current sigma-1.5 high-pass residual ranking,
  within-wave 5x5 NMS, exact-site-only cross-wave exclusion, 0.35 px isotropic rows at opacity
  0.8, inherited rows frozen, and an exact joint partial color solve after every wave.
- `generic_rgb_pursuit`: replace only the high-pass ranking score with current deep-region RGB
  residual MSE. Wave size, eligibility, geometry, opacity, NMS/deduplication, partial solve,
  protected gate, and row budgets remain identical.
- `static_highpass`: FIT-037's one-shot base high-pass ranking at the same nested budgets, with
  the same geometry, inherited-row freeze, joint partial solve, and protected gate. This is the
  control for iterative orthogonal remeasurement.

Any arm with insufficient unique sites records a terminal failure at that budget; it must not
silently reuse sites, widen eligibility, or change geometry.

An A/A identity or cold-replay check is non-degrading when the protected gate either accepts or
returns exactly `["no_material_gain"]`; this is not a scientific candidate pass. Every added-row
wave must receive an actual protected acceptance. Any A/A regression reason or any candidate
`no_material_gain` rejection still fails closed.

### Objective-control lane

Run the shipped FIT-031 error-only tail once from every identical base, unchanged and with its
natural requested/accepted count. Report global, boundary, deep-detail, row, byte, and work
outcomes, but do not enter this unequal-count arm into the primary equal-row gate. Also replay the
shipped FIT-040 production stop so the bounded operational behavior is reported separately from
the nested mechanism curve.

### Measurements

For every base, wave, and shipped-tail result, record:

- deep sigma-1.5 high-pass and Laplacian residual MSE, both absolute and relative to the base;
- foreground/boundary PSNR, MS-SSIM, raw LPIPS, CVaR99/p99 MSE, interior-hole fraction,
  containment, and exact outside-mask zero checks;
- improved/worsened deep-pixel fractions and the fraction of qualifying 32 px tiles with positive
  fine-detail change, so a concentrated average does not masquerade as broad recovery;
- added/active/physical rows, canonical and ordered site hashes, solve residuals/iterations,
  protected decisions, and termination reasons;
- tail wall-clock, renderer calls, gate evaluations, CG matvecs, and peak allocated memory;
- cold-decoded actual SSPL1 bytes/bpp and the same quality metrics under one pinned `CodecConfig`.
  NPZ size and analytical bits are not actual-rate evidence.

Write tidy JSONL/CSV rows per `(stage, target, seed, arm, added_rows)`, an aggregate JSON file, and
a portable `index.html`. Aggregate paired deltas by target, with target-cluster bootstrap
intervals and seed variation reported separately; do not pool all trajectories as independent.

## One-shot decision rules

1. **Killing screen.** Stop and record the independent generality hypothesis as refuted if no
   budget at or below 1,024 rows has `orthogonal_pursuit` ahead of both equal-row controls in
   paired mean high-pass and Laplacian reduction while every protected check passes. No retuning
   on these four targets is allowed.
2. If the screen survives, select the smallest qualifying budget by that frozen rule, seal it,
   and evaluate it once on the eight confirmation targets. The nested confirmation curve may be
   retained for diagnosis, but only the sealed budget enters the confirmatory gate.
3. **Bounded quality confirmation.** A recommended masked-image opt-in is authorized only if the
   target-cluster 95% confidence interval versus each equal-row control excludes zero in the
   favorable direction for both fine-detail metrics, at least six of eight targets have favorable
   seed-mean deltas on both comparisons, the paired raw-LPIPS interval excludes regression, and
   every protected check passes.
4. **Efficiency claims are separate.** Claim actual-rate or work efficiency only if the
   cold-decoded byte curve or measured time curve, respectively, clears the same paired
   fine-detail direction. A quality win with ambiguous cost remains a quality-only result.
5. FIT-031 winning global/boundary metrics while pursuit wins the declared deep-detail metrics
   confirms objective-specific dispatch; it is not evidence for combining the tails. Any combined
   or automatic selector becomes a new task.
6. Regardless of outcome, FIT-040 remains default-off. A default or current-recipe change requires
   a separate task and ADR decision.

## Acceptance criteria

- [ ] Source manifests, hashes, seed set, metric convention, exact statistical implementation,
      resource ceiling, codec config, and run commands are committed before the first
      result-bearing fit.
- [ ] Every result binds a clean commit or an archived exact executed-source snapshot, including
      dirty tracked diffs and result-relevant untracked files.
- [ ] Preflight fixtures prove equal base bindings, exact row budgets, inherited-row freezing,
      nested-snapshot consistency, the A/A-versus-candidate gate distinction, prepared-byte
      identity, cold decode, resume behavior, and per-cell failure isolation.
- [ ] The killing screen is evaluated exactly once; confirmation stays sealed if it fails.
- [ ] If authorized, the sealed confirmation completes without target-specific changes; all
      missing/error cells and exclusions remain visible in the aggregate.
- [ ] An independent artifact audit recomputes bindings, metrics, decision rules, spatial
      diagnostics, and cold-decode byte counts from persisted outputs.
- [ ] Evidence is committed under `ara/evidence/fit042-*` with config/provenance, tidy rows,
      aggregate/audit JSON, reproduction commands, and portable `index.html`.
- [ ] Add a supported or refuted ARA claim with the exact evidence scope; amend ADR-0030 and the
      task index with the bounded disposition. Do not rewrite C60's valid Janelle result.
- [ ] Run the results-audit and review skills, then `./scripts/verify.sh`.

## Interfaces touched

`scripts/experiments/fit042_*` and its focused tests; `scripts/experiments/README.md`;
`ara/evidence/fit042-*`; `ara/logic/claims.md`; ADR-0030 and `tasks/INDEX.md` only when recording
the result. Production fitting, renderer, primitive, codec format, and defaults are unchanged.

## Depends on

FIT-031, FIT-033, FIT-037, FIT-038, FIT-039, FIT-040, FIT-041, CORE-012, BENCH-001, BENCH-002,
COMP-001, ADR-0029, ADR-0030.

## Notes

This is the cheapest decisive generality test for the existing mechanism, not a new search over
scores, geometry, wave sizes, solvers, thresholds, or primitives. A negative result is productive:
it bounds C60 to the exposed source and closes independent promotion without disturbing the
default-off diagnostic interface. A positive result establishes only the scope actually sampled;
multiple views from one capture do not become a cross-scene claim.
