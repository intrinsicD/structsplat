# HIER-013 — Independent-image global projection development screen

## Status

In review. Completed negative development diagnostic on repository test images; not sealed
confirmation, not claim-ready, and not a default authorization. Distinct review remains pending.

## Context

HIER-012 found that projecting all 7,000 RGB coefficient rows on fixed HIER-005 geometry reduced
masked MSE by 40.26/38.29% on two exposed correlated Janelle views. Those views selected the
mechanism, the source was dirty, and there was no distinct prospective reviewer. The next useful
question is whether the frozen mechanism transfers to the independently supplied COCO/DIV2K
images under a full-frame exact-count protocol without another search.

## Goal

Measure the frozen HIER-012 global RGB projection against its strongest attribution controls on
all supported images under `tests/test_images`, with image-level paired statistics, three CUDA
replicates, complete work/reference-stream accounting, portable artifacts, and fail-closed
interpretation.

## Non-goals

- Do not tune projection, contraction, exchange, thresholds, images, or seeds after outcomes.
- Do not call these repository training images held-out, sealed, confirmation, or production data.
- Do not select Field V2 semantics, complete FIT-046, claim codec rate/speed, or change a default.
- Do not compare the additive endpoint directly with normalized plain fit as if the renderer
  semantics were matched; BENCH-020 owns that comparison.

## Frozen protocol

### Question and scope

Does HIER-012's all-row fixed-geometry RGB solve preserve its large remaining-error reduction on
the complete requested repository image set? A positive result may retain global projection as the
next development component for FIT-046. It cannot promote a maintained pipeline.

### Data

- Inputs are exactly the 16 files bound by `EXPECTED_SOURCES` in
  `scripts/experiments/hier013_global_projection_development.py`: four top-level COCO JPEGs and
  twelve `tests/test_images/DIV2K_train_HR/*.png` images. The driver rejects missing, additional,
  renamed, or hash-mismatched sources.
- Every source is deterministically resized with Pillow LANCZOS to maximum side 512. Evaluation
  uses an all-true full-frame mask; no alpha/mask selection is inferred.
- COCO and DIV2K are reported separately and together. The image, after averaging its paired CUDA
  replicates, is the inference unit. Seeds `0 1 2` are replicate labels and RNG bindings; the
  topology has no stochastic sampler, while CUDA recovery/renderer atomics can vary numerically.
- All requested inputs are development data. No confirmation set is opened or implied.

### Arms

Every cell has exactly 7,000 rows and starts from the same replicate-specific HIER-005 field:

1. `h005_control` — frozen HIER-005 contraction.
2. `touched_projection` — HIER-010 matrix-free projection restricted to contraction-touched rows.
3. `global_projection` — HIER-012 matrix-free projection over all 7,000 RGB rows.
4. `exchange_global_projection` — HIER-011 guarded residual exchange followed by the same all-row
   projection.

The projection uses ridge `1e-8`, tolerance `1e-6`, at most 48 PCG iterations, coefficient
absolute limit 16, and HIER-012's raw-SSE/displayed-normalized-violation transaction. Exchange uses
HIER-011's oriented eight-shape bank, 128-pivot cap, 96 residual sites, radius-1 NMS, 64 donors,
and a 24-proposal frontier. Contraction uses HIER-010's exact-count HIER-005 configuration,
including 16 progressive touched-recovery checkpoints with 50 CUDA steps apiece. Renderer is
`cuda_additive`, render chunk 256, device CUDA, target count 7,000, and LPIPS is required.

### Metrics, resources, and missing policy

- Primary: paired raw full-frame MSE ratio and PSNR delta for `global_projection` versus
  `h005_control`.
- Guardrails: MS-SSIM, LPIPS, display 8-bit pixel/7x7 maxima, exact count, bit-exact non-RGB
  arrays, projection checkpoint safety, coefficient bounds, adjoint error, maintained/cold/repeat
  renderer parity, and finite rows.
- Resources: contraction/exchange/projection seconds, projection forward/transpose applications,
  accepted/proposed/cold-rendered exchanges, pipeline algorithm seconds, metric/packaging time,
  peak CUDA allocation, canonical raw bytes, and the exact complete lossless Observation Field
  NPZ bytes. NPZ bytes are a self-contained reference stream, not a production codec-rate claim.
- All 192 expected cells (16 images × 3 replicates × 4 arms) must remain visible. Any missing,
  failed, non-finite, or LPIPS-unavailable cell fails the development gate; no survivor-only
  summary or in-place repair is permitted.

### Frozen decision rule

Average paired log-MSE ratios within image, then across the 16 images. Use 20,000 deterministic
image-cluster bootstrap resamples with seed 13013.

`global_projection` passes the development gate only if all of the following hold:

- all 192 cells are complete and every integrity/count/hash/parity/transaction predicate passes;
- geometric-mean MSE ratio versus HIER-005 is `<= 0.80` and its 95% image-bootstrap upper bound
  is `< 1.0`;
- COCO and DIV2K geometric-mean MSE ratios are each `< 1.0`, and no paired cell has higher MSE
  beyond relative tolerance `1e-8`;
- aggregate MS-SSIM does not decrease beyond `1e-7`, aggregate LPIPS does not increase beyond
  `1e-7`, and neither displayed pixel nor 7x7 maximum increases in any paired cell;
- median global-projection overhead divided by HIER-005 contraction time is `<= 0.25`.

Lower geometric-mean MSE chooses between direct global and exchange-plus-global only when the
aggregate LPIPS and both local maxima do not favor the alternative. Otherwise record a
heterogeneous Pareto outcome rather than silently scalarizing it. A failed gate is retained as a
negative/inconclusive development result; these images may not be retuned for this task.

### Exact command

```bash
PYTHONPATH=src python scripts/experiments/hier013_global_projection_development.py \
  --images tests/test_images \
  --out results/hier013_global_projection_test_images_development_2026-08-10 \
  --seeds 0 1 2 --target-gaussians 7000 --max-side 512 \
  --projection-ridge 1e-8 --projection-tolerance 1e-6 \
  --projection-max-iterations 48 --projection-coefficient-limit 16 \
  --max-exchanges 128 --site-count 96 --site-nms-radius 1 \
  --donor-count 64 --proposal-frontier 24 --coefficient-limit 16 \
  --device cuda --renderer cuda_additive --render-chunk 256 --lpips
```

## Acceptance criteria

- [x] The driver validates the exact source bank and frozen axes before outcome access, isolates
      every cell, persists all raw rows/checkpoints/fields/visuals, and never overwrites a bundle.
- [x] The report contains the 192-cell ledger, image/family/aggregate paired statistics, bootstrap
      interval, complete reference bytes, work/memory telemetry, representative full-resolution
      links/crops, and an explicit diagnostic decision.
- [x] Cold decoding reproduces every persisted field/count and all non-RGB arrays remain bit-exact
      within each arm's declared geometry source.
- [ ] `scripts/check_report_bundle.py` passes with the diagnostic dirty-source allowance, focused
      tests pass, and the outcome receives a results-audit and ARA evidence disposition.
- [x] Docs/task state are synchronized and the repository verification gate is run with all
      unrelated failures disclosed.

## Interfaces touched

`scripts/experiments/hier013_global_projection_development.py`, narrow HIER schema registration in
`scripts/check_report_bundle.py`, focused tests, this task/Index/session brief, report/evidence
artifacts, and outcome-only ARA/docs synchronization.

## Depends on

HIER-012, BENCH-002, ADR-0006

## Reversible fallback

HIER-005 and every maintained path remain unchanged. Projection step zero is the exact field
fallback, and a negative result simply retires this successor from further promotion.

## Diagnostic outcome (2026-08-10)

All 192 cells complete, but the frozen development gate fails. Global projection averages only
`+0.011700 dB` and a `0.269%` MSE reduction; the image-bootstrap MSE-ratio interval is
`[0.9932034, 1.0000000008]`, mean MS-SSIM regresses `7.77e-5`, and four active cells worsen the
7x7 maximum. Only DIV2K/0268 and 0534 run PCG: the other 42/48 global cells enter with coefficient
maxima above 16 (median 91.797, maximum 2010.808) and safely return step zero. All six active
solves hit iteration 48 and gain only `+0.0687/+0.1185 dB` by image.

Exchange plus global is stronger at `+0.072495 dB` and `1.655%` aggregate MSE reduction with
better mean perceptual/local metrics, but remains small, heterogeneous, and higher-work. Visual
review finds obvious square/lattice artifacts in representative full frames and native crops, so
neither arm is an exceptional or promotable pipeline.

File/canonical hashes, counts, row ledgers, and source bindings replay 192/192. The frozen CUDA
parity gate fails: cold-versus-in-memory parity reaches `0.001137`, and 141/192 rows exceed `2e-6`;
the report-bundle checker therefore fails with 141 parity findings even under `--allow-dirty`.
This is retained as an integrity limitation rather than repaired post hoc.

Evidence:
`ara/evidence/hier013-global-projection-test-images-development-2026-08-10/run.md`.
Report:
`results/hier013_global_projection_test_images_development_2026-08-10/index.html`.
Manifest SHA-256:
`e7315213558050b88b72470211a54779e7bb0df6aee3ce06d0d0144cdfe616a3`.

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `e7315213558050b88b72470211a54779e7bb0df6aee3ce06d0d0144cdfe616a3`

### Handoff log

No formal `### Protocol review` is claimed: the user selected the data bank, but the dirty tree
and absence of a distinct outcome-unseen reviewer restrict execution to a development diagnostic.

### Handoff

#### Objective

Test whether HIER-012's exact-7k all-row RGB solve transfers from two exposed Janelle views to the
complete requested COCO/DIV2K repository image bank without retuning.

#### Changes

Added the bounded HIER-013 driver, source/hash/protocol bindings, three-replicate four-arm report,
paired image-bootstrap analysis, schema registration, focused tests, and scoped evidence/task/docs
records. No maintained pipeline or numerical method changed.

#### Evidence

The 192-cell report manifest is
`e7315213558050b88b72470211a54779e7bb0df6aee3ce06d0d0144cdfe616a3`.
Global projection fails the frozen material, bootstrap, MS-SSIM, local, and integrity clauses;
hash/count/source replay passes, while the checker independently reports 141 parity failures.

#### Assumptions

All repository images are development-only; full-frame N=7,000 is the requested operating point;
seeds label CUDA replicates; reference NPZ bytes are complete persistence but not codec rate.

#### Uncertainties

The dirty tree and absent distinct review prohibit a formal claim. CUDA atomics and ill-conditioned
coefficients create parity drift. No independent plain-normalized or selected-semantics arm ran.

#### Review focus

Verify the 42/48 coefficient-bound fallbacks, six capped PCG trajectories, paired aggregation,
parity findings, obvious lattice artifacts, and the refusal to raise the cap after outcomes.

#### Protected actions not taken

No threshold/cap retuning, selective rerun, confirmation access, default change, FIT-046/BENCH-020
promotion, codec/rate claim, or scientific-state mutation was performed during execution or
evidence review. The user-requested repository publication is a later operational handoff and does
not alter the preserved bundle.

#### Recommended next action

Prospectively stabilize or constrain the incoming coefficient domain on new development data before
testing another all-row solve. Do not promote exchange or retune these 16 consumed images.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

The complete matrix and exact hashes/counts replay. The decision fails closed. The report checker
failure agrees with the recorded parity-integrity clause rather than contradicting the decision.

#### Evidence quality

The data bank is broader and paired across three replicates, but it is exposed development data on
a dirty tree without prospective review. The bundle is not claim-ready because numerical parity
fails.

#### Simplicity

The negative result localizes the blocker to HIER-005 coefficient conditioning. No post-hoc cap
increase or additional mechanism is added to this task.

#### Missing cases

Clean source, distinct review, stable bounded coefficients, selected Field V2 semantics, an
independently approved new bank, matched plain-fit controls, and bundle-clean parity remain open.

#### Verification

All 59 focused HIER-010/011/013 and pixel-contraction tests pass. The required
`./scripts/verify.sh` wrapper was run with the repository Python and a narrow Ruff module shim:
whole-tree Ruff passes, and the portable suite reaches 1,739 passed, 25 skipped, and three failures
in untouched subsystems (the affine rank-deficient condition-number expectation, missing CUDA
`pci_bus_id` property, and a descriptor path-swap race expectation). Because the wrapper stops at
pytest, its docs, ARA, task-policy, script-layout, and agent-workflow structural gates were also run
directly and all pass. The HIER-013 bundle checker continues to fail its intentionally retained
141-row CUDA parity finding; HIER-010/011/012 bundle checks pass.

#### Required changes

None for retaining this negative diagnostic. Promotion is rejected.

#### Optional improvements

Future reports should separate stage-zero precondition failure from completed PCG convergence in
the headline matrix and use a scale-aware, prospectively justified numerical-parity contract.

## Notes

Write a new output directory for any implementation correction discovered after first outcome
access. Do not mutate or selectively rerun the original result bundle.
