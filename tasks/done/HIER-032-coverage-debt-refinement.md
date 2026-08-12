# HIER-032 — Coverage-debt refinement for masked hair and boundaries

## Context

HIER-031's hash-bound selected exact-N7,000 four-array field has no raw support holes and renders
exactly zero outside the mask, but it still has 743 pixels below unit coverage 0.05 in 483
deterministic 8-connected components. Of those, 461 are in the fixed hair crop; the weak set
contributes 38.816% of foreground SSE. The frozen diagnosis and recomputation recipe are recorded
in `ara/evidence/hier032-coverage-debt-diagnosis-2026-08-12/run.md` and staged as O169.

The method deliberately combines known components: pixel-error allocation from
[Revising Densification](https://arxiv.org/abs/2404.06109), cancellation-resistant absolute detail
signals from [AbsGS](https://arxiv.org/abs/2404.10484), and medial/set-cover selection from
[Coverage Axis](https://arxiv.org/abs/2110.00965). The validated portfolio and adversarial prior-art
audit are `docs/research/2026-08-12-hier032-coverage-debt-portfolio.md`. This task tests a
source-specific relationship and makes no novelty claim.

## Goal

Determine whether certified positive-coverage closure, contribution-aware donor merging, and an
optional boundary high-pass batch can improve both boundary and fixed hair-crop quality over the
frozen HIER-031 field at exactly 7,000 persisted Gaussians, without weakening exact containment or
the interior quality floor.

## Frozen development protocol

### Question, hypothesis, and null

- Question: can rows be moved from locally redundant ordinary pairs to weak masked hair/boundary
  sites while preserving the exact four-array N=7,000 endpoint?
- Hypothesis: certified component cover plus exact local donor-error ranking will close all unit
  coverage below 0.05 and improve both boundary and hair-crop PSNR without taking interior PSNR
  below 35.2631 dB.
- Null: no arm passes every count, coverage, outside-zero, boundary, hair, and interior clause.
- Allowed decision: select one task-scoped diagnostic arm, or select none and retain the best
  tradeoff as negative diagnostic evidence. No default or generalization decision is authorized.

### Inputs, roles, and source identity

- Exposed development input only: canonical Janelle C0001 source SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b` and mask
  SHA-256 `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`,
  Pillow LANCZOS/nearest resized to exactly 1200x1038, seed 0.
- Frozen HIER-031 selected field:
  `artifacts/deep_only_terminal_closure_n7000/field.gaussian.npz` in its immutable local bundle,
  SHA-256 `a0a080ccbd255ce51f11489cd504956a1c5181a495bbca2b4bf74ecb0995c1db`.
- The base field came from a dirty-source, sequential, producer-reviewed diagnostic. A clean
  HIER-032 source commit and prospective review do not upgrade that input provenance.
- CUDA additive renderer, RTX 3050, 256-row chunks, C0 compact-support fade, margin 0.75 px,
  sigma cutoff 3, LPIPS required. CUDA replay is source/device/version-bound, not bit-exact.

### Exact five-arm matrix

Every successful endpoint persists exactly 7,000 rows and exactly `means`, `log_scales`,
`rotations`, and `colors`. Geometry is frozen after each successor batch.

1. `hier031_selected_control_n7000`: unchanged hash-bound HIER-031 selected field.
2. `fallback_per_weak_pixel_n7000`: one certified isotropic 0.08-px successor at every current
   weak pixel, funded by the HIER-031 mutual-nearest merge score.
3. `component_set_cover_n7000`: deterministic component/candidate cover, funded by the identical
   HIER-031 merge score.
4. `component_set_cover_contribution_merge_n7000`: the same candidate and greedy placement policy;
   its first-wave placement digest must equal arm 3, while donor pairs are ranked by exact local
   additive SSE after recertification and a local RGB least-squares fit.
5. `coverage_then_boundary_highpass_n7000`: arm-4 coverage closure recomputed from the frozen
   control, followed by one fixed 128-row boundary/thin-structure high-pass batch using the same
   contribution-aware funding.

### Coverage detector and candidate selector

- Debt is `max(0, 0.05 - unit_coverage)` inside the raw mask. Weak pixels use strict coverage
  `<0.05`; raw holes use `<=0.0`. Label row-major deterministic 8-connected components.
- Each weak pixel contributes a guaranteed centred 0.08-px fallback. Where the SDF normal is
  reliable, also propose centres at inward-normal offsets 1, 2, 3, and 4 px with the maximum
  mask-tangent ellipse admitted by the existing ADR-0019 station-ball certificate.
- Build exact sparse C0-faded candidate-to-weak-pixel incidence. Greedily select by: newly
  satisfied weak pixels descending; covered remaining deficit mass descending; weighted source-
  RGB appearance variance ascending; stable candidate ID ascending.
- Render exact unit coverage after every allocation wave. Use at most four waves and at most 1,536
  successor placements total. Missing candidate completeness, insufficient donor pairs, remaining
  weak pixels, or a reopened final debt is an explicit arm error; thresholds are not retuned.

### Donor funding and detail batch

- Both donor modes use disjoint ordinary mutual-nearest pairs with both centres at SDF>2 px, a
  1.05x covariance envelope, anisotropic recertification, existing-micro exemption, one absorbed
  row per successor, and the existing bounded global additive RGB projection.
- The HIER-031 mode ranks by its fixed distance/color/log-scale/axial-angle score.
- The contribution-aware mode subtracts the two original rows from the current additive render in
  the union support box, recertifies the merged geometry, fits its RGB coefficient by exact local
  least squares, and ranks by local merged SSE, then SSE delta, then stable pair ID.
- Every selected projected endpoint must retain coefficient absolute maximum <=16.0; an unbounded
  stage-zero rollback is an explicit arm error, not a selectable endpoint.
- The detail arm scores absolute high-pass residual after Gaussian blur sigma 1.5, restricts to the
  mask with SDF<=4, uses deterministic 2-px NMS, orients maximum certified strokes along the source-
  luminance image tangent, and allocates exactly 128 rows. It performs no later geometry fit or
  closure rescue.

### Metrics, report, and frozen decision

- Primary integrity: exact N, exact four-array payload, equal in-memory/cold decoded-field hashes,
  zero decoded-state max difference, maintained/repeated
  render parity, centre containment, unit coverage outside, reconstruction outside, and complete
  execution-error ledger.
- Coverage: raw holes, weak pixels/components/largest component/deficit mass, boundary/interior/
  ridge fractions, and min/q0.1%/q1%/q5%/median/q95%/max unit coverage.
- Quality: foreground PSNR/MS-SSIM/LPIPS/SSIM, boundary<=4 PSNR, interior>4 PSNR, fixed foreground
  hair-crop PSNR, fixed boundary-crop PSNR, high-pass/Laplacian/Sobel error, pixel maximum, and 7x7
  maximum.
- Work: candidate-bank size/incidence/compression, detector/incidence/selector timing, donor-pair
  counts/timing, local fitted/area-color merge SSE, placement counts, waves, and method time.
- A candidate passes only with exactly 7,000 four-array rows; zero raw holes; zero pixels below
  0.05 coverage; support and reconstruction outside <=1e-7; maintained render parity <=2e-5;
  boundary PSNR and fixed hair-crop PSNR each strictly above the paired HIER-031 control; and
  interior PSNR >=35.2631 dB.
- Among passing arms select highest foreground PSNR, with frozen arm order as an exact-tie breaker.
  If none pass, select no method and report the lexicographic best tradeoff by weak count,
  boundary PSNR, hair PSNR, foreground PSNR, then arm order.
- Any missing or errored arm makes the five-arm matrix incomplete and forbids selection. Arm 4
  fails unless arm 3 completed and their first-wave placement digests match; arm 5 fails unless
  arm 4 completed and their complete coverage-placement digests match. A sealed error bundle is
  immutable diagnostic evidence but does not pass `scripts/check_report_bundle.py`.
- The immutable portable report must expose source, reconstruction, absolute error, coverage debt,
  component, placement, donor, unit-coverage, fixed hair, and fixed boundary views for every
  arm, plus JSON/JSONL/CSV tables, fields, histories, hashes, limitations, and errors. Its HTML
  contains only bundle-local links.

### Execution identity

Protocol digest command:

```bash
PYTHONPATH=src python scripts/experiments/hier032_coverage_debt_refinement.py \
  --print-protocol-digest
```

Digest: `402588c6c32a93ac1dca615ad50d2cf15248892beaaae1bf80cd9f9e253c9898`.
The digest covers the canonical JSON `PROTOCOL` constant in the HIER-032 driver, binding input and
base hashes, clean named branch, RTX 3050 identity, renderer/chunk/LPIPS/support controls, exact
payload and decoded-state receipts, coverage/candidate/donor/projection/detail rules, coefficient
and parity bounds, crops/metrics, budgets, arm relationships, matrix-error policy, selection and
negative ordering, portable-report policy, and forbidden actions.

Formal command from a clean named linked worktree on
`agent/hier032-coverage-debt-refinement` at the protocol/implementation commit:

```bash
PYTHONPATH=src python scripts/experiments/hier032_coverage_debt_refinement.py \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png \
  /home/alex/Documents/structsplat/results/hier032_janelle_c0001_s1200_coverage_debt_s0_development_2026-08-12 \
  --base-bundle /home/alex/Documents/structsplat/results/hier031_janelle_c0001_s1200_exact7k_boundary_detail_s0_diagnostic_2026-08-12 \
  --max-side 1200 --seed 0 --device cuda --lpips
```

No resume, in-place repair, selective rerun, changed output directory contents, outcome-dependent
arm, or rescue threshold is allowed. A driver defect requires retaining the failed bundle and a
new task decision before any rerun.

## Non-goals

- No count scaling, native-5328 run, mask relaxation, post-render masking, endpoint mask payload,
  scale-floor/default change, maintained API, actual-rate/compression claim, held-out confirmation,
  or global novelty claim.
- No claim that unit-coverage closure reconstructs every internal hair or that one exposed field
  generalizes to other masks, images, seeds, counts, or devices.

## Acceptance criteria

- [x] Task/Index/session brief and the validated portfolio/prior-art audit are synchronized.
- [x] Deterministic components, fallback completeness, greedy cover, station-ball containment,
      contribution-aware ranking, exact count, decision gates, and report schema have focused tests.
- [x] Frozen protocol/implementation commit passes focused tests, portfolio validation, task
      digest review, and `./scripts/verify.sh` before the evidence run.
- [ ] A new clean-source immutable five-arm bundle passes `scripts/check_report_bundle.py`. This
      remains explicitly unmet: arm 5 failed closed after reopening nine weak pixels, and the
      frozen checker correctly rejects the incomplete success matrix.
- [x] Results receive an adversarial audit; task/docs/ARA receipts preserve either the passing
      selection or the negative no-selection outcome without retuning.
- [x] Final correctness self-review and `./scripts/verify.sh` pass with maintained defaults
      unchanged; the scientific outcome review is independent rather than self-reviewed.

## Interfaces touched

One driver under `scripts/experiments/`, focused tests, the task-specific report schema in
`scripts/check_report_bundle.py`, this task/Index/session brief, the validated research portfolio,
and result-driven ARA evidence/trace records. No public package API or maintained default.

## Depends on

HIER-031/030/029, CORE-010/011/012, FIT-040/046, BENCH-002, ADR-0003/0006/0019/0033

## Agent workflow

- Driver: codex-root
- Reviewer: codex-hier032-protocol-reviewer
- Turn: none
- Reviewed revision: f4cc2996d525b128ea511b96e3a7357009f347d7

### Handoff log

Completed negative after prospective protocol approval and an independent adversarial outcome
audit. No method was selected.

### Notes

The full report remains ignored under `results/`; only source-bound hashes, conclusions, and the
reproduction command will be committed under `ara/evidence/`. The `.idea` worktree state is
unrelated user material and is excluded from every commit.

The immutable outcome is under
`results/hier032_janelle_c0001_s1200_coverage_debt_s0_development_2026-08-12/`. Four arms persist
valid exact-N7,000 endpoints; all three successors close coverage and improve boundary/hair PSNR
but breach the interior floor. The fixed detail arm reopens nine weak pixels and fails closed.
`selected_arm` is null. The report's manifest SHA-256 is
`598d7f59ed87c2c5f0bbb6d17e32e2c8c236f7b3174640052a85b146392beb14`; the complete audit is
`ara/evidence/hier032-coverage-debt-refinement-2026-08-12/run.md`.

### Protocol review

#### Reviewer
codex-hier032-protocol-reviewer

#### Verdict
Rejected

#### Protocol digest
4302c7347acb481027e1ceb02200c89bba5edab62d57231b10a85008be634d3f

#### Digest scope
The original compact key-sorted `PROTOCOL` JSON only; executable code, task prose, checker, tests,
portfolio, dependencies, and Git revision were outside that digest.

#### Outcomes accessed
No

#### Review focus
Controls, leakage, budgets, metrics, killing rule, source/environment identity, digest coverage,
implementation alignment, and portable-report enforcement. Formal execution was rejected because
portable links/provenance conflicted with the checker, the matrix and gates failed open, the digest
underbound material controls, and decoded-field/coefficient receipts were incomplete.

### Protocol review

#### Reviewer
codex-hier032-protocol-reviewer

#### Verdict
Approved

#### Protocol digest
402588c6c32a93ac1dca615ad50d2cf15248892beaaae1bf80cd9f9e253c9898

#### Digest scope
Canonical compact key-sorted JSON serialization of the expanded driver `PROTOCOL`; it binds
source/base hashes, branch and RTX identity, representation, coverage, donor, projection/detail
controls, budgets, metrics, arm dependencies, decision rules, report portability, and forbidden
actions. The separately reviewed driver/checker/focused-test SHA-256 values are respectively
`f572b27a7da13db3d2bf02512b44e11daac423c8fdb47596c875cf91cc6f79da`,
`ba62e5ea809e156abfb6ac103e8eaaa9706fcb75df61053c75129b9aa9d569d4`, and
`ab503fe8906cdfef8d2322789f0367891f8438d3e0a6bf56f5809e2cb346f529`.

#### Outcomes accessed
No

#### Review focus
Controls, leakage boundaries, budgets, metrics, killing rule, source/environment identity,
implementation alignment, all six first-review blockers, and local-only HTML/crop exposure. The
reviewer authorized formal execution from the exact committed implementation on the clean named
branch and frozen RTX device, with a new empty output directory.

### Handoff

#### Objective
Determine whether certified coverage closure, contribution-aware donor merging, and a fixed
boundary high-pass batch improve boundary and hair quality at exact N=7,000 without losing the
protected interior or containment guarantees.

#### Changes
Added the default-off HIER-032 driver, fail-closed task-specific report checker, twelve focused
tests, validated research portfolio/prior-art audit, frozen task protocol, and diagnosis/outcome
ARA receipts. No public API, maintained method, representation, count, mask policy, or default
changed.

#### Evidence
Commit `f4cc2996d525b128ea511b96e3a7357009f347d7` passed the pre-run repository gate and exact
prospective review. The 153-entry manifest is hash-consistent. Four persisted fields independently
pass exact N/four-array/decoded-state/containment/outside-zero/parity checks. Every completed
successor closes the 743 weak pixels but fails the 35.2631 dB interior floor; arm 5 fails closed
after reopening nine pixels. The decision correctly selects no method. Native visuals and the
portable Chrome-rendered report were inspected.

#### Assumptions
The known source/mask and 1200x1038 raster are exposed development data; the HIER-031 field is the
hash-bound control despite its dirty-diagnostic lineage; RTX 3050/source-version replay is not
bit-exact across other devices.

#### Uncertainties
One image/seed/device, no persisted arm-5 intermediate morphology, no held-out or statistical
replication, no native-camera or actual-rate evidence, and no general claim about coverage closure
or donor ranking.

#### Review focus
Check the fail-closed checker interpretation, exact field/hash/metric receipts, first-wave
relationship, interior-floor rejection, null selection, visual interior degradation, and whether
the immutable error remains unmodified.

#### Protected actions not taken
No mask relaxation, post-render masking, count change, threshold rescue, selective rerun, output
repair, default change, method promotion, or `.idea` modification/commit.

#### Recommended next action
Close HIER-032 as negative. If the nine reopened pixels motivate investigation, create a new
prospectively reviewed task that persists failure-state intermediates to a new immutable output.

### Review

#### Verdict
Accepted with follow-up

#### Self-reviewed
No

#### Correctness
The clean reviewed commit and immutable bundle are hash-bound and internally consistent. Four
persisted endpoints independently satisfy the exact 7,000-row four-array, finite-coefficient,
containment, outside-zero, and renderer-parity contracts. The fifth arm reached its frozen detail
batch only after the contribution-arm coverage-placement comparison and then failed closed because
the batch reopened 9 weak pixels. The resulting incomplete matrix, null selection, and checker
failure follow the prospectively approved error policy; they are not a successful report-gate
result.

#### Evidence quality
Accepted as source-, seed-, device-, and resolution-bound negative development evidence. All 153
manifested files, source snapshots, table projections, field/state hashes, gates, placement
relationships, and decision arithmetic were independently checked. The completed candidates close
coverage and improve boundary/hair PSNR, but all breach the frozen interior floor and worsen
MS-SSIM/LPIPS. Arm 5 has no persisted endpoint or failure morphology, so only its sealed error
event is supported.

#### Simplicity
No maintained method or default is selected. The result kills the tested known-component
relationship at its frozen scope without adding production behavior.

#### Missing cases
Fresh images, additional seeds/devices, held-out confirmation, native resolution, actual-rate
accounting, and an independently inspectable arm-5 failure state are absent. The HIER-031 input
field also retains dirty-diagnostic lineage.

#### Required changes
Record HIER-032 as a completed negative task in the task, Index, and ARA evidence ledger; preserve
the exact bundle hashes and explicitly state that `scripts/check_report_bundle.py` does not pass.
Leave the checker-pass acceptance item unmet, select no method, keep defaults unchanged, and pass
the final repository verification gate before closure.

#### Optional improvements
If the reopened-debt morphology motivates future work, create a new prospectively reviewed task
that persists fail-closed intermediate state to a new immutable output. Do not repair, rerun, or
retune this bundle.
