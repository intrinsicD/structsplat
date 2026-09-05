# HIER-033 — Pixel-gradient operator oracle

## Context
The August 12 anatomy derives additive gradients but leaves finite operator value untested.

## Goal
Compare parameter-family signals against finite local edits and fixed optimizer recovery.

## Non-goals
- No default change, downstream 3D transfer, sealed images, or quality claim from synthetic cases.

## Acceptance criteria
- [ ] Analytic sums and curvature match renderer/autograd.
- [ ] Freeze fixtures, candidate bank, controls, recovery, metrics, gates, and source digest.
- [ ] Distinct prospective review and clean immutable run with portable raw artifacts.
- [ ] Independent audit, synchronized task/docs/ARA, full verification.

## Interfaces touched
Pixel-gradient reference, focused tests, and task-scoped driver.

## Depends on
HIER-031/032, ADR-0006

## Agent workflow
- Driver: codex-root
- Reviewer: codex-overnight-protocol-reviewer
- Turn: reviewer
- Reviewed revision: 7417be6

### Handoff log
The finite protocol is prospectively approved at the digest below. Frozen-source execution
has finished; the distinct artifact audit is in progress. This handoff envelope was recorded
while replay review was underway, not before that review request; no disposition was promoted.

### Handoff

#### Objective
Audit the finite operator atlas and its predeclared same-case selection gate.
#### Changes
Analytic packet, count-funded finite bank, identical recovery, portable driver and validator.
#### Evidence
Source7417be6; results/hier033_operator_oracle_2026-09-05 and its exact manifest/digest.
The source checkpoint passed2114portable tests and all repository structural gates.
#### Assumptions
Procedural additive reference semantics only; finite bank and donor budget remain explicit.
#### Uncertainties
Independent interpretation and ARA disposition remain pending; no general topology policy.
#### Review focus
Regenerate fixtures/bank; cold decode and raw-image replay; both-phase regret, cancellation,
fixed count, donor costs, work counters, complete artifacts and frozen protocol identity.
#### Protected actions not taken
No default change, sealed/natural-image access, selective rerun, artifact repair, push or merge.
#### Recommended next action
Finish the distinct scientific audit, then bind its bounded findings into the claim/evidence ledger.

## Notes
Design: docs/research/2026-08-12-hier-pixel-gradient-anatomy.md.

## Frozen protocol

The bounded assay uses six procedural defects (translation, width, rotation, RGB,
two lobes, and residual outside current finite support), seeds0/1/2, 64x64 images and three
initial Gaussians. This is mechanism evidence, not a synthetic-to-natural quality claim.

All finite trial states keep N=3. Continuous move/scale/rotation/RGB edits act on the designated
parent row at two frozen magnitudes. Symmetric splits halve the parent's RGB, preserve its
scales, and explicitly delete one of two donor rows. Residual-peak births replace one of those
same donors. A no-op control is retained. The resulting fixed bank has15 candidates per case;
every donor's reconstruction damage is included in the measured objective.

Pixel signed/absolute gradients, local Gram blocks and split matrices propose local scores,
but finite trial renders are authoritative. Compare prediction regret with the unrestricted,
continuous-only and split-only finite oracles, explicitly labelled privileged controls.
Evaluate every candidate immediately and after the same20-step Adam recovery, reset optimizer
state for every trial, and preserve all recovered states. The central question is whether
gradient cancellation identifies an edit family, not whether an unrestricted candidate search
can improve training loss. CPU reference execution avoids the occupied GPU and has no speed claim.

Executable authority is PROTOCOL in scripts/experiments/hier033_operator_oracle.py together
with the exact source hashes in SOURCES. Recompute using --print-protocol-digest. Formal command:
python scripts/experiments/hier033_operator_oracle.py results/hier033_operator_oracle_2026-09-05
--approved-protocol-digest EXACT_DIGEST. The task's distinct prospective approval must be
committed with clean source before launch. --smoke is translation condition77, all15 actions,
two recovery updates and preserved diagnostic source; it cannot establish an outcome.

- Primary hypothesis/null: local packet scores identify low-regret edits in this finite atlas.
  Null: more than20% of conditions exceed10% normalized regret either immediately or after
  recovery. Positive selection requires all18 cases/all270 cells complete and integrity-valid,
  and at least80% of the same cases have regret<=0.1 jointly at both measurement phases. Separate
  per-phase fractions are descriptive and cannot replace this intersection gate. Regret is the unrestricted
  finite oracle's gain minus the predicted choice's gain, divided by max(base_objective,1e-8).
  Stable ties use the frozen bank order. No policy/default or broader quality conclusion follows.
- Conditions0/1/2 deterministically vary parent angle0/0.25/0.5radians and donor cost; they are not
  independent random samples. All families and exact target/field formulas are source-bound.
  No natural or sealed image is used. Condition77 has a distinct angle and is wiring-only.
- CPU float32 reference additive renderer, 3sigma C0 fade, constant signed RGB, no mask,
  alpha/filter/affine terms. CPU is i9-11900KF, one torch thread. Images64x64; every trial and
  terminal field has exactly3 rows. Raw objective0.5 mean squared RGB error; PSNR floor MSE1e-12;
  MS-SSIM/LPIPS score display-clamped output. No speed claim on the shared CPU.
- Continuous candidates solve parent-only move/scale/rotation/RGB local GN groups, damping0.01
  times maximum scaled group diagonal (floor1e-12), trust units(2,2,.1,.1,.1,.1,.1,.1), cap maximum
  group trust ratio and try multipliers0.5/1. Split candidates use the least-eigenvalue position
  direction, displacements0.5/1 times the parent's minimum scale, half RGB and unchanged scales;
  remove donor1 or2. Birth candidates replace donor1 or2 with a1.6px atom at the row-major maximum
  residual-energy pixel, initialized by signed peak residual divided by its C0 peak weight.
- Scores are local approximations, not full finite oracle evaluations: continuous quadratic
  gain; split curvature plus exact linear donor-removal cost; single-pixel birth gain proxy minus
  donor cost. Finite support changes and donor/birth interactions can invalidate these scores.
  Record their failures instead of silently rescoring or switching candidates after the trial.
- Privileged unrestricted, continuous-only(+no-op), and split-only(+no-op) finite oracles are
  references, not official/native deployable baselines. They all derive from the same bank.
  Secondary falsification: position activity>1e-8, coherence<.01, and best continuous immediate
  gain exceeding every funded split(+no-op) by>1e-8 is a cancellation-to-split counterexample.
  This does not refute any published 3D theorem outside these additive equations and budgets.
- Recovery is the frozen ControlConfig: fresh-state Adam20 updates per candidate, parameter
  rates0.1/0.03/0.03/0.03, betas.9/.999, eps1e-8, means inside canvas, scales[.35,16], RGB[-2,2],
  terminal state only. Recover no-op identically. Row forward/gradient counters are recovery-only
  (21/20 formal). Separate fields count2 additional renders per cell (immediate and cold), and6
  shared renders per case (two target-generation renders, three warmup-fit renders, one proposal
  base render). Shared gradient work is two warmup backwards and one analytic pixel packet.
  Completed formal cases therefore invoke351 Gaussian renders; partial-error work is not fully
  reconstructible from successful rows and is not represented as zero. Per-row total_seconds is
  complete recovery time, not the entire case pipeline. Perceptual networks are separate work.
  report proposal_seconds once per case (duplicated row metadata must not be summed15 times).
- One fresh worker per case, deterministic bank order, two-update condition77 warmup,
  600second case timeout. Preserve partial successes and explicit errors; any incomplete case
  prevents a positive whole-atlas verdict. No repeated cells, threshold rescue or in-place repair.
- Artifacts per cell: original and edited starting field, terminal field, raw base/immediate/final
  renders and target, exact source/input/config bindings, complete history/progress, target/
  immediate/final/error images and iteration/time curves. Per-case signed/absolute/Gram/split
  packet is saved. Native cold-decoded parameters/count must be exact and reference pixel max
  error<=1e-7. Portable index.html plus tidy JSON/JSONL/CSV and every decision predicate are required.

### Protocol review

#### Reviewer
codex-overnight-protocol-reviewer

#### Verdict
Approved

#### Protocol digest
bfa19a882ccf107d6f82626cf1aee5b232f0e13fbf8c8bd8c39ecfdeba3ecd59

#### Digest scope
Canonical executable PROTOCOL plus every source hash in SOURCES, including the finite bank,
recovery fitter, analytic packet, renderer/field, metrics, report and artifact checker.
Independently recomputed after the joint-case decision and render-ledger corrections.

#### Outcomes accessed
No

#### Review focus
Count-funded donor deletion, deterministic ties, frozen predictor approximations, identical
recovery, saved-state integrity, complete matrix, same-case joint regret gate, and explicit
shared/proposal/recovery/replay work accounting. Independent focused verification:35 tests
passed. The reviewer had previously audited HIER-035, but accessed no HIER-033 smoke or formal
outcomes before this approval. Conclusions remain confined to this finite procedural mixed
gradient/curvature/residual-birth selector; no unrestricted split theorem, 3D, natural-image,
speed, or default claim is authorized.
