# HIER-034 — Fixed-geometry basis cache

## Context
The fixed-geometry color projection repeats tile and weight construction on every operator call.

## Goal
Compare streaming, cached scatter, and sparse operators with build cost, memory, and parity.

## Non-goals
- No maintained default change, count/rate claim, or cache reuse after geometry changes.

## Acceptance criteria
- [x] Bounded immutable cache and forward/transpose oracle tests.
- [x] Frozen protocol, distinct prospective review, and clean-source evidence.
- [ ] Portable report, independent audit, synchronized docs/ARA, full verification.

## Interfaces touched
`additive_basis.py`, `contraction_refinement.py`, tests and task-scoped experiment.

## Depends on
HIER-031/032/014, ADR-0006

## Agent workflow
- Driver: codex-root
- Reviewer: codex-overnight-protocol-reviewer
- Turn: reviewer
- Reviewed revision: 50396b6e06a3ff668fbea641ab6ae20c52ecaf1f

### Handoff log
Historical entry at the original freeze: protocol and implementation are prospectively approved at the digest below. The clean source
checkpoint is4b2c79f2a97e0bde5109d11ab717edd5881a1ca1. Formal timing remains unlaunched while an
unrelated GPU workload is active; no other process was stopped and no speed claim is made.

### Handoff

#### Objective
Prospectively review the shared-resource correctness amendment before any formal cache outcome.
#### Changes
Separate shared-profile driver, externally sampled worker occupancy, explicit timing eligibility,
retained numerical interchangeability gates and report checks for resource/decision bindings.
#### Evidence
The exact source digest above;63focused tests pass, including original cache/projection regressions.
The original exclusive-resource source/protocol remains preserved and unrun.
#### Assumptions
Shared-GPU execution is permitted only for numerical agreement and descriptive resource evidence.
#### Uncertainties
Acceleration remains unassessed; shared-resource failures cannot establish intrinsic infeasibility.
#### Review focus
No numerical/matrix/limit change, no possible speed promotion, complete occupancy provenance,
separate numerical eligibility, exact source/input bindings and preserved errors.
#### Protected actions not taken
No formal execution before amended approval, foreign-process termination, default change,
selective rerun, in-place repair, push or merge.
#### Recommended next action
Approve or reject the amended digest prospectively; require full verification and clean source
before launching the separately named immutable shared-correctness bundle.

### Handoff

#### Objective
Independently audit the shared-profile correctness bundle, then the preserved original timing
matrix when it completes. Keep the two protocol dispositions separate.
#### Changes
No post-approval numerical change. Shared execution used clean50396b6e06a3ff668fbea641ab6ae20c52ecaf1f.
The original complete matrix uses preserved clean4b2c79f2a97e0bde5109d11ab717edd5881a1ca1.
#### Evidence
At this handoff, results/hier034_shared_correctness_2026-09-05 was complete. The original
results/hier034_basis_cache_2026-09-05 was running; its external observational sidecar is
results/hier034_timing_occupancy_2026-09-05. Both retain their original approved digests.
The latest source checkpoint passed2158portable tests and all structural gates.
#### Assumptions
Shared-profile speed eligibility remains false regardless of observed occupancy or ratios.
The original assay is not an independent numerical confirmation: it uses the same frozen matrix.
#### Uncertainties
Backend transaction equivalence; float32 path sensitivity; generalization and isolated timing.
A fresh preflight found no GPU compute process, but an unrelated CPU compilation was later
observed. Point-sampled GPU occupancy does not prove complete workstation exclusivity.
#### Review focus
All source/input/metric/configuration bindings, checkpoint predicates and selected iterations,
paired raw rasters, complete matrix and paired timing gates, occupancy chronology, bounded
memory scope, repeated baseline variability, and native display artifacts.
#### Protected actions not taken
No foreign process stopped, selective repeat, threshold change, failed-artifact repair,
default promotion, push or merge. Original timing execution was approved prospectively before
either formal cache outcome and explicitly admitted by the reviewer as the unanswered assay.
#### Recommended next action
Lightweight shared-bundle inspection may proceed now. Defer GPU and heavy CPU replays until
the original timing matrix finishes, then return a separate scientific verdict for both scopes.

### Handoff

#### Objective
Review the final evidence/claim/documentation integration for this completed bounded assay.
#### Changes
No post-freeze numerical change. Added the independent audit receipt, explicitly partial tracked
archive, scoped ARA findings and morning handoff; kept all maintained defaults unchanged.
#### Evidence
ara/evidence/overnight-method-research-2026-09-05/run.md and archive/archive_manifest.json;
docs/research/2026-09-05-overnight-findings.md; ARA C68–C72 as applicable.
The exact staged integration tree and final full-gate result are bound at review dispatch.
#### Assumptions
Each claim retains its own frozen protocol, source, data, comparator and resource scope.
The two cache profiles are not independent numerical confirmation.
#### Uncertainties
Generalization, meaningful visual improvement, isolated speed, accepted-update cache utility,
baseline repeat mechanism and the unrun gradient-rescue design remain unresolved.
#### Review focus
Check that the ledger faithfully records your distinct audits, preserves nominal failed/passing
gates, and includes numerical-polish, rollback, within-baseline variability and contention caveats.
Review the archive helper's immutable-input and sidecar fixes; do not rerun optimization.
#### Protected actions not taken
No default change, new formal matrix, selective rerun, original-artifact edit, foreign-process
termination, sealed-data access, push or merge.
#### Recommended next action
Accept or require a bounded correction to the integrated record. Retire this task only after
the distinct verdict and full verification; keep future hypotheses explicitly unrun.

## Notes
Overnight research authorized September 5, 2026.

## Frozen protocol

The executable PROTOCOL in scripts/experiments/hier034_basis_cache.py is authoritative for
the matrix and resolved choices. Its digest additionally binds the driver, report/checker,
projection/cache, field adapters, metrics, maintained reference, and owned CUDA sources.
Recompute with: python scripts/experiments/hier034_basis_cache.py --print-protocol-digest.

- Question/null: does retaining finite-support weights reduce complete fixed-geometry projection
  time? Null: neither cache achieves at least 1.1x median paired speedup with every parity gate.
- Arms: unchanged off; cached scatter; two-direction CSR. Same PCG defaults (48 maximum
  iterations, tolerance 1e-6, ridge 1e-8, input start/regularization, transaction selection,
  absolute coefficient bound 16), C0 fade, three-sigma support, float32, CUDA additive, chunk256.
- Data: three procedural 128x128/144-row families (smooth overlapping, rotated thin, irregular
  mask), seeds 0/1/2; one hash-bound exposed HIER-031 1200x1038/N7000 field with colors multiplied
  by 0.97. This is a color-solver workload, not a geometry/quality improvement or held-out assay.
  Original geometry and raw mask remain fixed. Existing HIER-031 diagnostic lineage persists.
- Pairing: family/seed/repeat. Six repeats cover all six lexicographic backend-order permutations.
  Each cell runs in a new process with all three libraries/backends warmed on synthetic seed77.
  Warmup, input IO, post-run perceptual scoring, export, and cold validation are outside the
  primary interval; the complete projection call, including cache build, all PCG/checkpoint work,
  and maintained replay, is inside it. GPU synchronization brackets that interval.
- Resources: RTX3050, one torch CPU thread, 256MiB retained-cache ceiling, 600-second worker
  timeout. Peak allocated/reserved GPU memory spans the entire projection; process peak RSS also
  includes imports/warmup/input loading. Neither is represented as cache storage alone.
- Integrity: exact count and geometry, cold decoded coefficients, raw active-pixel cold/maintained
  parity <=2e-4. Cached-versus-off active-pixel difference <=2e-4 and raw SSE discrepancy
  <=1e-4*max(off_SSE,1e-6). Inspect selected-iteration/transaction disagreement; any disagreement
  prevents a blanket interchangeable-backend conclusion. Record all coefficient differences.
  Compare every checkpoint's iteration, bounded/selectable/transaction booleans and trace length;
  normalized-violation disagreement beyond absolute 1e-6 also excludes interchangeability.
- Report: raw masked PSNR/SSE, display black-matted MS-SSIM/LPIPS, attempted/selected iterations,
  forward/transpose counts, complete checkpoint traces, build time, retained bytes, measured wall
  time and memory; target/field/reconstruction/error/curve artifacts and source/input hashes.
- Decision: workload-specific median paired projection acceleration only if all six pairs are
  complete, pass parity, agree on selection, and execute at least two iterations. Report ratios
  and dispersion within each workload; do not treat repeats as independent source images.
- Missing/OOM/timeout/nonfinite/budget errors remain explicit rows and exclude that workload from
  positive selection. No limit increase, selective rerun, in-place repair, or threshold rescue.
- Formal run uses a fresh results/hier034_basis_cache_2026-09-05 directory, from a clean reviewed
  commit/worktree, with --base-bundle pointing to the frozen HIER-031 bundle and
  --approved-protocol-digest set to the distinct review receipt. --smoke uses synthetic seed77
  only, one backend order and three iterations, with preserved dirty source; it is wiring only.

## Design sources
docs/research/2026-09-05-overnight-research-portfolio.md records the research alternatives.

## Prospective shared-resource amendment

At this prospective amendment, the occupied GPU prevented an exclusive-resource timing interpretation. Before any formal
HIER-034 outcome exists, narrow the next run to correctness/parity and descriptive resource
telemetry. Preserve the original timing protocol/source at4b2c79f2a97e0bde5109d11ab717edd5881a1ca1
and /tmp/structsplat-overnight-VmVXmg/source; it remains unrun.

The amended executable authority will be PROTOCOL plus SOURCES in
`scripts/experiments/hier034_shared_cache.py`, with a new distinct prospective digest approval.
Keep all180cells, six permutations, inputs,48-step solver, numerical/checkpoint/count gates,
256MiB retained ceiling,600second timeout, warmup and error policy unchanged. Record GPU
occupancy before/after every worker plus a parent-process monitor throughout the matrix.
All elapsed times and ratios are descriptive; timing eligibility and every speed gate must
be false irrespective of the measured ratio. Performance disposition is inconclusive due to
shared-GPU contention, not a refuted speed hypothesis. Numerical interchangeability keeps
the original six-complete-pair gates. Shared-resource OOM/timeouts do not prove intrinsic
cache infeasibility. No foreign process is stopped, and no selective repeat is permitted.

New immutable output: results/hier034_shared_correctness_2026-09-05. Native inputs and
base-bundle path remain the original hash-bound HIER-031 lineage. Formal command:
python scripts/experiments/hier034_shared_cache.py results/hier034_shared_correctness_2026-09-05
--base-bundle results/hier031_janelle_c0001_s1200_exact7k_boundary_detail_s0_diagnostic_2026-08-12
--approved-protocol-digest EXACT_DIGEST. Diagnostic smoke stays smooth77, three backends,
three iterations; it cannot establish a formal outcome.

### Protocol review

#### Reviewer
codex-overnight-protocol-reviewer

#### Verdict
Approved

#### Protocol digest
a5be4997c39a13d59eada1962a67837ff447bf9145e44b5dcca3eaaaacb66436

#### Digest scope
Canonical executable PROTOCOL plus SHA256 of every file in the driver's SOURCES list,
including projection/cache, semantic adapters, maintained render/CUDA sources, metrics and
portable report/checker. Recomputed before and after independent review, unchanged.

#### Outcomes accessed
No

#### Review focus
Paired saved-state integrity; all-checkpoint decisions; build-inclusive timing; six-repeat
counterbalancing/completeness; retained and peak memory scope; error preservation; prospective
source/commit identity; fault-injection tests. Independent focused gate: 32 tests passed.
Approval is prospective and workload-specific, not a result or default approval. Initial
intermediate digest 24bb46bdc7153e8bda77d2210cdf1b620eb48acc7fa9505d20efe0df009d4a9a was not
approved because decision and provenance gates were incomplete; the corrections above close
those issues. Post-run independent evidence audit remains required.

### Protocol review

#### Reviewer
codex-overnight-protocol-reviewer

#### Verdict
Approved

#### Protocol digest
9e3dc020572b724646f0f323a73d9457839f31a1c674b2ae799031787b2ff439

#### Digest scope
Shared-profile executable PROTOCOL plus all SOURCES: wrapper, original driver, numerical
implementations, report and checker. The11bound numerical implementation files independently
match the original preserved source byte-for-byte; matrix, solver, inputs and limits are unchanged.

#### Outcomes accessed
No

#### Review focus
Independent63focused tests passed. Timing gates are unconditionally false; numerical
interchangeability stays separate. Occupancy hashes/intervals and decision coverage are checked;
monitoring ends before manifest sealing. Failed workers killed by timeout/SIGKILL can lack
their after-snapshot: retain that as explicitly partial telemetry, never complete coverage.
Performance is inconclusive regardless of descriptive ratios. Full verification, clean source
and an independent post-run audit remain required. No blocking corrections identified.
