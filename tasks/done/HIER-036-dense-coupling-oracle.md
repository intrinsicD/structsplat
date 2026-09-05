# HIER-036 — Dense cross-Gaussian coupling oracle

## Context
HIER-035 compares diagonal and per-Gaussian block curvature, but does not intervene on
cross-Gaussian curvature or separate it from trust-cap choice. A small factorial control can
isolate those mechanisms before any scalable optimizer is built.

## Goal
Measure the effect of cross-Gaussian Gauss–Newton entries at fixed trust-cap semantics, with
separate comparisons against the strongest Adam control.

## Non-goals
- No production dense solver, default change, speed, natural-image, held-out, or novelty claim.
- No retuning after seeing formal outcomes; no uncharged Jacobian or trial-render work.

## Acceptance criteria
- [x] Bound the dense oracle and test Jacobian, Gram, update, ownership, and work semantics.
- [x] Freeze executable protocol and obtain distinct prospective digest approval.
- [x] Clean immutable complete run, native artifacts/curves, and independent results audit.
- [x] Synchronize task/docs/ARA and pass the full verification gate.

## Interfaces touched
Task-scoped benchmark oracle, experiment driver, focused tests, and portable report validator.

## Depends on
HIER-033/035, ADR-0006

## Agent workflow
- Driver: codex-root
- Reviewer: codex-overnight-protocol-reviewer
- Turn: none
- Reviewed revision: cb97dce4e0c4b927f7a9228f42afc570d210dcc1

### Handoff log
Historical initial entry: design reviewed before implementation; executable approval and outcomes were then pending.

### Handoff

#### Objective
Prospectively review the exact dense coupling/cap factorial before formal execution.
#### Changes
Bounded dense GN oracle, seven-arm driver, matched numerical bridges, strict artifact/gate tests.
#### Evidence
The source digest above;78focused tests pass, including CPU/CUDA system and update bridges.
No formal HIER-036 outcomes exist; reviewer must not inspect diagnostic results.
#### Assumptions
Known additive GN equations, finite support, proceduralN16 scope and explicit charged inner work.
#### Uncertainties
Whether coupling or cap choice changes terminal quality; no performance or scalability inference.
#### Review focus
Identical full/block gradient, ridge, bounds and line search; only coupling/cap axes differ.
Whole-matrix failure closure, separate strata/comparators, exact counters and raw scoring.
#### Protected actions not taken
No formal execution before approval, default change, sealed access, artifact repair, push or merge.
#### Recommended next action
Independently recompute source digest and approve or reject prospectively; then finish full
verification and clean-source commit before any formal run.

### Handoff

#### Objective
Independently audit the completed coupling/cap factorial and its separate Adam comparisons.
#### Changes
No post-approval numerical change; executed the exact approved clean source.
#### Evidence
results/hier036_coupling_2026-09-05, source4cd0331bdbb357df295a271e62736024feafdea9
and approved digest524da500f133c294248bd0e8f2f20a54271d79163d48b935d8366704ce78ea1e.
The source checkpoint passed2146portable tests and all structural gates; frozen bundle checker passes.
#### Assumptions
FixedN16 procedural fixtures; exposed and additional conditions reported separately.
#### Uncertainties
Practical visual benefit near precision limits, nonlinear texture behavior, scalability and timing.
#### Review focus
Regenerate inputs and matrices/updates; cold replay, uncapped raw scoring, both cap conventions,
separate coupling/Adam gates, full work/occupancy evidence and native artifact interpretation.
#### Protected actions not taken
No default change, matrix-limit increase, selective repeat, held-out access, artifact repair,
foreign-process termination, push or merge.
#### Recommended next action
Return a distinct scientific verdict and bind only its scoped findings into ARA.

### Handoff

#### Objective
Review the final evidence/claim/documentation integration for this completed bounded assay.
#### Changes
No post-freeze numerical change. Added the independent audit receipt, explicitly partial tracked
archive, scoped ARA findings and morning handoff; kept all maintained defaults unchanged.
#### Evidence
ara/evidence/overnight-method-research-2026-09-05/run.md and archive/archive_manifest.json;
docs/research/2026-09-05-overnight-findings.md; ARA C68–C72 as applicable.
Integration commitcb97dce4e0c4b927f7a9228f42afc570d210dcc1; full gate2163passed,26skipped,514deselected,
all structural checks clean. Only this workflow binding was appended after that checkpoint.
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

### Review

#### Verdict
Accepted

#### Self-reviewed
No

#### Correctness
Distinct reviewer codex-overnight-protocol-reviewer accepted this completed bounded research
task for retirement, not method/default promotion. Reviewed integration commit
cb97dce4e0c4b927f7a9228f42afc570d210dcc1, tree
6b82818bd1633b84444911c90d2117bf8cb212a9, plus the four disclosed task workflow-binding
diffs and N314 provenance correction.
C71 retains cap-specific and exposure-stratum scope. Conditional overlap coupling gains do not
establish texture optimizer preference; the ultimate texture failure cause remains unestablished.
Precision-limit polishing is not demonstrated visual improvement.

#### Evidence quality
The review independently reproduced all 6,039 archive hashes, 1,397 contained HTML links,
the exact copied/omitted partition and complete 6,040-file inventory; all five original
manifest/source/protocol/cell-count bindings match. Five packager tests and all five structural
checkers passed independently. Earlier distinct numerical audits provide the replay evidence.
The 2,163-test full gate was producer-run, not rerun by the reviewer. The archive is explicitly
partial; no original immutable artifact was repaired.

#### Simplicity
Implementations remain bounded experimental tools with maintained defaults unchanged.
O182/O183 and the proposed gradient rescue remain unpromoted, unimplemented and unrun.

#### Missing cases
Natural-image transfer, scalable dense alternatives and standalone-gradient synergy remain
unresolved. Any fallback assay needs a new task and prospective executable approval.

#### Required changes
No scientific, numerical-code or artifact correction required. Routine closure: record this
verdict, complete the acceptance checklist, retire the task, synchronize Index/session brief,
and pass the final repository verification gate before committing.

#### Optional improvements
Future documentation cleanup may improve number/unit spacing. Future assays require their own
frozen scope and exact prospective approval; no further experiment is authorized by this verdict.

## Frozen protocol
Executable authority: PROTOCOL plus every SOURCES hash in
`scripts/experiments/hier036_coupling.py`. Recompute with --print-protocol-digest. Formal
execution requires a distinct Approved digest receipt committed with clean source.
Command: python scripts/experiments/hier036_coupling.py
results/hier036_coupling_2026-09-05 --approved-protocol-digest EXACT_DIGEST.
Diagnostic --smoke uses translated77 and cannot establish a formal outcome.

- Same HIER-035 overlap/texture fixtures,64x64,N16. Exposed conditions0/1/2 and additional
  procedural conditions3/4/5 form separate reporting strata. They are not natural or statistically
  independent confirmation sets. Diagnostic wiring uses translated condition77 only.
- Seven arms: Adam multipliers0.3/1/3; block_row, block_shared, full_row, full_shared.
  Four curvature arms materialize the same image Jacobian, gradient and dense Gram. The two
  interventions are retaining cross-Gaussian Gram entries and row-wise versus shared trust caps.
  Compare full_shared with block_shared primarily; full_row with block_row is a secondary,
  independently reported intervention under HIER-035's cap convention.
- CUDA float32 owned additive renderer, C0 fade,3sigma,constant signed RGB; RTX3050;
  torch threads1; disable TF32 and require highest float32 matrix multiplication precision.
  Raw objective0.5mean RGB squared error, no mask/clamp in fitting. Report raw MSE, ceiling-
  limited PSNR, uncapped PSNR (null only for exactly zero MSE), and ceiling-applied flag.
  Display-clamped MS-SSIM and LPIPS are perceptual guards.
- J has shape(H*W*3,N*8); g=J-transpose*r/numel and H=J-transpose*J/numel. Shared parameter
  trust units(1,1,.1,.1,.1,.1,.1,.1), per-row damping0.01*maximum scaled row diagonal with
  floor1e-12, identical bounds and six halved trial renders. A shared cap divides the complete
  scaled direction by max(1,maxabs(direction)); a row cap does so per Gaussian. Explicit solve
  failures are errors, not fallback directions. Global finite-loss acceptance includes ties.
- Every arm attempts exactly160 updates, returns terminal rather than best state, preservesN16.
  Adam rates0.1/0.03/0.03/0.03 times multiplier, betas.9/.999,eps1e-8,foreachFalse.
  Bounds: means in canvas,scales[.35,16],RGB[-2,2]. Rejections retain the previous exact state.
- Dense J retained-array ceiling64MiB and maximum256parameters, checked before allocation.
  This is not a peak-memory bound: record allocated CUDA peak including Gram/solve work,
  worker RSS, retained J/Gram bytes, every Jacobian construction, solve and trial render.
  Complete-fit seconds are descriptive on the shared GPU, not a speed result.
  The worker ledger additionally counts all seven two-update warmups, four fixture-generation
  renders, one initial diagnostic render and one cold replay. Warmup has14 backwards/Jacobians
  in total, including8 dense constructions/solves; its trial-render count is measured.
  Perceptual network work is separate. Partial errors are preserved, not assigned zero work.
- Same-state bridge tests: dense gradient vs autograd and diagonal blocks vs analytic packet;
  dense block_row step vs HIER-035 step. CPU float64 tolerances rtol1e-6/atol1e-8; CUDA
  float32 gradient/Gram rtol1e-5/atol1e-7 and update rtol1e-3/atol1e-3, at condition77.
- Whole84-cell matrix must be complete, count/horizon/parity/input/config/trace valid before
  any positive gate. Per family and exposure stratum, coupling gate: median paired PSNR
  gain>=1dB, no condition loss>.1dB, MS-SSIM loss<=.005, LPIPS increase<=.01. Keep the two
  cap conventions' gates separate; passing either cannot substitute for the other.
- Separately compare each full-curvature arm with per-condition best-of-three terminal-PSNR
  Adam, stable lexical-method tie break, median gain>=.5dB and the same worst-case/perceptual
  guards. Passing coupling but failing Adam competitiveness is not optimizer preference.
  Near-ceiling gains remain numerical-polish evidence, not meaningful visual improvement.
- One fresh warmed worker per cell; rotate all seven arms by seed. Warm every method on
  translated77 for two steps; diagnostic run uses three steps. Timeout600seconds, retain
  all errors, no selective reruns or in-place repair. Native cold render tolerance2e-5 and
  exact decoded parameters. Initial/terminal fields, raw images, full traces, configs/source
  hashes, progress, native PNGs, curves, HTML and tidy JSON/JSONL/CSV required.

## Notes
Dense GN is a known diagnostic, not a new algorithm. Relevant primary implementations:
[LM-RS](https://vcai.mpi-inf.mpg.de/projects/LM-RS/) and
[3DGS-LM](https://lukashoel.github.io/3DGS-LM/). Their 3D pipelines are not matched native
baselines here. Any matrix-free or sparse production transfer requires a separate task.

### Protocol review

#### Reviewer
codex-overnight-protocol-reviewer

#### Verdict
Approved

#### Protocol digest
524da500f133c294248bd0e8f2f20a54271d79163d48b935d8366704ce78ea1e

#### Digest scope
Canonical executable PROTOCOL and every SOURCES hash: both task drivers, dense and local
control fitters, analytic packet, field/render/owned CUDA sources, metrics, report and checker.
Independently reproduced without accessing HIER-036 outcomes.

#### Outcomes accessed
No

#### Review focus
Independent86focused tests passed: CPU/CUDA system and old-update bridges, owned-CUDA gradient
parity, factorial isolation, memory guards, finite rejection, ownership/work accounting,
report contracts and HIER-035 regressions. No blocking correction identified. Row-wise caps
need not preserve a full-system descent direction; finite-loss backtracking is authoritative.
Positive results can establish only a conditional toy-system coupling effect, not scalability,
practical visual improvement near the PSNR ceiling, speed, or held-out confirmation.
