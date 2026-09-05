# FIT-051 — Actual-render color transactions

## Context
FIT-050's first formal source is e2bf6ae6e06ca8050d9aa8a93d713679a0c9c150. Its retained bundle
is results/code-driven-2026-09-05/fit050-v1. The prospective compatibility check prevented most
ray proposals from being evaluated; the original frozen utility decision remains unchanged.
This follow-up changes the proposal/evaluation mechanism, not its tolerance or quality gate.

## Goal
Determine whether actual-render fraction evaluation and a renderer-native color gradient allow
useful safe progress from the same prescribed parents, without cross-backend image interpolation.

## Non-goals
- No relaxed gate, more fractions, repeated ray, new data, topology, default or novelty claim.
- No claim that the prior line search was ineffective on cases aborted before a direction.
- No code changes in the separate source worktree used by the still-running PORT-007 study.

## Acceptance criteria
- [x] Native autograd and actual-render trials pass CPU dense/ownership tests and CUDA checks.
- [x] Distinct exact prospective approval and clean immutable complete comparative report.
- [x] All parent provenance, raw trials, gates, work and complete artifacts independently verified.
- [x] Original negative evidence retained; independent results audit and final docs/ARA/gate.

## Interfaces touched
New opt-in actual-render refinement module, bounded driver, tests and report task registration.
Original method/default behavior stays unchanged.

## Depends on
FIT-050, ADR-0003, ADR-0011

## Agent workflow
- Driver: codex-root
- Reviewer: codex-code-research-reviewer
- Turn: none
- Reviewed revision: 8de800406e608d7c7a47cc3dfc56217ed69bbb53

### Handoff log
New development hypothesis, implementation and exact prospective freeze pending. Work is isolated
from the active PORT-007 source commit.

### Final outcome handoff — 2026-09-05

Prospective text is historical design, not current status. Original protocol/task bytes remain
at the exact source revision; current closure prose does not rewrite a frozen digest.

### Handoff

#### Objective
Close this completed bounded assay after distinct numerical and evidence-integrity review.
#### Changes
Implemented opt-in methods/controls, ran the complete approved matrix and preserved all artifacts.
Final integration adds a partial archive, scoped findings and ADR-0034; report-only mask geometry
reuse caches no images, metrics or decisions.
#### Evidence
All56 cells in results/code-driven-2026-09-05/fit051-v1 complete from clean source
8de800406e608d7c7a47cc3dfc56217ed69bbb53. Original checker and independent CPU/GPU/raw/work audits pass.
See ara/evidence/code-driven-method-research-2026-09-05/run.md and ARA C75.
#### Assumptions
Exposed development images, frozen reporting/safety contracts, descriptive image-level units,
charged instrumented work and point-sampled rather than continuous resource observations.
#### Uncertainties
Practical perceptual utility, generalization, production speed, numerical cause and novelty
remain unestablished. Preserve every study-specific limitation in the evidence note.
#### Review focus
Complete cells/hashes, actual field/raw/gate/work bindings, baseline controls, failed utility
gates, explicit archive omissions and no-promotion wording.
#### Protected actions not taken
No default/tolerance change, sealed-data access, selective repeat, immutable-result repair,
foreign-process termination, cloud spend or push. Final local integration is fast-forward only.
#### Recommended next action
Retire after accepted outcome audit and final verification. Any new mechanism/claim needs its
own prospectively reviewed task; no further experiment is implied.

### Review

#### Verdict
Accepted
#### Self-reviewed
No
#### Correctness
Distinct reviewer codex-code-research-reviewer accepted this source-bound assay for bounded
findings and retirement, not default promotion. ActualCG accepts8/8 with four within-transaction fractional rescues, but median+0.005293dB misses0.1dB utility. Native gradient has no demonstrated quality advantage; independent CG directions differ and accepted CVaR may use existing numerical slack.
#### Evidence quality
Original maintained report gate passes without allowances. Independent source/artifact/metric/
decision/work and GPU native replay pass; exact counts/tolerances are in the evidence note.
Native browser and raw links were inspected. The archive is partial; complete originals remain.
#### Simplicity
Retain experimental tools and existing defaults; no complexity or tolerance rescues a failed
utility gate. Report geometry caching changes validation cost only.
#### Missing cases
No held-out/full-resolution, general speed/perceptual, actual-rate, downstream3D or novelty
evidence follows. Native and masked workloads beyond this protocol remain untested.
#### Required changes
No numerical implementation/scientific correction remains. Final integrated records and full
repository verification are mandatory before the closure commit.
#### Optional improvements
A separate diagnostic may isolate baseline null-gain sensitivity/rejected work. Older HIER
rescue and other portfolio candidates remain explicitly unrun.

## Prospective design (executable freeze pending)
Reuse all eight initial/750-step parents from the complete source-bound FIT-050 bundle, preserving
input/config/optimizer hashes and their originating commit. Four exposed COCO images, seeds0/1,
max-side512, N2000. No new parent selection or tuning on another dataset.
Seven one-transaction arms: noop, legacyCG32 endpoint, actual-render CG fractions, actual-render
streaming gradient fractions, actual-render streaming Jacobi fractions, renderer-native gradient
fractions, inherited-moment Adam32. Each logical arm pays its own construction and evaluation.

All trial images are actual maintained renders of changed fields, never interpolated images.
Streaming directions are explicitly approximate cross-backend proposals, not exact CUDA
Jacobians, and use the actual parent-render residual, not literal FIT-050 direction vectors.
Native gradients use autograd of the maintained renderer with respect to colors only;
geometry, opacity and support are frozen. For non-CG directions render q for the proposed color
vector and use alpha=(residual*q).sum/(q.square().sum+ridge*direction.square().sum).
CG uses fractions of its independently computed endpoint (alpha1).
Keep ridge1e-4, six ordered fractions1,1/2,1/4,1/8,1/16,1/32, first actual safe nonzero change,
complete unchanged quality vector, and exact rollback. Cold selected-field replay remains required.

Retain the same utility threshold: median of four image-level seed-mean PSNR gains>=0.1dB,
no cell loss>0.01dB, MS-SSIM loss<=0.001, LPIPS increase<=0.002, all safe-gate fields protected.
Comparison to parent alone is not preference over CG/Adam. Report all seven arms and all
image-seed pairs; descriptive development evidence only. Actual-render work replaces the
cross-backend interpolation shortcut and is fully charged, not called free.
Protocol/source digest, exact commands, raw-array/temporal artifacts, counters and error policy
must be completed and distinctly approved before any formal execution.

## Executable protocol and reproduction
`scripts/experiments/fit051_actual_color_ray.py` owns PROTOCOL and SOURCES. The source digest
binds canonical protocol JSON plus the SHA256 of all listed package Python, CUDA, report,
driver and control sources. The original FIT-050 manifest is bound to SHA256
`5c5629df090b7946f1ee85ab98d988921998096888261c40192e7cd3a7a4f427`;
the 48 explicitly enumerated parent payload hashes preserve all eight targets, initial/terminal
fields, optimizer states, configurations and histories. Cross-commit parent transfer is explicit;
there is no refit, new natural-image data, or selection among the original parents.

Print the prospective protocol without executing outcomes:
`python scripts/experiments/fit051_actual_color_ray.py --protocol-only`.

After distinct exact-digest approval and a verified clean source commit:
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u scripts/experiments/fit051_actual_color_ray.py --out /home/alex/Documents/structsplat/results/code-driven-2026-09-05/fit051-v1 --parent-bundle /home/alex/Documents/structsplat/results/code-driven-2026-09-05/fit050-v1 --approved-protocol-digest DIGEST`.

Use the pinned Python3.12/torch2.9.0+cu128/RTX3050 environment and one torch CPU thread.
The new output directory must not exist. A new process per parent warms all seven arms on a
procedural 64-pixel fixture, then rotates the seven prescribed arms by parent index. Each arm
owns its proposal, actual quality evaluations and captured tensors. Native backward may compute
unused parameter-gradient buffers internally; its whole invocation is charged. The cost is an
instrumented transaction, not an uninstrumented production-speed estimate. Cold reader replay,
perceptual scoring, serialization and plots are separately reported rather than called free.
Endpoint and curve scores use canonical CPU float32 copies of the retained images with one
CPU thread. Native CUDA/reference safety decisions are unchanged. This scoring contract does
not authorize direct perceptual-metric comparisons to the older FIT-050 reporting backend.

Retain native fields, exact CG endpoints, signed direction images, gradients/diagonals, all
actual trial images and raw denominators, selected replay, rejected control endpoints, configs,
optimizer histories, full quality vectors, first-safe versus final-replay acceptance, and exact
work counters. Actual quality-evaluation plots are distinct from native optimizer-step histories.
All 56 prescribed cells, all six non-noop arms, and all eight image-seed pairs must be reported.
The four image-level seed means are descriptive development units, not eight independent images.
No held-out, novelty, p-value, population confidence interval, speed, or default claim is opened.
Retain errors and partial outputs; a source correction requires a fresh freeze and directory,
not selective reruns or in-place repairs. Point-sampled foreign GPU activity disqualifies timing
without erasing numerical outcomes. Tiny CPU/CUDA diagnostics do not count as scientific outcomes.

### Prospective review status
The distinct static mechanism review found no blocking defect. It required explicit native-
residual semantics, exact saved-CG endpoint reconstruction, trial-versus-replay distinction, and
a tiny CUDA check after the PORT-007 timing window. Exact executable approval remains pending.
An independent source review also required canonical method/stage operand inventories, so
removing a direction/native-forward image cannot be concealed by lowering the work counters.
Procedural-only pre-run checks rejected the assumption that CUDA LPIPS and CPU LPIPS agree at
the frozen reporting tolerance. Canonical CPU reporting was selected before formal execution;
no tolerance was widened and no original report was changed. The revised GPU-input/CPU-scoring
checks pass, alongside native VJP and actual-trial CUDA diagnostics.

## Notes

This is a fresh mechanism test, not an in-place correction or selective rerun of FIT-050.

### Protocol review

#### Reviewer
codex-code-research-reviewer

#### Verdict
Approved

#### Protocol digest
e1c3a421cef3e546f8135d405ee1315b76c47e11f9286b8ea68170930ee32010

#### Digest scope
Canonical JSON of PROTOCOL and SHA256 of every SOURCES entry in
`scripts/experiments/fit051_actual_color_ray.py`, computed by
`benchmarks.hier_research_report.protocol_digest`. Both protocol CLI forms reproduce the
complete inventory/digest without executing formal outcomes.

#### Outcomes accessed
No

#### Review focus
Distinct reviewer `/root/overnight_protocol_reviewer` independently recomputed this digest twice
and reran 122 method/control/artifact CPU tests (4 GPU/integration tests deselected). Canonical
operand inventories, method/stage-derived work, exact CG endpoint, first-safe actual trials,
selected replay, unchanged gates, immutable parent transfer, complete matrix and descriptive
aggregation were checked. One detached CPU scorer serves endpoint and all curve metrics;
no direct old FIT-050 perceptual-value comparison is permitted. Drivers separately report three
passing CUDA diagnostics; the prospective reviewer did not independently rerun those checks.
FIT-050 and PORT-007 are acknowledged prior/parallel exposed evidence, not FIT-051 outcomes.
Approval remains bounded to this new 56-cell development protocol and requires a verified clean
source commit and complete report validation. No speed, default, held-out or novelty claim.
