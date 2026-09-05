# FIT-050 — Safeguarded normalized color-ray refinement

## Context
The normalized fitter has a linear RGB subproblem, but the current CG proposal is expensive and
the safe schedule accepts or rejects the complete endpoint. Rejected full steps may contain a
useful smaller correction. This is a new candidate/acceptance question, not a rescheduled FIT-010
or a relaxation of the failed event-color gate.

## Goal
Test whether one cheap safeguarded color direction yields useful accepted real-image improvement
at fixed geometry/count, against unchanged CG and continued Adam.

## Non-goals
- No default change, compression, masked generalization, held-out, or global novelty claim.
- No geometry/opacity changes in color-only arms; full-parameter Adam is the explicit control.
- No guard relaxation, outcome-tuned threshold, or repeated ray rescue.

## Acceptance criteria
- [x] Tiny dense and finite-difference tests establish the direction, scalar optimum and replay.
- [x] Frozen executable protocol has distinct prospective approval and clean-source execution.
- [x] Complete real-image report includes every trial, accepted/rollback distinction and full work.
- [x] Independent outcome review, ARA/docs synchronization and full repository verification.

## Interfaces touched
New `src/structsplat/color_ray.py`, focused tests, task-scoped driver/report validation.
Reference solver and maintained conversion defaults remain unchanged.

## Depends on
ADR-0003, ADR-0011

## Agent workflow
- Driver: codex-root
- Reviewer: codex-code-research-reviewer
- Turn: none
- Reviewed revision: e2bf6ae6e06ca8050d9aa8a93d713679a0c9c150

### Handoff log
Design selected from independent code-reading lanes; executable approval and outcomes pending.

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
All48 cells in results/code-driven-2026-09-05/fit050-v1 complete from clean source
e2bf6ae6e06ca8050d9aa8a93d713679a0c9c150. Original checker and independent CPU/GPU/raw/work audits pass.
See ara/evidence/code-driven-method-research-2026-09-05/run.md and ARA C73.
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
findings and retirement, not default promotion. All utility gates fail;21/24 ray transactions abort compatibility before any direction. Do not infer general line-search/preconditioning failure.
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

## Frozen prospective design
Four canonical exposed COCO development images (IDs9,25,30,34), full frame, Pillow LANCZOS
max-side512, seeds0/1, quadtree_wse N2000. No DIV2K confirmation data. Every parent comes from
the same prescribed 750-step CUDA normalized L2-only Adam fit with fixed topology and terminal
selection; preserve parent field, complete history, optimizer state, source/config/input hashes.

One transaction per method from an identical parent: unchanged/no-op; legacyCG32; independently
executed legacyCG32 algorithm/config with interpolation; unpreconditioned residual ray; exact-diagonal Jacobi
residual ray; 32 additional Adam updates with inherited moments. The latter is a practical
continuation control with unequal work, not a mechanism comparator. All methods use the same
whole-image L2 objective; ridge1e-4 pulls toward the parent. Colors remain signed/unclamped.

For rays, r=target-Ac, g=A-transpose*r; v=g or g/(diag(A-transpose*A)+ridge). Compute q=Av and
alpha=(g*v).sum/(q.square().sum+ridge*v.square().sum), the algebraically equivalent
coefficient-space numerator used by the implementation. Reject invalid/nonpositive directions.
Test alpha times1,1/2,1/4,1/8,1/16,1/32, first safe useful candidate; each candidate image is
parent_render+alpha*q. CG interpolation uses the same six fractions of its solved endpoint.
No geometry/opacity projection is allowed; actual maintained render and full unchanged safe
commit gate revalidate the selected field. A failed revalidation restores the exact parent;
do not try another fraction after a replay failure. Measure every proposal/trial/replay/metric.

Primary utility is a nonzero selected correction with median image-averaged PSNR gain>=0.1dB
over the parent, no image-seed loss>0.01dB, MS-SSIM loss<=0.001 and LPIPS increase<=0.002;
all existing safe-commit metrics must also pass. Useful improvement is separate from a gate
accepting a numerical tie or rollback. Report quality and complete transaction cost against
CG32, interpolatedCG32 and Adam32 separately; no preference from the no-op comparison alone.
Independent seeds remain clustered within four exposed images, not eight independent images.

Record raw/scored images, native fields, complete ray trial curve, inherited Adam continuation
history, scalar metrics, wall time including construction/checks, peak CUDA/RSS, render/operator
calls, failure reasons and actual selected fraction. Point-sampled occupancy qualifies timing;
foreign GPU activity makes speed evidence ineligible but does not erase numerical outcomes.
Exact executable sources, digest, command and smoke fixture will be frozen before any formal run.

## Executable protocol and reproduction
The driver `scripts/experiments/fit050_color_ray.py` owns the complete PROTOCOL dictionary and
SOURCES inventory. Its SHA256 binds canonical JSON of that dictionary plus each listed file's
SHA256, including all package Python sources, owned CUDA sources, report validation, driver,
controls and the four exact JPEG inputs. Print without executing outcomes:
`python scripts/experiments/fit050_color_ray.py --protocol-only`.

Formal command, after the distinct digest approval below and full verified clean commit:
`python scripts/experiments/fit050_color_ray.py --out results/code-driven-2026-09-05/fit050-v1 --approved-protocol-digest DIGEST`.
The output must not already exist. Use the pinned Python3.12/torch2.9.0+cu128 environment on
RTX3050, one torch CPU thread; all full configs and package versions are retained in the bundle.
Parent logging is every25steps, terminal750; Adam32 is all-parameter inherited-moment continuation.
Atomic CUDA proposals are independent executions, not promised bit-identical directions.
Perceptual curves contain actual parent/selected endpoints; ray trial and native Adam/parent
histories have their own explicit axes and are not fabricated dense perceptual trajectories.
Aggregation is descriptive over the four image-level seed means; retain every image-seed pair.
No p-value, population confidence interval, held-out generalization or default selection is
authorized at this sample size. Each predeclared method is reported, not an outcome-selected winner.

The initial prospective source audit requested stronger artifact/gate binding after synthetic
fault injection; no formal outcomes were accessed. Rehashed request/config/history/field/metric,
optimizer-state and decision corruption tests precede the corrected executable freeze.

### Protocol review

#### Reviewer
codex-code-research-reviewer

#### Verdict
Approved

#### Protocol digest
edad3bb041fb2e34697a101f729402bf12470c411a5abe2567a91be04a027153

#### Digest scope
Canonical JSON of PROTOCOL and SHA256 of every SOURCES entry in `scripts/experiments/fit050_color_ray.py`,
computed by `benchmarks.hier_research_report.protocol_digest`; the reproduction command above
prints the exact source inventory and digest without executing image outcomes.

#### Outcomes accessed
No

#### Review focus
Distinct reviewer `/root/overnight_protocol_reviewer` independently recomputed the digest
before and after CPU verification (154 passed, 2 skipped, 23 deselected). Controls, exact matrix,
configuration/work budgets, numerical transactions, raw-artifact gate reconstruction, provenance,
rollback, A/A and resource qualifications were checked. Previously identified source-integrity
gaps are resolved. Approval is prospective only: complete clean-source report validation and
independent outcome audit are required before any result is promoted. No default, novelty,
held-out or unrestricted pipeline claim is approved.

## Notes

Portfolio: `docs/research/2026-09-05-code-driven-portfolio.md`. Code-derived practical hypotheses,
not claims that preconditioning or line search is new. Future repeated/alternating rays require
a separate prospectively approved test.
