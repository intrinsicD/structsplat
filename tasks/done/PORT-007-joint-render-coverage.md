# PORT-007 — Reuse same-render coverage and shared tail statistics

## Context
`safe_schedule.evaluate_quality` renders RGB, then rebuilds raw Gaussian coverage in torch.
The owned CUDA forward already returns that denominator, but its Python interface discards it.
This opportunity is local common-subexpression reuse, not a cache across moving geometry.
The same evaluator separately selects the largest 1% of errors for CVaR and sorts all errors
for p99; a single sorted tail can serve both consumers on the pinned torch implementation.

## Goal
Remove redundant coverage/order-statistic work while preserving complete quality/gate semantics, and measure
both full quality-evaluation cost and a bounded complete conversion workload.

## Non-goals
- No maintained default, renderer equation, training backward, cached geometry or count change.
- No bit-exact CUDA trajectory promise or isolated speed claim under foreign compute activity.

## Acceptance criteria
- [x] No-grad joint RGB/coverage API, explicit safe fallback, and CPU/CUDA reference tests.
- [x] All quality fields/rejection reasons and near-threshold/outside cases are tested.
- [x] Frozen distinct approval, clean immutable full reports, whole-operation timings and A/A.
- [x] Independent review, synchronized docs/ARA and full verification.

## Interfaces touched
`cuda_render.py`, `safe_schedule.py`, `config.py`, `pipeline.py`, focused tests and bounded driver.

## Depends on
ADR-0003, ADR-0011, ADR-0025

## Agent workflow
- Driver: codex-root
- Reviewer: codex-code-research-reviewer
- Turn: none
- Reviewed revision: e2bf6ae6e06ca8050d9aa8a93d713679a0c9c150

### Handoff log
Code opportunity identified; implementation and exact executable approval pending.

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
All110 cells in results/code-driven-2026-09-05/port007-v1 complete from clean source
e2bf6ae6e06ca8050d9aa8a93d713679a0c9c150. Original checker and independent CPU/GPU/raw/work audits pass.
See ara/evidence/code-driven-method-research-2026-09-05/run.md and ARA C74.
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
findings and retirement, not default promotion. All frozen primary component/pipeline promotion gates fail. Component timing is descriptive; legacy null-gain and pipeline A/A instability prevent a causal end-to-end reuse claim.
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
Expose owned untiled CUDA forward's existing(out,raw_den) only under no_grad with constant RGB.
Keep reference behavior for other devices/dtypes/renderers/affine fields, nonfinite raw coverage,
coverage close to the fixed tau, or nonzero coverage outside the mask. Near-threshold fallback
is conservative empirical engineering, not a proved all-fields floating-point error bound.
No persistent state, hidden cache or cross-call validity protocol is introduced.
The independent tail axis keeps the CVaR mean of exactly ceil(0.01*N) largest values and derives
p99 from the two ascending-rank neighbours, using input-dtype q/rank and torch.lerp_ as in
PyTorch2.9.0. Nonfinite errors and N>2**24 fall back to the original wrapper, retaining its
nearest-rank large-input contract. Both axes default to reference.

Same-state assay: four exposed COCO images IDs9/25/30/34 at max-side512, N2000 quadtree_wse,
seeds0/1, initial and 750-step terminal normalized L2 Adam fields, reused from the complete
hash-bound FIT-050 parent bundle. Compare legacyA, legacyB, joint coverage, shared tail, and both
over ten counterbalanced fresh measurements (five cyclic orders and their reversals); record all quality fields, per-pixel
coverage classifications, full safe-commit rejection-reason vectors, timing and allocated peak.
Keep both null and useful changed candidate states in the gate comparisons; numerical tolerance
cannot replace exact discrete coverage/reason agreement. Synthetic near-tau, outside support,
active-prefix, empty/invalid and dtype fallback controls precede formal outcomes.

Bounded complete pipeline assay: COCO9 full-frame and COCO25 with a fixed synthetic ellipse mask,
max-side512; seeds0/1/2; the same five modes (30 complete pipeline cells). Capacity1000, step_scale0.025,
block_steps25 and Pareto checkpoints25, all other PipelineConfig choices identical. The synthetic
mask exercises containment but is not a semantic segmentation or masked-quality generalization
dataset. Record actual counts, every selected/rejected transaction, complete runtime and final
quality; retain the full pipeline with all scheduled stages. This is a scaled development
workload, not evidence at the default11000-row/full-step regime.
All observer snapshot costs are charged in the instrumented-pipeline total; perceptual scoring
of those retained snapshots occurs offline. Native attempted-step, acceptance and rejection
telemetry is retained even where the observer does not expose a rejected field.
The both-axis arm is primary against legacyA; single-axis comparisons are predeclared explanatory
tests, not a post-hoc fastest-arm selector. All raw repetitions are retained in 80 aggregate
same-state cells plus the 30 pipeline cells (110 total), with image as aggregation unit.

Same-state correctness requires all prescribed cases complete, raw render max difference<=2e-5,
coverage and acceptance/reason vectors identical, and finite metrics. A component speed finding
requires >=1.1x median per-image quality-evaluation ratio with no image slower by>5%; a pipeline
speed finding separately requires >=1.05x median paired total-time ratio, no image slower by>5%,
no PSNR loss>0.05dB, MS-SSIM loss>0.001 or LPIPS increase>0.002, and no worse outside containment.
Compare candidate divergence with the independent A/A arm; count/decision divergence prevents
claiming execution-equivalent pipeline acceleration even if aggregate quality is similar.
Foreign GPU activity or unstable paired timings prevents isolated speed promotion. Preserve
point samples, errors and complete work scope; never reinterpret a failed gate after outcomes.
Exact source hashes/digest, counterbalancing and commands will be frozen before formal launch.

## Executable protocol and reproduction
`benchmarks/port007_controls.py` owns the complete PROTOCOL and SOURCES inventory; the bounded
driver is `scripts/experiments/port007_quality_reuse.py`. The digest binds canonical JSON of
PROTOCOL plus SHA256 for each source, including the complete package Python inventory, CUDA,
both task drivers, controls and the shared artifact validator. Print it without any outcomes:
`python scripts/experiments/port007_quality_reuse.py --protocol-only`.

Formal command after exact distinct approval and the clean verified source commit:
`python scripts/experiments/port007_quality_reuse.py results/code-driven-2026-09-05/port007-v1 --parent-bundle results/code-driven-2026-09-05/fit050-v1 --approved-protocol-digest DIGEST`.
The FIT parent bundle must be complete, gate-valid, and from that same source commit. Copy all
eight prescribed initial/terminal parents, preserving manifest/source/config/state/input hashes;
do not select parents by outcome. A new empty output location is mandatory.
Use Python3.12/torch2.9.0+cu128, RTX3050, one torch CPU thread. Warmups are procedural and charged
outside measured cells as explicitly described by PROTOCOL. Worker errors/timeouts are preserved.

Same-state A/A timing must pass [0.9,1.1] for each of the16 image/seed/state medians of ten
paired ratios, not just low within-arm CV<=0.25. Pipeline A/A timing must pass the same interval
for all six image/seed pairs. Both baseline repeats and candidates must have finite full quality
vectors and actual final reference-image error<=2e-5. A/A/candidate discrete trajectory parity
and every quality guard remain required independently. Image aggregation is descriptive;
publish every paired value, with no population CI or significance claim on four/two images.

The initial prospective source audit used synthetic fault injection to require replay/finite
gates, exact identity/budget validation and raw artifact recomputation. The report gate reconstructs
same-state parity/reasons from all ten raw arrays/records, checks full pipeline native trajectories,
mask and replay arrays, and independently recomputes summary/timing/provenance bindings.
No formal outcome was available when these requirements were frozen.
Portable CPU checks bind objective, foreground/boundary/tail and coverage scalars to retained
float32 arrays with rtol5e-6/atol1e-8, allowing CPU/GPU reduction differences. This is only an
artifact-consistency tolerance; original CUDA metric vectors and exact discrete gate reasons
remain authoritative and are never replaced by a recomputed CPU gate. Final raw reference
denominators are retained, and derived fit/schedule configs, cumulative event counters, and
the final selected snapshot must agree with the frozen pipeline and final field.

### Protocol review

#### Reviewer
codex-code-research-reviewer

#### Verdict
Approved

#### Protocol digest
730b7704c78e61de6c8d213cc2a013b455516fc73d668b4899564b7b1dea507c

#### Digest scope
Canonical JSON of PROTOCOL and SHA256 of every SOURCES entry in `benchmarks/port007_controls.py`,
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

Portfolio: `docs/research/2026-09-05-code-driven-portfolio.md`. Common-subexpression elimination
is known; practical value must be measured. A true CUDA atomic error bound remains out of scope.
