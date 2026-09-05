# HIER-035 — Additive convergence controls

## Context
The additive anatomy permits curvature-scaled continuous updates. Test simple controls first.

## Goal
Compare Adam, diagonal curvature scaling, and local Gauss–Newton at fixed count, charging time.

## Non-goals
- No maintained defaults, count growth, official native-method or general image-quality claim.

## Acceptance criteria
- [ ] Checked derivatives; frozen objectives, fixtures, arms, step bounds, and work accounting.
- [ ] Strong Adam controls, fixed terminal convention, all failures preserved.
- [ ] Prospective review, clean immutable run, portable curves and raw artifacts.
- [ ] Independent audit, synchronized evidence and docs, full verification.

## Interfaces touched
Task-scoped optimizer controls, pixel-gradient reference, focused tests and driver.

## Depends on
HIER-033, ADR-0006

## Agent workflow
- Driver: codex-root
- Reviewer: codex-overnight-protocol-reviewer
- Turn: driver
- Reviewed revision: pending

### Handoff log
Design in preparation; no formal outcome exists.

## Notes
Overnight research authorized September 5, 2026.

## Frozen protocol

Executable authority: PROTOCOL in scripts/experiments/hier035_convergence.py, bound together with
every SOURCES file by --print-protocol-digest. Formal command:
python scripts/experiments/hier035_convergence.py results/hier035_convergence_2026-09-05
--approved-protocol-digest EXACT_DIGEST. Run only after the distinct prospective receipt below
is committed in a clean source tree. --smoke uses only translated seed77 and three updates,
retains a dirty source snapshot and is wiring-only.

- Question: do simple exact diagonal or local 8x8 Gauss–Newton updates improve fixed-count
  additive convergence beyond a three-learning-rate Adam envelope? Null: no candidate passes
  the frozen family-level iteration-quality gate.
- Data: four procedural 64x64/N16 families, translated/anisotropic/overlap/texture, seeds0/1/2.
  Generated truth and starting fields are defined completely by the driver. All are development
  mechanisms; no natural image, sealed bank, whole-pipeline or actual-rate inference is allowed.
- Renderer/objective: owned cuda_additive, float32 RTX3050, chunk256, 3sigma C0 fade, constant
  signed RGB, no mask/opacity/filter/affine terms. Raw 0.5 mean squared RGB error. PSNR uses raw
  output and MSE floor1e-12; perceptual metrics use display-clamped output.
- Arms: Adam multipliers0.3/1/3 applied to means/log-scales/rotation/RGB rates0.1/0.03/0.03/0.03,
  betas0.9/0.999, eps1e-8; exact diagonal GN; local block GN. These are local research controls,
  not official native implementations of any published second-order splatting method.
- All arms execute exactly160 updates, terminal state only. Same starting fields and domain
  bounds: means inside canvas, scales[0.35,16], RGB[-2,2]. Curvature trust units are
  (1,1,0.1,0.1,0.1,0.1,0.1,0.1); normalize each row's maximum trust ratio; damping0.01 times
  its maximum scaled diagonal (floor1e-12). Up to six halved global trial steps; loss-increasing
  trials are rejected and the exact previous field is retained if all trials fail. Equal-loss
  trials may be accepted. Adam is not forced monotone.
- Each cell is a fresh process, one torch CPU thread, with two-step Adam/diagonal/block warmups
  on translated seed77. Rotate arm order by seed. Timeout600seconds. Record every attempted
  update, accepted/rejected status, forward/gradient counts and line-search trials. Complete fit
  time includes gradients, curvature, all trial renders, domain projection and scalar trace IO.
- Timing is descriptive on a shared workstation. GPU process snapshots before/after fitting and
  peak allocated VRAM/process RSS are recorded. No speed or isolated-time claim is authorized;
  snapshots cannot establish absence of transient interference. The cache timing assay remains
  separate and must not overlap this driver's GPU work.
- Pair by family/seed. Primary comparator is the highest terminal raw PSNR among all three Adam
  controls in that seed, a deliberately strong oracle envelope. A candidate passes a family only
  if all15 cells are complete/integrity-valid, median three-seed PSNR gain>=0.5dB, no seed loses
  >0.1dB, and every seed has MS-SSIM difference>=-0.005 and LPIPS increase<=0.01 against that
  seed's selected Adam comparator. Report all seeds, median and individual differences; no
  population significance or image-level generalization from three procedural seeds.
- Descriptive matched-time comparison uses each arm's last trace sample at or before the minimum
  final elapsed time across the five arms in the paired seed. It is not a speed verdict.
- Artifacts: source/initial/target bindings, terminal native field, raw float target/reconstruction,
  target/reconstruction/4x error PNGs, iteration/time PSNR curves, full histories/progress JSONL,
  configuration/environment, tidy JSON/JSONL/CSV, decision predicates and portable index.html.
  Cold decoded parameters/count must be exact; cold renderer max error<=2e-5.
- Any timeout/OOM/nonfinite/missing cell remains visible and excludes its entire family from
  positive selection. No selective repeat, threshold adjustment or in-place artifact repair.

### Protocol review

#### Reviewer
codex-overnight-protocol-reviewer

#### Verdict
Approved

#### Protocol digest
9c389e880fefd590005477b23f2194a4a7c1b2ae27460b12a8e9b9cced36f1da

#### Digest scope
Canonical executable PROTOCOL plus all source hashes in the driver's SOURCES: task driver,
control fitter, pixel Jacobian reference, field/render/owned CUDA sources, metrics, portable
report and checker. Independently recomputed before and after review; unchanged.

#### Outcomes accessed
No

#### Review focus
Gradient/Gram normalization, scaled trust/damping, global finite trial acceptance, field
ownership, fixed terminal160-step convention, charged inner work, strongest Adam envelope,
complete family and perceptual gates, source/matrix/configuration/serialized-count/trace/raw
metric contracts. Independent focused verification: 60 tests passed including owned-CUDA
gradient parity. Approval is procedural iteration-quality evidence only: extra curvature
work is not free, shared-GPU times do not establish speed, no default/natural-image claim.
