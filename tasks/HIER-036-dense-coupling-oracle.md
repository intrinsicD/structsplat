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
- [ ] Bound the dense oracle and test Jacobian, Gram, update, ownership, and work semantics.
- [ ] Freeze executable protocol and obtain distinct prospective digest approval.
- [ ] Clean immutable complete run, native artifacts/curves, and independent results audit.
- [ ] Synchronize task/docs/ARA and pass the full verification gate.

## Interfaces touched
Task-scoped benchmark oracle, experiment driver, focused tests, and portable report validator.

## Depends on
HIER-033/035, ADR-0006

## Agent workflow
- Driver: codex-root
- Reviewer: codex-overnight-protocol-reviewer
- Turn: driver
- Reviewed revision: pending

### Handoff log
Design reviewed independently before implementation. Executable approval and outcomes pending.

## Proposed protocol
Freeze in the driver before prospective source review; this section does not authorize execution.

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

