# HIER-022 — Normalized-to-additive pure-Gaussian continuation

## Context

HIER-008 proves that a sufficiently overlapping direct-additive Gaussian lattice can have a
stable, nearly exact appearance solution, but its reduced fields develop support holes. HIER-013
and HIER-014 show that coefficient-only reconditioning cannot rescue a poorly conditioned fixed
basis, while HIER-015 shows a much stronger direct normalized fit than the contracted additive
line. HIER-017 then isolates normalized compositing's own low-coverage/epsilon failure, and
HIER-021 repairs the bounded tail only with explicit source-RGB patches outside the Gaussian
field.

The research audit in
`docs/research/2026-08-11-pure-gaussian-additive-continuation.md` selects a smaller causal question:
can normalization serve only as an optimization scaffold while the fitted and persisted endpoint
is one strict direct-additive Gaussian sum? The components are known (normalized RBFs,
partition-of-unity constraints, continuation, and stable RBF approximation); no novelty claim is
made. The recipient-specific hypothesis is the measured interaction among coverage conservation,
the normalized-to-additive path, and coefficient stability under StructSplat's finite-support
anisotropic renderer.

## Goal

Implement and diagnostically test a default-off continuation that starts at the current normalized
equation and ends at an exact, cold-replayable direct-additive `GaussianField` with no denominator,
mass payload, source pixels, exception map, or non-Gaussian residual.

## Method contract

For shared RS geometry and peak-one kernels, maintain trainable direct RGB coefficients `p_i` and
positive training-only masses `m_i`:

```text
A(x) = sum_i p_i G_i(x)
D(x) = sum_i m_i G_i(x)
I_lambda(x) = A(x) / (lambda * (D(x) + eps) + (1 - lambda))
```

- `lambda=1` is a normalized weighted sum. Initialization sets `p_i=m_i*c_i`, so it reproduces
  the ordinary initialized normalized colors up to the explicitly logged epsilon/scale effect.
- `lambda=0` is exactly `A(x)`. Only geometry plus `p_i` survives; `m_i` is discarded and cannot
  appear in the saved field, byte ledger, cold render, or result label.
- Training-only masses are positive, finite, bounded in log space, and coverage is encouraged
  toward `D(x)=1`. This makes the intended partition of unity explicit instead of letting the
  denominator hide support defects.
- The frozen schedule holds `lambda=1` for the first 35% of steps, cosine-anneals to zero over the
  next 50%, and spends the final 15% at the exact additive endpoint. Model selection may inspect
  only checkpoints from that final `lambda=0` tail.
- Geometry, coefficient, and mass updates use separate logged Adam groups. Means remain in the
  canvas, scales stay in `[0.35, max(H,W)]`, log masses stay bounded, and nonfinite states fail
  closed to the best earlier exact-additive checkpoint when one exists.
- Reference and owned CUDA additive accumulation are both supported. The continuation composes
  two additive accumulations; it does not add a third renderer equation to the maintained
  `render_field` dispatch and does not change ADR-0003/0006 defaults.
- Telemetry includes attempted steps, lambda, raw loss/PSNR, coverage loss and denominator
  quantiles, coefficient and mass ranges, renderer calls, elapsed time, exact-additive checkpoint
  identity, and cold parity.

## Non-goals

- Do not change `run_pipeline`, `scripts/convert.py`, the normalized default, Field V2 semantic
  selection, a codec, or any maintained dispatch.
- Do not call training-only mass downstream structure or charge it as a transmitted field.
- Do not add topology events, lifting, source-RGB patches, ordered alpha, quantization, entropy
  coding, or an actual-rate claim in this task.
- Do not tune on HIER-015--021's source-bound images or consume any sealed confirmation split.
- Do not describe a dirty or producer-reviewed diagnostic as formal semantic selection,
  confirmation, general superiority, or a publication-ready novelty result.

## Phase A — correctness and mechanism

- Add a typed continuation configuration and a small reference implementation under
  `src/structsplat/`.
- Closed-form fixtures must prove both endpoints, finite weak/zero-support behavior, positive mass,
  intended gradients to geometry/appearance/mass, exact mass discard, CPU determinism, and
  reference/CUDA forward and backward agreement at representative tolerances.
- Five programmatic 48x48 RGB fixtures select one coverage weight from the frozen set
  `{0.01, 0.05, 0.2}`: constant `[0.2,0.5,0.8]`; coordinate ramp
  `[x,y,(x+y)/2]`; a centered vertical two-color step; a four-pixel black/white checker; and
  gray `[0.35,0.35,0.35]` with red, green, and blue singleton pixels at `(12,12)`, `(35,13)`,
  and `(24,35)` in `(x,y)` order. Each cell uses `aniso_onedge`/WSE, `N=128`, seed `0`, the
  12-pixel feature cap, 160 steps, the exact owned CUDA additive renderer, and otherwise the
  Phase-B continuation settings. Selection is by mean exact-additive terminal MSE, subject to
  finite output and coefficient maximum `<=16`; ties choose the smaller weight. The selected
  value is recorded before natural-image execution. These synthetic fixtures are mechanism
  calibration, not image-quality evidence.

## Phase B — frozen development diagnostic

This phase is explicitly consumed development evidence because no distinct prospective reviewer
is available. Use the four repository `tests/test_images` COCO images, deterministic LANCZOS
maximum-side 160 rasters, budgets `640`, seeds `0,1`, 500 attempted optimizer steps, exact owned
CUDA renderers on the available RTX 4090, and required LPIPS. Bind source hashes, repository diff,
GPU/software identity, and all output artifacts. Every arm starts from an identical cloned
`aniso_onedge`/WSE field with the existing 12-pixel feature cap and constant colors.

The frozen source binding is:

| file | SHA-256 |
|---|---|
| `COCO_train2014_000000000009.jpg` | `35cdfe8259aca40d564baf33ee749d82ce852446bd9574f0c47551d8bfffda99` |
| `COCO_train2014_000000000025.jpg` | `d8f12a26d8803701cabac80494b080f998e5ed9bafaf61a2825ce6212c85487a` |
| `COCO_train2014_000000000030.jpg` | `0444b10826d376ad9075805061405f6071a62b80eda29c5f284ed77b093d5b1d` |
| `COCO_train2014_000000000034.jpg` | `2c46871034fa901ae795a8bb916ba7f2f728507cab9e511cced0986bd083d193` |

Arms:

1. `normalized_plain` — current normalized fit, same horizon and terminal-count best-PSNR policy;
2. `additive_plain` — current direct-additive fit, otherwise matched;
3. `continuation_no_coverage` — the exact schedule above with coverage weight zero;
4. `continuation_coverage` — the same schedule with the Phase-A-selected frozen weight.

All rows report raw and displayed PSNR/MSE, SSIM/MS-SSIM/LPIPS, attempted-step PSNR-AUC, displayed
pixel and complete-7x7 maxima, count, coefficient range/cancellation diagnostics, coverage
quantiles, fit and total seconds, renderer calls, peak CUDA memory, cold/repeated parity, complete
field/config/history artifacts, full reconstructions/errors, and worst crops. Failed or missing
cells stay visible. Iteration, renderer-call, and wall-time scope remain separate; two additive
accumulations are not called equal work to one ordinary render.

The bounded mechanism passes only if all eight `continuation_coverage` cells:

- finish at exact `lambda=0`, exact `N=640`, finite, coefficient maximum `<=16`, and cold parity
  `<=2e-5`, with no serialized mass or auxiliary RGB payload;
- have mean terminal PSNR no worse than `0.25 dB` below `normalized_plain` and close at least half
  of any positive `normalized_plain - additive_plain` mean PSNR gap;
- have mean LPIPS no worse than `additive_plain`, no per-image-seed LPIPS regression above `0.01`,
  and no mean displayed pixel/7x7 regression versus `additive_plain`;
- reduce mean `D-1` coverage MSE by at least 25% versus `continuation_no_coverage`; and
- show no lattice, checker, ringing, isolated black hole, wash, or blur in native report images.

If the gate fails, retain the bundle unchanged. A successor is allowed only when telemetry maps
the failure prospectively to one of: coverage target, basis conditioning, or continuation path.
It must use a new task/output/data selection rather than retune this consumed bank.

Intended command:

```bash
python scripts/experiments/hier022_additive_continuation.py \
  tests/test_images results/hier022_coco4_s160_n640_i500_s01_diagnostic_2026-08-11 \
  --max-side 160 --budgets 640 --seeds 0 1 --iters 500 \
  --arms normalized_plain additive_plain continuation_no_coverage continuation_coverage \
  --coverage-weights 0.01 0.05 0.2 --device cuda --lpips
```

## Acceptance criteria

- [x] Typed default-off implementation satisfies the endpoint, gradient, stability, deterministic
      CPU, mass-discard, and owned-CUDA parity fixtures without changing existing defaults.
- [x] Synthetic calibration is reproducible and freezes one coverage weight before Phase B.
- [x] The bounded driver produces a portable immutable report with raw JSON/JSONL/CSV, configs,
      histories, fields, curves, native-resolution visual links, errors, provenance, and a stated
      diagnostic decision.
- [x] Phase B executes once under the frozen rule; negative/missing cells are retained and no
      failed gate is rescued in place.
- [x] Results receive an adversarial audit and appropriately scoped ARA disposition; task, Index,
      session brief, architecture, and additive design documents remain synchronized.
- [ ] Focused tests and `./scripts/verify.sh` pass. Any self-review is explicitly provisional.

## Interfaces touched

`src/structsplat/additive_continuation.py`, focused tests, one driver under
`scripts/experiments/`, narrow report-checker support if required, `docs/architecture.md`,
`docs/additive_field_v2.md`, the research note, ARA records after results, this task,
`tasks/INDEX.md`, and generated `tasks/SESSION-BRIEF.md`.

## Depends on

HIER-008/014/015/017/021, FIT-022, CORE-013, BENCH-002, ADR-0003/0006

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `a334ed4eb21eb2bd635627a8eaeb8f0968905d929397175ace9e6cf6e47ff9fc`

### Handoff log

This task begins as a development diagnostic. No formal `### Protocol review` is claimed without
a distinct outcome-unseen reviewer and a clean committed implementation revision.

### Handoff

#### Objective

Test whether a training-only learned coverage gauge can carry a normalized fit to a persisted,
mass-free, exact direct-additive Gaussian endpoint without losing the normalized quality advantage.

#### Changes

Added a lazy default-off continuation fitter, exact endpoint and gradient/parity fixtures, a frozen
synthetic calibration plus natural-image driver, report-schema validation, and synchronized task,
architecture, design, research, and ARA records. No maintained renderer or pipeline dispatch changed.

#### Evidence

All ten focused tests pass, including owned-CUDA forward/backward parity. Synthetic calibration
selected coverage weight `0.05`. The immutable 32-cell report at
`results/hier022_coco4_s160_n640_i500_s01_diagnostic_2026-08-11` has manifest
`a334ed4eb21eb2bd635627a8eaeb8f0968905d929397175ace9e6cf6e47ff9fc` and passes the report checker
with `--allow-dirty`. All eight coverage endpoints are finite exact `lambda=0`, exact `N=640`,
mass-free fields with maximum cold parity `4.77e-7` and maximum coefficient `2.831`.

The frozen quality gate fails. Mean normalized/plain, additive/plain, no-coverage continuation,
and coverage continuation PSNR are `26.840`, `26.291`, `26.045`, and `25.837 dB`. Coverage reduces
mean `(D-1)^2` from `0.51250` to `0.01405` (97.3%) but trails plain additive by `0.454 dB`, raises
LPIPS from `0.16838` to `0.18038`, worsens pixel/7x7 maxima, and takes `2.25x` the fit time.

#### Assumptions

The comparison is iteration- and count-matched, not equal renderer work. The bundle's source
snapshots, rather than later documentation-only edits, are the executed-source authority.

#### Uncertainties

The bank is consumed development evidence from one GPU and two seeds, with producer visual review
and dirty executed sources. It does not prove an asymptotic representation limit or settle a
different gauge/schedule on disjoint images.

#### Review focus

Endpoint identity, mass discard, schedule/model-selection boundaries, renderer-call accounting,
the separation of coverage success from quality failure, and whether the successor is genuinely
new rather than a retune of this bank.

#### Protected actions not taken

No maintained default, semantic/codec selection, formal novelty claim, mutation of the completed
bundle, push, or unrelated user file changed.

#### Recommended next action

Use a new task and disjoint development selection to test a unit-mass gauge whose start is exactly
the ordinary normalized equation and whose endpoint is exactly additive. Remove learned masses and
coverage loss; give the exact additive tail most of the horizon and isolate Adam-state reset.

### Review

#### Verdict

Provisionally accepted as a negative diagnostic (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

Endpoint fixtures, deterministic fitting, finite-state fallback, mass-free persistence, and
CPU/CUDA forward/backward checks cover the new implementation. Every natural-image candidate
cold-replays at the exact additive endpoint within tolerance.

#### Evidence quality

The frozen cells are complete and immutable, the bundle checker passes, and the failure is
multi-metric: coverage improves dramatically while terminal quality and local guards regress.
The dirty-source, consumed-bank, single-device, and producer-review limitations remain explicit.

#### Simplicity

The method is small and training-only, but independent learned masses create an unnecessary
appearance/mass gauge and double the rendering work. The evidence justifies rejecting that gauge.

#### Missing cases

A trajectory exactly equal to ordinary normalized fitting before annealing, a longer exact-additive
tail, optimizer-state reset at the equation boundary, disjoint images, independent review, and
larger-budget/full-resolution confirmation remain untested here.

#### Required changes

None for retaining the negative result. Do not tune the completed bank or promote this mechanism.

#### Optional improvements

Run the preregistered unit-gauge successor; pursue basis/frame redesign only if that cleaner path
still cannot match cold additive fitting.

## Notes

The reversible fallback is complete removal of the new default-off module/driver. The existing
normalized and additive renderers, fits, fields, codecs, and maintained pipeline remain unchanged.
