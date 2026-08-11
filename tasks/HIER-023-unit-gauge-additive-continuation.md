# HIER-023 — Unit-gauge normalized-to-additive continuation

## Context

HIER-022 proves that a learned-mass continuation can reach finite, cold-replayable, mass-free
direct-additive endpoints, but rejects its mechanism. Explicit coverage weight `0.05` reduces
`(D-1)^2` by 97.3% while losing `0.454 dB` to ordinary additive fitting and worsening LPIPS and
local maxima. Even the no-coverage arm loses `0.246 dB`. Trajectory telemetry identifies a design
confound: independent numerator and mass variables do not reproduce ordinary normalized fitting
during the nominal `lambda=1` hold, and only 15% of training uses the true additive equation.

The next cheapest causal test removes the learned gauge. With unit kernel mass,

```text
A(x) = sum_i c_i G_i(x)
D(x) = sum_i G_i(x)
I_lambda(x) = A(x) / (lambda * (D(x) + eps) + (1 - lambda))
```

`lambda=1` is exactly the ordinary opacity-free normalized equation and `lambda=0` is exactly the
ordinary direct-additive equation. No auxiliary variable or coverage objective remains.

## Goal

Implement and diagnostically test whether exact ordinary normalized pretraining followed by a
short unit-gauge anneal and a long exact-additive tail yields a pure Gaussian sum that matches or
improves cold additive fitting at the same count and attempted-step horizon.

## Method contract

- The schedule is fixed at 35% ordinary normalized hold, 15% unit-gauge cosine anneal, and 50%
  exact direct-additive tail. For 500 steps these lengths are 175/75/250. The final anneal step is
  strictly positive (`progress=k/(anneal_steps+1)`); the first `lambda=0` operation is the first
  dedicated endpoint step.
- Hold steps call the maintained normalized renderer directly, not a reconstructed two-pass
  quotient. Endpoint steps call the maintained additive renderer directly and have no denominator
  dependency. Only intermediate anneal steps compose additive numerator and unit-coverage renders.
- The field has only means, scales, rotations, and constant RGB coefficients. Opacity, learned
  mass, coverage loss, source RGB, exception maps, and residual layers are forbidden.
- Adam groups and learning rates match the ordinary controls. `gauge_locked_no_reset` retains
  moments; `gauge_locked_endpoint_reset` rebuilds Adam once, immediately before the first exact
  endpoint step. No parameters or schedules change at reset.
- Model selection inspects only exact-additive-tail checkpoints and orders them by raw MSE then
  earlier step. Nonfinite states fail closed to the best valid exact endpoint when available.
- Means stay inside the canvas, scales in `[0.35, max(H,W)]`, coefficient magnitude at most 16,
  and all projections/failures are logged. The returned value is an ordinary additive
  `GaussianField`; no optimizer/gauge state is serializable.
- This module remains default-off and does not alter `render_field`, `fit`, conversion, codec,
  semantic selection, or maintained dispatch.

## Phase A — correctness and path identity

Add typed schedule/render/fit fixtures proving:

- exact phase lengths and a positive last anneal lambda;
- bit-identical direct normalized dispatch at `lambda=1` and object-identical numerator dispatch at
  `lambda=0`;
- finite weak/zero support, intended geometry/appearance gradients, bounded projection, and
  deterministic CPU fit;
- exactly one endpoint optimizer reset when requested and none otherwise;
- mass/opacity/auxiliary-free persistence; and
- reference versus owned-CUDA forward/backward parity at normalized, intermediate, and additive
  points within declared tolerances.

## Phase B — frozen development diagnostic

This is a new consumed development selection, not held-out confirmation. Before opening image
pixels, rank the twelve repository DIV2K filenames by
`SHA256("HIER-023-v1:" + filename)` and take the first four. Although distinct from HIER-022's
COCO bank, these repository images appeared in earlier HIER-013/HIER-020 work and therefore cannot
be called unseen or held out.

| rank | file | selection SHA-256 | file SHA-256 |
|---:|---|---|---|
| 1 | `0001.png` | `10083e2041d2c0bc6f03e615d1d0492274e07ad4444db36ab98a0bb7f1598aeb` | `cdb20d7a462744c269d8e197f735c7bc42e7cda367a940a9b7bc27803b1c8619` |
| 2 | `0343.png` | `2a383550912212efa3c76a17623f2e2ed033d2b4197e86ba191d7bcf1c65f899` | `f70f775deb82a5744fae0640b5b095e35374f7228893dead5750a4b9d7ef8781` |
| 3 | `0685.png` | `2a826a2dab4101c14069a055da21800cc3493803493d3abb92d623c80d458528` | `c42e9a8e92f57ed8ebff3ba247c7578aa85b59785021123f673c56d895e63364` |
| 4 | `0534.png` | `3171ef416d1a74d0fa4e69988adb20492a3ab0b860d247a11d397749b62f4c15` | `c605f2a1092cafc85280d618eb55344c58830313dc75b0469a8f7321f11aa4d3` |

Use deterministic LANCZOS max-side 160 rasters, `N=640`, seeds `0,1`, 500 attempted optimizer
steps, exact owned CUDA renderers on the available RTX 4090, and required LPIPS. Every arm clones
the identical `aniso_onedge`/WSE initialization with the 12-pixel feature cap and constant colors.
The loss is L1 + 0.3 SSIM; checkpoint interval is 25; ordinary controls use the existing
best-PSNR/final-count policy.

Frozen arms:

1. `normalized_plain` — ordinary normalized fitting;
2. `additive_plain` — ordinary direct-additive fitting;
3. `gauge_locked_no_reset` — exact hold/anneal/endpoint schedule with continuous Adam state;
4. `gauge_locked_endpoint_reset` — identical path with the one frozen endpoint reset.

Among the two integrity-eligible continuation arms, the predeclared development selector chooses
higher mean exact-endpoint PSNR; a difference at most `0.02 dB` chooses no-reset as the simpler
arm. This selector is applied once to the complete 8-cell ledger and does not authorize tuning.

Every row reports raw/display metrics, SSIM/MS-SSIM/LPIPS, attempted-step PSNR-AUC, displayed pixel
and complete-7x7 maxima, count, coefficient/cancellation and unit-coverage diagnostics, phase and
lambda, reset count/step, renderer calls by equation, fit/total seconds, peak CUDA memory,
cold/repeated parity, fields/configs/histories/curves, native reconstructions/errors/worst crops,
and failed cells. Iteration, renderer-call, and wall-time scope stay separate.

The bounded unit-gauge mechanism passes only if:

- all eight cells of the selected arm finish at exact `lambda=0`, exact `N=640`, finite,
  coefficient maximum `<=16`, cold parity `<=2e-5`, and contain no opacity, mass, denominator,
  optimizer state, or auxiliary RGB payload;
- at step 175, each continuation arm is within `0.05 dB` of the matched ordinary normalized
  trajectory in every image-seed cell, and reset telemetry is exactly zero before the endpoint;
- selected mean terminal PSNR is no worse than `0.05 dB` below `additive_plain` and closes at least
  half of any positive mean `normalized_plain - additive_plain` PSNR gap;
- selected mean LPIPS is at most `additive_plain + 0.002`, no cell regresses LPIPS by more than
  `0.01`, neither mean displayed pixel nor 7x7 maximum regresses by more than `0.005`, and at least
  one of those local means is noninferior to additive;
- selected PSNR-AUC exceeds additive, with renderer-call and wall-time overhead stated rather than
  treated as equal work; and
- native report review finds no lattice, checker, ringing, isolated black hole, wash, or material
  fine-detail blur.

Failure retains the complete bank unchanged. Any successor must map the failure to transition
speed, optimizer state, or basis conditioning and use a new task/output/data selection. No arm or
threshold may be retuned on these images.

Intended command:

```bash
python scripts/experiments/hier023_unit_gauge_continuation.py \
  tests/test_images/DIV2K_train_HR \
  results/hier023_div2k4_s160_n640_i500_s01_diagnostic_2026-08-11 \
  --max-side 160 --budgets 640 --seeds 0 1 --iters 500 \
  --arms normalized_plain additive_plain gauge_locked_no_reset \
    gauge_locked_endpoint_reset --device cuda --lpips
```

## Non-goals

- No maintained default, Field V2 semantic choice, codec/rate claim, hierarchy/topology event,
  source-derived payload, downstream claim, or novelty claim.
- Do not call the development selector independent confirmation or equal-work comparison.
- Do not revisit HIER-022's COCO cells or tune this DIV2K bank after natural execution begins.

## Acceptance criteria

- [x] Typed default-off unit-gauge implementation and focused CPU/CUDA tests satisfy Phase A.
- [x] Filename ranking, source hashes, protocol, selector, and gates are frozen before pixel access.
- [x] All frozen Phase-B attempts execute once into an immutable portable report; failures remain.
- [x] Results receive adversarial audit, native visual review, scoped ARA disposition, and synced
      task/Index/session brief/architecture/additive-design/research documentation.
- [ ] Focused tests and structural gates pass; the full verification result and any inherited
      baseline failures are recorded exactly. Any self-review remains provisional.

## Interfaces touched

One default-off module under `src/structsplat/`, focused tests, one experiment driver, narrow report
schema support, this task, Index/session brief, and results-driven docs/ARA records only.

## Depends on

HIER-022/015, FIT-022, CORE-013, BENCH-002, ADR-0003/0006

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `2d8fd2a12c4f02bd70439d300d7193dd06a1d64b42ba77ecd146875b06f6c13c`

### Handoff log

This is a dirty-source development diagnostic. No formal protocol review or independent outcome
review is claimed; source snapshots and immutable receipts must bind the executed implementation.

### Execution amendment — invalid harness run retained

The first natural command wrote
`results/hier023_div2k4_s160_n640_i500_s01_diagnostic_2026-08-11` and is retained unchanged as an
invalid harness run. All 16 continuation cells completed, but all 16 ordinary-control attempts
failed after fitting because the driver demanded a step-175 record from `fit`'s post-step
checkpoint ledger, whose 25-step cadence records steps `1,26,...,176` rather than 175. The method,
images, arms, schedules, selector, gates, optimization, and metrics are unchanged.

Before rerun, the driver adds a read-only `iteration_observer` that renders the baseline field
exactly after step 175, records that PSNR, and charges the extra renderer call. The complete frozen
matrix will rerun once into the new immutable output
`results/hier023_div2k4_s160_n640_i500_s01_diagnostic_rerun1_2026-08-11`; the invalid output is not
resumed, overwritten, used for selection, or cited as method evidence.

### Handoff

#### Objective

Determine whether a mass-free path that is exactly ordinary normalized before transition and
exactly ordinary additive afterward can retain the normalized quality advantage in one pure sum.

#### Changes

Added the default-off unit-gauge fitter, frozen 35/15/50 schedule, one-shot endpoint-reset ablation,
12 focused CPU/CUDA tests, a hash-bound 32-cell report driver, schema validation, and synchronized
task/docs/ARA records. Hold and endpoint phases dispatch the maintained renderers directly.

#### Evidence

The valid repair bundle at
`results/hier023_div2k4_s160_n640_i500_s01_diagnostic_rerun1_2026-08-11` has manifest
`2d8fd2a12c4f02bd70439d300d7193dd06a1d64b42ba77ecd146875b06f6c13c` and passes the report checker
with `--allow-dirty`. The first incomplete harness bundle remains intact and excluded.

All 16 continuation endpoints pass integrity. Maximum coefficient is `2.7631`, cold parity is
`4.17e-7`, and maximum step-175 PSNR difference from ordinary normalized is `0.0344 dB`. The frozen
selector chooses no-reset; reset is `0.0700 dB` worse. No-reset reaches `29.0524 dB`, just
`0.0326 dB` below additive's `29.0850`, while improving mean MS-SSIM `0.000172`, LPIPS `0.003574`,
pixel maximum `0.004065`, 7x7 maximum `0.007027`, and PSNR-AUC `1.2050`. It uses 620 versus 522.1
fit renderer calls and `1.72x` fit time.

The mechanism gate still fails: normalized is `29.7498 dB`, a `0.6648 dB` advantage over additive,
and the continuation closes none of that positive gap. One `0343` seed has LPIPS `+0.01226` versus
additive, above the `+0.01` guard. Seven of eight selected checkpoints are still improving at step
500; one selects 475.

#### Assumptions

Iteration/count matching is not equal renderer work. Filename selection is new relative to
HIER-022 but all DIV2K files are historically consumed development sources.

#### Uncertainties

The screen is max-side 160, N=640, two seeds, one device, dirty-source, and producer-reviewed. It
does not determine whether fixed-geometry optimal additive coefficients, basis redesign, higher
count, or longer equal-work training can retain normalized quality.

#### Review focus

Direct endpoint dispatch, the strictly positive anneal boundary, exact reset timing, step-175
observer accounting, exclusion of the invalid harness run, the frozen selector, and the difference
between matching additive and retaining normalized representation efficiency.

#### Protected actions not taken

No in-place tuning, result overwrite, maintained default/semantic/codec change, representation-
limit or novelty claim, unrelated baseline repair, commit, or push.

#### Recommended next action

On a new data selection, apply the existing safeguarded all-row additive coefficient solve to both
plain-additive and unit-gauge geometry. This isolates whether the remaining gap is appearance
optimization or basis geometry before changing topology/count.

### Review

#### Verdict

Provisionally accepted as a negative mechanism result with a positive efficiency signal

#### Self-reviewed

Yes

#### Correctness

Endpoint and schedule tests, deterministic fitting, reset/no-reset telemetry, persistence checks,
and reference/CUDA forward/backward comparisons pass. Every valid report field cold-replays and is
free of opacity, mass, denominator, optimizer, or auxiliary RGB payload.

#### Evidence quality

The repaired matrix is complete, hash-bound, immutable, checker-valid, and multi-metric. Native
full-frame and worst-crop review finds no lattice, checker, ringing, holes, or new wash; all arms
share expected N=640 blur. The invalid first output, dirty source, consumed images, and producer
review are explicit.

#### Simplicity

Unit gauge removes every HIER-022 auxiliary and makes each endpoint a direct maintained dispatch.
No-reset is both simpler and better than resetting Adam on this screen.

#### Missing cases

Optimal fixed-geometry additive RGB, distinct images, larger count/resolution, equal renderer work,
longer convergence, downstream utility, and independent review remain absent.

#### Required changes

None for retaining the near-miss. Do not call it normalized-quality preservation or tune this bank.

#### Optional improvements

Run the frozen fixed-geometry projection factorial; change basis/topology only if the gauge geometry
still loses after both paths receive the same coefficient solve.

## Notes

The reversible fallback is deletion/omission of the new default-off module and driver. The
maintained normalized and additive renderers remain unchanged regardless of outcome.
