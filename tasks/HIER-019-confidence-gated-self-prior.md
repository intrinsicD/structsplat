# HIER-019 — Confidence-gated same-field tail recovery

## Context

HIER-015 established that the direct normalized exact-7k fit is globally and visually much
stronger than the contracted additive hierarchy, but one prospective image failed the literal
worst-pixel comparison. HIER-016 showed why a fixed-geometry RGB tail cannot repair that failure:
the color gradient vanishes with raw support. HIER-017 confirmed that the fixed `1e-8`
normalization floor attenuates nearly uncovered pixels, but fitting with a lower floor moves the
coverage defect and regresses global/local quality. HIER-018 then guaranteed order-one support
with 64 counted broad rows, yet all four fresh images lost raw MSE (3.7--8.5%), every 7x7 maximum
regressed, LPIPS rose, and median fit time rose about 20%. The broad denominator is active
everywhere, so detail colors co-adapt to it rather than receiving a local safety net.

The normalized renderer already exposes a confidence value. With detail numerator `N`, raw
denominator `D`, and floor `eps`, its output is `N/(D+eps)`: equivalently, the normalized color
`N/D` receives confidence `D/(D+eps)` and the missing mass `eps/(D+eps)` is implicitly black.
For `D < eps` only, replace that black prior with a second normalized render of the *same 7,000
rows* at exactly twice their fitted standard deviations:

```text
C0 = N / (D + eps)
P2 = normalized_render(same field, scales * 2, eps)
C1 = C0 + 1[D < eps] * eps / (D + eps) * P2
```

This adds no row or learned coefficient, leaves every supported pixel bit-exact, and is a
confidence-gated normalized-convolution prior rather than another globally mixed background.
The source image is available to the encoder, so it may store one candidate-mode decision only
when a cold metric transaction proves non-regression. Exploratory attribution on the already
consumed HIER-015--018 fields fixed the threshold, one-octave scale, and gate before the fresh
bank below; those numbers are development choices, not evidence about the new sources.

## Goal

Determine prospectively whether fixed confidence-gated same-field recovery removes normalized
exact-7k support-tail counterexamples while preserving all raw, structural, perceptual, local,
count, persistence, and visual properties; if so, replay the immutable rule on every consumed
COCO bank in this lineage and all 16 repository `tests/test_images` sources.

## Non-goals

- Do not change or refit the baseline field, normalization epsilon, Gaussian count/attributes,
  objective, optimizer, 750-step horizon, initializer, topology, or production default.
- Do not tune the `D < 1e-8` activation threshold, scale multiplier 2, missing-mass formula,
  transaction tolerances, or image bank after outcomes.
- Do not describe the wider render as extra representational capacity: it reuses identical stored
  rows/colors and costs decoder work plus a future mode bit, which must be reported honestly.
- Do not claim a complete codec result until selected render semantics are versioned in a
  self-described stream, or a held-out/general result from dirty one-seed diagnostics.

## Prospective bank

Before any selected pixels were opened, all available COCO `train2014` basenames not referenced
in repository text were sorted by `SHA256("HIER-019-v1:" + basename)` and the first four bound:

| source | selection SHA-256 | file SHA-256 |
|---|---|---|
| `COCO_train2014_000000489983.jpg` | `0000b25ae4eee1321e1371aee79521a5579847c8135de72feb81b20417c86a2f` | `be167d03370237f18b6121d760c20e7c6ac42269d842165e868174f7299c8bf1` |
| `COCO_train2014_000000568599.jpg` | `0000ca76903c8227e2a2e6d8b994f627c71ab3dd18556e8b61a5d2285e784a0f` | `1510b72c75a7eb0f7583be727bd64312382b8a05825a6b3dd09503923c90fa2e` |
| `COCO_train2014_000000078213.jpg` | `00018c7ff0b18668d93f606ebe0f7186687d8d7d212d7a2a55741e07795d3947` | `fe89777b6b42252a44108d7491f6104872fd667941ae9cf1258c67b8c065fb5a` |
| `COCO_train2014_000000564341.jpg` | `00018fbe03e7a72c12514ebc9ef3b04b1c0ed4e7fb6bb7587df31ce7e32f68ae` | `b4808c36dbbb8ddd835b052be392a7659a672277a262f962196bfa6b372bb209` |

All use deterministic Pillow LANCZOS maximum-side-512 rasters, full-frame masks, exact N=7,000,
seed 0, required LPIPS, immutable output directories, and the HIER-015 direct-fit configuration
with `normalization_eps=1e-8`.

## Phase A — fresh development screen

Run one HIER-005 control and one direct normalized fit per source. From the persisted direct field,
record two cold render interpretations:

1. `direct_no_recovery` — unchanged `1e-8` normalized render;
2. `direct_self_prior2` — the exact formula above, with a strictly `<1e-8` mask derived from the
   persisted field's raw denominator and a detached same-row twice-scale prior.

The candidate is selected independently per image only if it is finite, exact-count, changes no
field byte/attribute, is exactly equal to baseline outside the activation mask, has cold/repeated
parity `<=2e-5`, and versus baseline has raw MSE ratio `<=1+1e-8`, displayed pixel/complete-7x7
max deltas `<=1e-12`, MS-SSIM delta `>=-1e-7`, and LPIPS delta `<=1e-7`. Otherwise that image's
selected output is exactly `direct_no_recovery`. Among feasible outputs, select the candidate only
if it strictly improves at least one of raw MSE, displayed pixel maximum, or displayed 7x7 maximum
beyond those same tolerances; an exact/non-material tie selects the cheaper baseline. This
target-known transaction chooses one global render mode; it stores no spatial selector.

The selected portfolio passes Phase A only if all four image cells are complete and exact, every
output is finite/parity-clean, every selected result satisfies the baseline transaction, and
versus HIER-005 every image gains at least 2 dB without worsening displayed pixel or 7x7 maxima,
while mean MS-SSIM/LPIPS are noninferior. Median full pipeline time must be at most 1.25x baseline,
candidate cold-render time at most 5x baseline cold-render time, and all full frames/difference
maps/worst crops must be free of isolated speckles, seams at the confidence threshold, ringing,
checker/lattice structure, color wash, or new blur. Report activation count, coordinates,
denominator/missing-mass/error quantiles, and per-pixel candidate deltas. If the bank contains a
baseline local failure, the selected output must repair it; otherwise lack of a fresh failure is
reported rather than counted as mechanism confirmation.

Any numeric or visual failure stops the task without replay or retuning.

## Phase B — frozen consumed replays

Only after Phase A passes, replay the identical candidate and transaction on persisted direct
fields from HIER-015, HIER-016, HIER-017, and HIER-018 (16 consumed COCO images). Require complete
baseline/candidate/selected rows, exact same field hashes, and no selected regression on any raw,
structural, perceptual, pixel, or patch metric. The HIER-015 offending image and HIER-016 known
low-coverage image must no longer lose their recorded HIER-005 local comparisons. No consumed
outcome may alter the formula or gate.

Then run HIER-005, direct baseline, candidate, and selected interpretation once on all 16 sources
bound by HIER-013 under `tests/test_images`. Require every selected direct output to gain at least
2 dB versus HIER-005 and be noninferior in raw MSE, MS-SSIM, LPIPS, displayed pixel maximum, and
7x7 maximum; require no visible lattice or tail speckles. Any counterexample rejects the bounded
"works everywhere" statement and is retained in full.

## Acceptance criteria

- [ ] A default-off typed same-field recovery helper proves zero added rows/parameters, exact
      high-confidence identity, finite zero-support filling, field immutability, and CPU/CUDA
      parity through focused tests.
- [ ] The source-bound driver records complete field hashes, candidate/selection metrics,
      confidence telemetry, exact work, persistence, and portable visuals with isolated failures.
- [ ] Phase A is executed once; only a numeric and visually eligible fixed rule accesses Phase B.
- [ ] Every consumed/test-image counterexample is retained without feedback into tuning.
- [ ] Results receive adversarial audit, ARA/task/docs synchronization, bundle checks, focused
      and full verification; codec/default integration remains a separately reviewed task.

## Interfaces touched

One default-off helper under `src/structsplat/`, one experiment driver, focused tests, report
schema registration, architecture documentation, ARA records, this task, the Index, and generated
session brief. Maintained renderer, codec, CLI, and pipeline defaults remain unchanged.

## Depends on

HIER-018/017/016/015/005, ADR-0003/0006, CORE-013, BENCH-002

## Reversible fallback

Removing the helper/driver restores the unchanged normalized renderer and exact direct field. A
failed metric transaction selects the byte-identical baseline interpretation.

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `28c00c02df95857d5d5e773369afa7fd461a5de9ed1ed7dec0fcf3b2fc47e4b9`

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`. This remains a
dirty-source diagnostic without formal prospective review.

### Handoff

#### Objective

Test a zero-row twice-scale same-field prior only at normalized low-confidence sites.

#### Changes

Added default-off confidence recovery, exact identity/parity tests, synchronized telemetry, and a
four-image candidate/transaction driver.

#### Evidence

`results/hier019_coco_confidence_tail_2026-08-10` has manifest
`28c00c02df95857d5d5e773369afa7fd461a5de9ed1ed7dec0fcf3b2fc47e4b9`. The one useful proposal
repairs raw/local metrics but raises LPIPS `0.000855`; ordinary mode remains selected.

#### Assumptions

The target-known whole-image transaction, not proposal quality alone, defines the returned mode.

#### Uncertainties

One seed/device, four development images, and no distinct review; the original cold timer is not a
quality authority.

#### Review focus

High-confidence identity, zero added capacity, source-free decode, perceptual rollback, and the
no-replay stop.

#### Protected actions not taken

No field mutation, maintained renderer/codec/default, replay, commit, or push changed.

#### Recommended next action

Retain the negative candidate and test explicit pointwise-safe coordinates on a new bank.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

The same-field prior adds no rows, is identity on high-confidence sites, decodes without a source,
and uses a whole-image fail-closed transaction. The only useful proposal raises LPIPS and is
correctly rolled back.

#### Evidence quality

The four-image bundle, parity/identity tests, synchronized metrics, and manifest support the
negative decision. It remains a one-seed/device producer-reviewed diagnostic.

#### Simplicity

The candidate is a default-off interpretation of one field and does not mutate the stored field or
maintained renderer.

#### Missing cases

No consumed replay or broader prior sweep was run after the perceptual gate failed.

#### Required changes

None for retaining the negative candidate. Independent scientific review is outstanding.

#### Optional improvements

Keep the LPIPS counterexample as a regression case for future spatially coherent recovery.
