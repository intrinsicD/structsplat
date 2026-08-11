# HIER-017 — Normalization-epsilon coverage floor

## Context

HIER-015 shows that the direct normalized exact-7k fitter is globally/perceptually strong and
visually clean but can lose a literal worst-pixel comparison to blurry HIER-005.  HIER-016 tests
whether this is a sparse color-objective failure.  Its complete 16-cell prospective screen is
negative: the 1% color-tail arm always returns step zero, while the 0.1% arm reduces raw SSE by
`0.1705%` on the hard image without changing its displayed maximum.

Post-outcome mechanism inspection localizes that maximum to right-edge pixel `(511,100)`.  The
source and nearby learned colors are approximately `0.9`, yet the accumulated Gaussian weight is
only `4.03e-11`.  The normalized renderer computes `num / (den + 1e-8)`, so its fixed epsilon
attenuates a mathematically white weighted average to `0.004` (black).  A diagnostic rerender at
epsilon `1e-12` changes the pixel to `0.878`, reduces the image maximum from `0.9255` to `0.7020`
(below HIER-005's `0.8000`), remains finite, and slightly improves PSNR.  Epsilon zero creates 12
non-finite channel values, so removing the guard is invalid.  A support-radius ladder repairs the
edge but loses about `0.28 dB` and moves the maximum elsewhere; it is not advanced.

## Goal

Determine prospectively whether a positive `1e-12` normalized-renderer denominator epsilon removes
exact-7k low-coverage black attenuation across diverse images without numerical, global,
perceptual, local-patch, parity, or visual regressions, and whether fitting under the same epsilon
is safer than decode-only reinterpretation of an `1e-8` field.

## Non-goals

- Do not tune epsilon on the new sources, revisit the `1e-10/1e-12/1e-16/0` exposed ladder, or
  select support radius, coverage regularization, loss, initialization, topology, or row count.
- Do not make `1e-12` the default, silently reinterpret existing fields, change additive/gsplat
  semantics, or claim that epsilon fixes genuinely zero-support pixels.
- Do not call dirty one-seed development or consumed replays confirmation/held-out evidence.
- Do not claim novelty for positive denominator floors, safe division, or coverage diagnostics.

## Diagnostic status and prospective bank

Before any selected pixels were opened, all available unreferenced COCO `train2014` basenames were
sorted by `SHA256("HIER-017-v1:" + basename)` and the first four were bound:

| source | selection SHA-256 | file SHA-256 |
|---|---|---|
| `COCO_train2014_000000206968.jpg` | `00009784c5ab7287963bf1eced12bd8ab0b3a59a1a2592a4977b9b2553429bb5` | `a0b28389459001bce14be52f2c37195f308909ff75b8ae21ec357d1c05857ac6` |
| `COCO_train2014_000000265833.jpg` | `00034eef72fa7e4009010071f92bbb8331f81f63a1a967226392c7e5896bcfe2` | `52adab433db8427ef58d526e9a766fd5826336f967d81b2000ba9ca9b17ecb7f` |
| `COCO_train2014_000000048658.jpg` | `000397c192d0cd0a303d05f72f43313b588546457fd02f938a18c1e6318af891` | `5ba9a6424106155ce4903e859e5af9d8a7e03bb67490dec38e327b850143b3f4` |
| `COCO_train2014_000000170371.jpg` | `00080f4ccc21f72043039d698a27e8abbf37ae623e848c8b41ed64a25e632bfd` | `29b1ea26c79133987846e8e4930eec2c470e5cf639227cf616c9970a65d1b108` |

All use deterministic Pillow LANCZOS maximum-side-512 rasters, full-frame masks, exact N=7,000,
seed 0, required LPIPS, and immutable output directories.

## Phase A — typed epsilon semantics

Add `FitConfig.normalization_eps`, default `1e-8`, requiring a finite positive value.  Thread it
through normalized reference/CUDA/tiled forward and backward paths, fixed-geometry normalized color
operators, responsibility attribution, and fit rendering.  Additive and gsplat equations retain
their behavior.  The default must remain parity-compatible.

Correctness tests cover analytic weak-support attenuation, zero-denominator finite black output,
default compatibility, CPU/CUDA forward and gradient parity at `1e-12`, validation, color-solve
operator agreement, and persistence/reporting of the exact epsilon.  No renderer may silently use
a different epsilon during fitting, checkpoint scoring, cold render, or metrics.

## Phase B — frozen prospective screen

Run four arms:

1. `h005_control` — exact HIER-015/016 HIER-005 control, once per source;
2. `direct_eps1e8` — exact HIER-015 direct fit and renderer;
3. `decode_eps1e12` — the exact fitted `direct_eps1e8` field cold-rendered at `1e-12`, isolating
   interpretation from trajectory;
4. `fit_eps1e12` — a separate 750-step fit from the same deterministic 7,000-row initialization,
   with `1e-12` used consistently in every forward/backward/checkpoint/cold render.

Both direct fits retain `aniso_onedge`/WSE seed 0, feature cap 38.4, no topology, L1+0.3 SSIM,
`cuda`, and same-final-count best-PSNR checkpointing.  Record standard HIER-016 telemetry plus raw
denominator min/quantiles, exact zero count, counts below each `1e-12` and `1e-8`, attenuation
quantiles, error stratified by denominator band, field/init hashes, and epsilon-sensitive pixel
coordinates/colors.

`fit_eps1e12` passes only if all four cells are complete, exact-count, finite, cold/repeated parity
`<=2e-5`, and use the recorded epsilon consistently; every image gains at least 2 dB versus HIER-005
and does not worsen its exact displayed worst pixel or complete-7x7 maximum; mean MS-SSIM/LPIPS are
noninferior to HIER-005.  Relative to `direct_eps1e8`, no image may worsen raw MSE, displayed pixel
maximum, or 7x7 maximum beyond `1e-8` relative / `1e-12` absolute tolerances; mean MS-SSIM may fall
by at most `0.001`, mean LPIPS may rise by at most `0.002`; at least one image must reduce an
epsilon-sensitive pixel error by one 8-bit level; every full frame/worst crop must remain clean.

`decode_eps1e12` is attribution only.  Agreement with `fit_eps1e12` supports a render-floor cause;
disagreement says the trajectory must adapt.  It cannot be selected because its saved field lacks
the interpretation used during its fit.  If `fit_eps1e12` fails, do not tune this source bank and
do not access Phase C.

## Phase C — frozen consumed replays

Only after Phase B numeric and visual gates pass, run `direct_eps1e8` and `fit_eps1e12` once on:

- the four HIER-015 sources;
- the four HIER-016 sources, requiring repair of the known `(511,100)` counterexample;
- all 16 HIER-013 `tests/test_images` sources.

Require complete exact-count finite/parity-clean fields, no per-image MSE or displayed local
regression versus epsilon `1e-8`, and clean visuals.  On the first two banks also compare against
their recorded HIER-005 rows and preserve the 2 dB/local gates.  Any counterexample rejects the
bounded “works everywhere” statement and cannot alter epsilon.

## Acceptance criteria

- [ ] Positive configurable normalization epsilon is typed, tested across maintained normalized
      operators, default-compatible, and explicit in artifacts.
- [ ] The source-bound four-arm driver isolates failures and emits complete metric, denominator,
      persistence, work, and visual evidence.
- [ ] Phase B is evaluated exactly once; only a passing, visually clean `fit_eps1e12` can replay.
- [ ] Conditional replays retain all rows/counterexamples without feedback into tuning.
- [ ] Results receive adversarial audit, ARA/task/docs synchronization, bundle checks, focused and
      full verification; no default changes without a later clean distinct-reviewed task.

## Interfaces touched

`src/structsplat/config.py`, `src/structsplat/render.py`, `src/structsplat/fit.py`, CUDA parity tests,
one task driver under `scripts/experiments/`, `docs/architecture.md`, ARA records, this task, the
Index, and generated session brief.

## Depends on

HIER-016/015/005, ADR-0003/0006, PORT-002/003, CORE-013, BENCH-002

## Reversible fallback

The default stays `1e-8`; every new path is selected only by an explicit config.  Removing the
config plumbing and driver restores current behavior.  No stored field or production entrypoint is
reinterpreted automatically.

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `3df26b68ccad11c28c423281525fac7738c0c16c3d0c2a151741e9ab1a9f7fa7`

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`.  This remains a
dirty-source diagnostic without formal prospective review.

### Handoff

#### Objective

Determine whether the normalized-render epsilon causes and safely repairs exact-7k holes.

#### Changes

Threaded validated `normalization_eps` through fit/render/codec/CLI paths with default `1e-8`,
added parity/unit tests, and ran frozen `1e-8` versus `1e-12` attribution/refit arms.

#### Evidence

The final 16-cell bundle at `results/hier017_coco_normalization_epsilon_recovery_v2_2026-08-10`
has manifest `3df26b68ccad11c28c423281525fac7738c0c16c3d0c2a151741e9ab1a9f7fa7`. Lower epsilon repairs
the sensitive pixels but fails five raw/local gates; replay is closed.

#### Assumptions

The exact renderer equation is unchanged except for the explicit epsilon parameter.

#### Uncertainties

One seed/device, four development images, dirty source snapshot, and no distinct review.

#### Review focus

Default preservation, codec/config round trip, renderer parity, and separation of attribution from
consistent refitting.

#### Protected actions not taken

No default epsilon, maintained pipeline policy, consumed replay, commit, or push changed.

#### Recommended next action

Keep `1e-8`; test counted coverage rather than weakening the denominator floor globally.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

The epsilon is validated and threaded consistently through fit, render, codec, and CLI paths while
the `1e-8` default and normalized equation are preserved. Attribution and consistent-refit arms
are kept separate; the lower epsilon fails the frozen safety gates and is not promoted.

#### Evidence quality

The final 16-cell bundle and focused parity/config tests establish mechanism attribution on the
four-image bank. One seed/device, dirty snapshotted source, no replay, and no distinct reviewer
bound the result.

#### Simplicity

One explicit scalar parameter replaces hidden constants without changing ordinary behavior.

#### Missing cases

No broad-resolution or multi-device epsilon sweep was justified after the raw/local regressions.

#### Required changes

None for retaining `1e-8` and recording the negative experiment. Independent review is still
needed for publication use.

#### Optional improvements

Keep epsilon sensitivity as diagnostic telemetry in future coverage experiments.
