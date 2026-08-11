# HIER-018 — Counted broad-background coverage certificate

## Context

HIER-015 established that the direct normalized exact-7k fitter is globally and visually far
stronger than HIER-005, but a literal low-coverage pixel can still lose the local gate. HIER-016's
fixed-geometry color tail cannot repair such a pixel because RGB gradients vanish with coverage.
HIER-017 then localized the mechanism and rejected the obvious renderer-floor change: fitting at
`normalization_eps=1e-12` repairs epsilon-sensitive pixels on all four fresh images, but it also
lets the trajectory allocate even less coverage and moves the extreme elsewhere. Relative to the
unchanged `1e-8` fit, three of four cells worsen raw MSE, three worsen the 7x7 maximum, and two
worsen the pixel maximum; five frozen robustness clauses fail. Decode-only reinterpretation is
also not monotone. The normalized `1e-8` equation therefore stays unchanged.

CORE-009 already provides a counted, default-off coverage mechanism with earlier equal-budget
evidence: `background_fraction=0.05, background_grid=8` reserves at most 64 broad, near-isotropic
rows whose geometry is frozen while colors learn. At N=7,000 this is exactly 64 background plus
6,936 detail rows, not hidden capacity. Their overlapping broad supports should keep every pixel's
raw denominator safely above `1e-8`, preventing the optimizer from relying on accidental
axis-aligned-box corner tails while retaining the established normalized renderer.

## Goal

Determine prospectively whether the existing counted 8x8 broad-background initialization gives
the normalized exact-7k fitter a full-frame coverage certificate and eliminates local
counterexamples without sacrificing raw, structural, perceptual, timing, or visual quality; if
so, replay the frozen recipe across every consumed COCO and `tests/test_images` bank in this
lineage.

## Non-goals

- Do not change the normalized equation, epsilon, row count, renderer, fit objective, 750-step
  horizon, topology policy, detail initializer, scale cap, HIER-005, or production defaults.
- Do not tune background fraction/grid/scale/color, coverage thresholds, loss weights, or a
  renderer fallback on the fresh sources. `frac0.05_grid8` is inherited exactly from CORE-009.
- Do not treat the broad rows as free parameters: all 64 count against N=7,000 and their frozen
  geometry must be verified after fitting and persistence.
- Do not call dirty one-seed development or consumed replay held-out confirmation, and do not
  claim novelty for a coarse background layer, mixture support, or denominator diagnostics.

## Prospective bank

Before any selected pixels were opened, all available COCO `train2014` basenames not referenced
in repository text were sorted by `SHA256("HIER-018-v1:" + basename)` and the first four were
bound:

| source | selection SHA-256 | file SHA-256 |
|---|---|---|
| `COCO_train2014_000000402844.jpg` | `0000071156a02cf7316b9402a234e5f81ea4d719479c28b8b2b80ce8760141d6` | `c9894417161c13b10e9df7c3cca75471d5a0ef5d801036aa5097330296879412` |
| `COCO_train2014_000000210071.jpg` | `00006c4b471f11776ed96529eb99610add61c929598149ade8c22d34c644f2b1` | `609b781d3a3baa8c939f84f777d00fb147f21a1769a087bb7dc26c67ba0c1ba2` |
| `COCO_train2014_000000091348.jpg` | `0000ea0b8b93ed69e58fd3a1f1f2280318999ec7978e24a469e049584c4a260b` | `6a8b9aae88e9b40e73aad18135737c68220b49fd1727f91b5c79a8e8a04c4670` |
| `COCO_train2014_000000165574.jpg` | `0000ebdadce300565df93731db6bb0197eee321e282e322524a3104edf738cb0` | `fbc83f705db6519111056875a4cdc76dd9feee01a3995d8bd10e0f88f4ee4205` |

All use deterministic Pillow LANCZOS maximum-side-512 rasters, full-frame masks, exact N=7,000,
seed 0, required LPIPS, and immutable output directories.

## Phase A — frozen development screen

Run three arms:

1. `h005_control` — exact HIER-015--017 HIER-005 control, once per source;
2. `direct_no_background` — exact HIER-017 `1e-8` direct normalized fit;
3. `direct_bg64_grid8` — the same fit with only `background_fraction=0.05` and
   `background_grid=8`, yielding 64 frozen-geometry background rows and 6,936 ordinary detail
   rows from the same seed and WSE/feature-cap recipe.

Both direct arms retain `aniso_onedge`, WSE seed 0, feature cap 38.4, no topology, L1+0.3 SSIM,
`cuda`, `normalization_eps=1e-8`, 750 steps, and same-final-count best-PSNR checkpointing. Record
the HIER-017 raw/display/perceptual/denominator/persistence/work telemetry plus background/detail
counts, pre/post background geometry hashes and maximum shifts, background coverage contribution,
and error stratified by total/background denominator bands.

`direct_bg64_grid8` passes only if all four cells are complete, finite, exact-count, exactly
64/6,936 background/detail, background geometry bit-exact, cold/repeated parity `<=2e-5`, and
have total raw denominator minimum `>=1e-8`. Every image must gain at least 2 dB versus HIER-005
and not worsen its displayed pixel or complete-7x7 maximum; mean MS-SSIM/LPIPS must be noninferior
to HIER-005. Relative to `direct_no_background`, no image may worsen raw MSE, displayed pixel
maximum, or 7x7 maximum beyond `1e-8` relative / `1e-12` absolute tolerances; mean MS-SSIM may
fall by at most `0.001`, mean LPIPS may rise by at most `0.002`, and median algorithm time may
rise by at most 10%. At least one baseline pixel with denominator `<1e-8` must improve its
displayed RGB RMSE by one 8-bit level. All full frames and worst crops must remain free of lattice,
checker, ringing, broad color wash, or new blur.

If the arm fails, retain every counterexample and do not tune this bank or access Phase B.

## Phase B — frozen consumed replays

Only after Phase A numeric and visual gates pass, run `direct_no_background` and
`direct_bg64_grid8` once on:

- the four HIER-015 sources;
- the four HIER-016 sources, including the known `(511,100)` low-coverage failure;
- the four HIER-017 sources;
- all 16 HIER-013 `tests/test_images` sources.

Require complete exact-count finite/parity-clean fields, the 64-row coverage certificate, frozen
background geometry, no per-image MSE or displayed-local regression versus the no-background
control, and clean visuals. On the COCO banks also preserve the recorded HIER-005 2 dB/local
gates. Any counterexample rejects the bounded “works everywhere” statement and cannot alter the
frozen recipe.

## Acceptance criteria

- [ ] The source-bound three-arm driver records exact counted-background, denominator,
      persistence, work, and visual evidence with isolated failures.
- [ ] Phase A is evaluated exactly once; only a numerically and visually eligible fixed recipe
      may access the consumed banks.
- [ ] Conditional replays retain all rows/counterexamples without feedback into tuning.
- [ ] Results receive adversarial audit, ARA/task/docs synchronization, bundle checks, focused
      and full verification; no default changes without a later clean distinct-reviewed task.

## Interfaces touched

One task driver under `scripts/experiments/`, focused tests, `docs/architecture.md`, ARA records,
this task, the Index, and generated session brief. CORE-009 implementation and maintained defaults
remain unchanged.

## Depends on

HIER-017/016/015/005, CORE-009, ADR-0003/0006, CORE-013, BENCH-002

## Reversible fallback

The background layer remains an existing opt-in initializer flag and the normalized default stays
`1e-8`. Removing the driver restores the current pipeline; no production consumer is added.

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `2c1beba2071b4dfe516f6d9aaa36c79552355685e87e045b91d07f3684ffae5c`

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`. This remains a
dirty-source diagnostic without formal prospective review.

### Handoff

#### Objective

Test whether 64 budget-counted broad rows safely provide a normalized coverage certificate.

#### Changes

Added exact background allocation/config plumbing, denominator telemetry, focused tests, and the
three-arm source-bound HIER-018 report.

#### Evidence

`results/hier018_coco_counted_background_2026-08-10` has manifest
`2c1beba2071b4dfe516f6d9aaa36c79552355685e87e045b91d07f3684ffae5c`. Coverage becomes order-one,
but all four MSE rows, every 7x7 maximum, mean LPIPS, and the timing gate regress.

#### Assumptions

The fixed 8x8/64-row allocation is charged inside N=7,000 and is the only experimental axis.

#### Uncertainties

One seed/device and four development images; no distinct review or conditional replay.

#### Review focus

Exact row accounting, background metadata, denominator evidence, and frozen negative decision.

#### Protected actions not taken

No production background/default, renderer change, replay, commit, or push changed.

#### Recommended next action

Retain the negative result and test source-free same-field recovery with global rollback.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

The 64 broad rows are charged inside the exact 7,000-row budget, tagged explicitly, and measured
against the unchanged normalized control. Coverage rises as intended, but every decisive quality
and timing regression is retained and the method is not selected.

#### Evidence quality

The source-bound report records denominator evidence, exact row accounting, metrics, timing, and
artifacts. Its four-image, one-seed/device, producer-reviewed scope supports only the negative
diagnostic.

#### Simplicity

The experiment uses the existing opt-in background mechanism and adds no production consumer.

#### Missing cases

No replay or background-layout sweep was run after the frozen fresh-bank failure.

#### Required changes

None for retaining the negative result. Distinct review remains required for a stronger claim.

#### Optional improvements

Reuse the exact budget-accounting test for future coverage-basis proposals.
