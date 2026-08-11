# HIER-025 — Folded multiscale residual Gaussian sum

## Context

HIER-022 shows that learned coverage mass reaches exact additive endpoints but harms quality.
HIER-023 removes that gauge confound and reaches ordinary additive efficiently, yet retains none of
normalized rendering's fixed-count advantage. HIER-024 then gives ordinary-additive and unit-gauge
geometries the same safeguarded all-row RGB solve: both improve by `0.1300/0.1719 dB`, but the
projected fields differ by only `0.0105 dB` and remain about `0.54 dB` below normalized. The
remaining causal branch is therefore basis geometry/topology, not coefficient optimization.

LIG's Level-of-Gaussian method fits a coarse low-frequency Gaussian level and then a high-frequency
residual level. Its official implementation renders levels separately and persists residual
scaling metadata. That is useful donor evidence, not this endpoint contract:

- paper: https://arxiv.org/abs/2502.09039
- official implementation: https://github.com/HKU-MedAI/LIG

HIER-018 is also not this test. It inserted 64 frozen broad rows into a normalized field, initialized
both broad and detail colors from the original image, and asked for a coverage certificate at
N=7,000. It did not construct a signed residual basis or fold levels into one direct-additive sum.
FIT-016 changed only the normalized loss-target curriculum and was likewise not a separate basis.

## Goal

Determine whether a counted low-pass/residual Gaussian basis can recover materially more of the
normalized fixed-count advantage while persisting exactly one cold-replayable, denominator-free,
opacity-free additive `GaussianField` at N=640.

## Method contract

The candidate is fixed as `folded_grid16_residual`:

1. Form `B = Up_bilinear(Down_area_2(I))` at the original raster dimensions.
2. Build exactly 16 coarse rows with the maintained deterministic grid initializer on `B`, no
   scale cap, constant RGB, and additive semantics. Fit them to `B` for 100 L2-only updates.
3. Render the selected coarse field once and form the signed residual `R = I - L0`. Build exactly
   624 `aniso_onedge`/WSE detail rows from `R`, with the inherited 12-pixel feature cap and signed
   bilinear colors. Fit the detail field to `R` for 300 L2-only additive updates.
4. Concatenate the selected levels. Mark the 16 coarse rows only long enough to freeze their
   optimized geometry during 100 full-target L1 + 0.3 SSIM additive polish updates; coarse colors
   and all detail parameters remain trainable. Remove the training-only mask from the returned field.
5. Persist/render all 640 rows in one ordinary additive pass. No residual scaler, second level,
   denominator, opacity, mass, optimizer, target, auxiliary RGB, or level metadata may survive.

All three stage lengths, targets, losses, count allocation, scale rules, checkpoint cadence (25),
and geometry-freeze boundary are fixed. Stage-local best-PSNR checkpoints may select only against
their declared target (`B`, `R`, then `I`); every attempted update is charged. The global trajectory
uses read-only full-target observers and separately charges their renders. The baseline uses the
same original target, N=640 initializer family, seed, learning rates, support, additive renderer,
500 updates, and checkpoint cadence.

The HIER-024 projection is reused unchanged after both additive endpoints. Its PCG and target-known
metric transaction remain tolerance `1e-6`, at most 48 iterations, ridge `1e-8`, coefficient limit
16, input-centered start/regularization, explicit frozen base, strict-lower raw MSE, MS-SSIM within
`1e-5`, noninferior LPIPS, and pixel/7x7 maxima within `1e-6`. Projection changes RGB only and may
return the incoming field exactly.

## Phase A — correctness and procedural killing fixtures

Add a typed default-off fitter and focused CPU/CUDA tests proving:

- deterministic factor-two area/bilinear low pass and finite signed residual construction;
- exact 16/624 allocation, stage boundaries, attempted-update and observer accounting;
- concatenated one-pass additive render equals the sum of the two level renders within `2e-5`;
- coarse geometry is exact through joint polish while its RGB can change;
- stripping the training mask is render-exact and leaves no auxiliary payload;
- finite bounded endpoint, persistence/cold parity, gradients, and CPU/CUDA agreement; and
- constant, ramp, edge, blob, and texture fixtures complete without nonfinite values or count loss.

These fixtures validate algebra and implementation only; they do not select a count, schedule,
filter, loss, or threshold and cannot be called natural-image evidence. A failure seals Phase B.

## Phase B — frozen remaining-DIV2K development diagnostic

Exclude all eight HIER-023/024 filenames. Rank the four remaining repository DIV2K filenames by
`SHA256("HIER-025-v1:" + filename)` before opening pixels. All are historically consumed by earlier
repository work; this is a new lineage-local selection, not held-out confirmation.

| rank | file | selection SHA-256 | file SHA-256 |
|---:|---|---|---|
| 1 | `0115.png` | `1d9d1f9fc7d952b603b0ea635477bbaba4865650f501f14a9ed7d65cf9095cfe` | `b08214ed8a205d5ff148eb14541de6117f282350bc3e4fc46d2efa8c848073e1` |
| 2 | `0457.png` | `9071b79f6d8b7cd3f8988fdd5ba475da831141ab173c0d7d885be085666aefd1` | `565bb5b65c50abd4b0715b9318851de400cae1475db9c44a138a3bae275d2a05` |
| 3 | `0229.png` | `c6dd4fde23e17f3572009eaef56ebc1f3adf02ab2b643f628691d6f96a2962e1` | `e985cdadc0861ae47a76ae66a46290b7aa322b4d2596727634b144cb205c2d18` |
| 4 | `0799.png` | `fae7587fcfbbd2cd988a7af0b4bef936cb8dab2bd8631e0108dddec6bca42460` | `ad42d7e2fe2ee15461e6999e7673a1f96b1be791b4be8c01baca26812f5667db` |

Use max-side 160, N=640, seeds 0/1, exact owned CUDA renderers, required LPIPS, no opacity,
three-sigma hard support, no AA dilation, and 256-row render chunks. Frozen arms are:

1. `normalized_plain` — maintained normalized fit, 500 updates;
2. `additive_plain` — maintained additive fit, 500 updates;
3. `additive_projected_safe` — the exact same additive endpoint plus HIER-024 projection;
4. `folded_multiscale_additive` — the 100/300/100 candidate above;
5. `folded_multiscale_projected_safe` — the exact same folded endpoint plus projection.

Rows bind source/selection/initial/final hashes; all stage targets, fields, selections, trajectories,
calls, rendered pixels, time, memory, coefficients, level geometry, payload keys, parity, projection
transactions, raw/display metrics, and native full/error/worst-crop artifacts. Fit work and
projection/selection work remain separate.

The candidate passes only if:

- all eight endpoints are finite exact N=640 one-pass additive fields, coefficient maximum `<=16`,
  cold/internal parity `<=2e-5`, exact 16/624 accounting and coarse-geometry freeze, and contain no
  opacity, mass, denominator, optimizer, target, auxiliary RGB, residual scaler, or level payload;
- every projected arm satisfies all safety clauses or returns its incoming field exactly;
- unprojected folded mean PSNR is at least `0.05 dB` above `additive_plain`;
- projected folded mean PSNR is at least `0.10 dB` above `additive_projected_safe` and closes at
  least half of any positive `normalized_plain - additive_projected_safe` gap;
- candidate mean MS-SSIM/LPIPS/pixel/7x7 are noninferior to projected additive, with no per-cell
  LPIPS regression above `0.01` and no pixel/7x7 regression above `0.005`;
- candidate full-target PSNR-AUC over 500 attempted updates exceeds additive, and all observer,
  stage-selection, projection, and target-known metric work is charged; and
- native review finds no lattice, checker, ringing, hole, wash, color lobe, or material new blur.

If this gate fails, do not tune its count, scale, stage lengths, loss, or residual on these pixels.
The next task must change topology/basis construction under a new output and data binding; likely
successors are progressive residual insertion or local frame-conditioned column exchange, not
another fixed-geometry color solve.

## Phase C — conditionally sealed fresh confirmation

Only if Phase B passes numerically and visually, acquire the official DIV2K validation archive and
run the unchanged five-arm protocol on the first four names from
`SHA256("HIER-025-confirm-v1:" + filename)` over canonical `0801.png`--`0900.png`:

| rank | file | selection SHA-256 |
|---:|---|---|
| 1 | `0895.png` | `0644b064658788ac2695cfa2d57d4c2704d3d5e3173f310daf06262914deb703` |
| 2 | `0860.png` | `082cd6a3d95e3b16ec770c3502325c1fcb6cc890e9791a7a27b61614e028ef4e` |
| 3 | `0898.png` | `0b554a43bfb78b6ebda36539d5d3f2cdd1568a394ac430981ee0ac5d96aaab7c` |
| 4 | `0847.png` | `10494b910838e73fad90d013d95d07dfc4ffd618f6416f819d372d9788c6d096` |

Bind the official archive/source hashes at acquisition without changing selection. The same gates
apply independently; no development threshold or method field may change. Failure is retained as a
counterexample and blocks a general/default claim.

## Non-goals

- No maintained renderer/fitter/pipeline/default/semantic/codec change, adaptive count, opacity,
  persistent residual image, second decode pass, hidden carrier capacity, or publication novelty
  claim.
- No claim that LIG is reproduced; this is a strict single-sum transfer at a very different scale.
- No retuning HIER-018, FIT-016, HIER-022--024, or their consumed banks.

## Acceptance criteria

- [x] The method and focused CPU/CUDA/procedural tests satisfy Phase A.
- [x] Counts, stages, controls, data hashes, metrics, work accounting, and gates freeze before
      Phase-B pixel access.
- [x] The complete frozen Phase-B matrix executes once into an immutable checker-valid report.
- [x] A results/visual audit either authorizes unchanged Phase C or records a negative successor.
- [ ] Tasks/docs/ARA synchronize and focused/structural/full verification outcomes are recorded.

## Interfaces touched

One default-off module under `src/structsplat/`, focused tests, one experiment driver, narrow report
schema, this task/Index/session brief, and results-driven docs/ARA only.

## Depends on

HIER-024/023/018, FIT-016/046, CORE-009/013, BENCH-002, ADR-0003/0006

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `36d255c78ae39c9cfc70e8615df1a9821b58bcdd78a35288e8e0cd4816608dcb`

### Handoff log

Dirty-source consumed-development diagnostic; no formal prospective or independent review claim.

### Notes

The reversible fallback is omission of the module/driver and retention of ordinary additive or
normalized controls. A failed Phase B leaves the fresh confirmation names sealed.

### Diagnostic outcome

The frozen 40-row Phase-B matrix completed once at
`results/hier025_div2k4_s160_n640_i500_s01_diagnostic_2026-08-11`. Its report checker passes and
the immutable manifest SHA-256 is
`36d255c78ae39c9cfc70e8615df1a9821b58bcdd78a35288e8e0cd4816608dcb`.

All endpoint/integrity requirements pass. Every candidate has exactly 16 coarse plus 624 detail
rows, completes all 500 updates, freezes coarse geometry exactly, removes the training mask, and
cold-replays in one additive pass. Maximum fold/endpoint/cold parity is
`4.17e-7/2.38e-7/1.79e-6`; maximum candidate coefficient magnitude is `1.5373`; and no saved field
contains opacity, mass, denominator, optimizer, auxiliary RGB, scaler, residual, or level payload.
All projected transactions select safely or return their incoming field exactly.

The basis gate is decisively negative. Mean PSNR is `33.3654` normalized, `32.4276` additive,
`32.6322` projected additive, `30.8734` folded, and `31.2239` projected folded. The candidate loses
`1.5542 dB` before projection and `1.4083 dB` after the same projection; projection itself gains
`0.3505 dB`, so fixed-geometry RGB optimization does not explain the loss. MS-SSIM, LPIPS, both
local maxima, every per-cell guard, and full-target AUC also fail. Native review finds material
diffuse fine-detail blur on insect contours, skyline/window structure, and aircraft edges, though
no new gross checker, lattice, ringing, hole, wash, or isolated color lobe.

Phase C is not authorized and its four official DIV2K validation files remain unopened. Do not
tune this count split, filter, stage schedule, loss, or residual construction on the consumed
pixels. The result rejects the disconnected 100/300/100 folded basis, not all pure additive
fields or all multiscale topology. Evidence:
`ara/evidence/hier025-folded-multiscale-additive-2026-08-11/run.md`.

### Handoff

#### Objective

Test whether a counted low-pass/residual basis can preserve more of normalized rendering's N=640
advantage while persisting exactly one ordinary pure-additive Gaussian sum.

#### Changes

Added the default-off typed folded fitter, exact factor-two low pass, 16/624 staged construction,
training-only coarse-geometry mask, one-pass fold/parity checks, 14 focused CPU/CUDA/procedural
tests, a hash-bound five-arm driver, report-schema validation, and synchronized result records.

#### Evidence

The 669-file bundle is complete, immutable, source-snapshotted, and checker-valid. Focused folded
plus projection tests pass 18/18. Every endpoint integrity clause passes, while every declared
candidate quality gate fails. Exact metrics, visual diagnosis, hashes, and limitations are in the
evidence note above.

The official `./scripts/verify.sh` reaches 1,909 passes, 26 skips, and nine failures. None touches
HIER-025: one is the inherited infinite rank-deficient affine condition number, one is the
torch-2.7 CUDA-property mismatch, one is the opened-descriptor race, and six lazy-import
subprocesses resolve the unrelated installed `structsplat` because this shell needs
`PYTHONPATH=src`. Targeted HIER-025 tests, Ruff, report validation, and all five structural checks
pass. The final full-gate criterion remains unchecked rather than hiding these failures.

#### Assumptions

All 500 attempted updates are charged equally even though proxy-target updates are not equivalent
to full-target optimization. The read-only full-target observers are separately counted. These
historically consumed images are development evidence, not held-out confirmation.

#### Uncertainties

The screen is max-side 160, N=640, two seeds, one device, dirty-source, and producer-reviewed.
It does not determine the extra additive capacity needed for parity, whether progressive residual
insertion helps, or whether a different primitive can eliminate the fixed-count gap.

#### Review focus

Check exact 16/624 accounting, stage-local targets and checkpoint selection, global observer/call
accounting, coarse-geometry freezing with trainable RGB, removal of the training mask, fold/cold
parity, projection reuse/rollback, and the interpretation that proxy-stage blur—not endpoint
serialization—causes the loss.

#### Protected actions not taken

No in-place retuning, negative-result overwrite, Phase-C access, maintained renderer/fitter/
pipeline/default/semantic/codec change, formal claim, commit, or push.

#### Recommended next action

On a new output and untouched data binding, measure the pure-additive capacity exchange rate and
test progressive residual insertion that preserves a full-target additive base instead of fitting
disconnected levels.

### Review

#### Verdict

Provisionally accepted as a negative basis result

#### Self-reviewed

Yes

#### Correctness

Focused tests, exact counts, deterministic hashes, geometry-mask removal, endpoint payload audit,
renderer parity, and transaction rollback all pass. The report checker independently validates the
complete bundle.

#### Evidence quality

The protocol preceded Phase-B access and retains all arms, seeds, histories, fields, visuals, and
failed gates. Prior image exposure, dirty sources, one GPU, and absent distinct review limit it to
a development diagnostic.

#### Simplicity

The candidate folds to an ordinary four-array `GaussianField`; no decoder branch or hidden level
state survives. Its rejection therefore removes the method without changing maintained behavior.

#### Missing cases

Fresh natural images, larger counts/resolutions, capacity-matched additive fields, progressive
births, matched renderer work/time, complete streams, downstream response, and distinct review.

#### Required changes

None for retaining the negative result. Do not authorize HIER-025 Phase C or tune its bank.

#### Optional improvements

Reuse the sealed validation files only under a new prospectively frozen capacity/topology task;
keep fixed-count and equal-byte conclusions separate.
