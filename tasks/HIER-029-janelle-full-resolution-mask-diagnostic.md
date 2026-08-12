# HIER-029 — Janelle full-resolution HIER-028 mask diagnostic

## Context

HIER-028 is a bounded max-side-160 positive for a projected cold additive N=960 base plus 64
deterministic worst-residual rows. Its task explicitly leaves full-resolution and masked behavior
open. The canonical exposed Janelle C0001 source is native 5328x4608; StructSplat's established
full-resolution Janelle evaluation regime is deterministic max-side 1200 (1200x1038). The user
requested a paired visual/metric comparison with and without the available foreground mask.

No distinct prospective reviewer is available. This task is therefore an exposed single-image,
single-seed diagnostic only: it may test scaling and expose failure modes, but cannot confirm
HIER-028, change a default, or promote an ARA claim.

## Goal

Run the complete HIER-028 four-arm ladder on the canonical 1200x1038 Janelle C0001 raster under
paired full-frame and foreground-mask objectives, and produce a portable `index.html` containing
the metrics, native-resolution reconstructions, foreground comparisons, and error maps.

## Frozen diagnostic protocol

- Source: `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg`,
  SHA-256 `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b`,
  native 5328x4608.
- Mask: sibling `mask/mask_C0001.png`, SHA-256
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`,
  threshold 0.5 with nearest-neighbor resizing.
- Raster: Pillow LANCZOS max-side 1200, yielding exactly 1200x1038. Seed 0, RTX/CUDA, LPIPS
  required, 256-row render chunks.
- Arms in each objective mode: normalized N=640; projected cold additive N=960; the exact
  projected N=960 base plus 64 HIER-028 pursuit rows; and projected cold additive N=1024.
- Fit contract: unchanged HIER-028 initializer, feature cap 12 px, L1 + 0.3 SSIM, Adam learning
  rates `5e-2/3e-2/1e-2/3e-2`, 500 updates, best-PSNR/final-count checkpoints every 25,
  sigma cutoff 3, no AA dilation, no support fade.
- `full_frame` uses ordinary initialization and the entire RGB raster for fitting, projection,
  and pursuit.
- `masked_foreground` uses `build_masked_field(..., contain=False)`, mask-weighted L1/SSIM,
  mask-restricted coefficient projection, and mask-restricted residual selection. The mask is
  encoder-only and is absent from the four-array endpoint. This is **not** a mask-containment or
  zero-outside test; CORE-010/011 containment controls remain separate.
- Every row reports its objective-domain metrics plus common full-frame and black-matted
  foreground-crop PSNR, MS-SSIM, LPIPS, pixel maximum, and 7x7 maximum. Full-frame metrics for
  `masked_foreground` are descriptive because that arm does not train the background.
- Preserve source/mask/config/code hashes, fields, fit/projection/pursuit histories, raw and
  foreground reconstructions, error maps, worst crops, timing, memory, and parity receipts.
- No outcome-dependent rerun, changed count/scale, threshold relaxation, or rescue arm.

Exact command:

```bash
PYTHONPATH=src python scripts/experiments/hier029_janelle_mask_diagnostic.py \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png \
  results/hier029_janelle_c0001_s1200_mask_factorial_s0_diagnostic_2026-08-11 \
  --max-side 1200 --seed 0 --device cuda --lpips
```

## Non-goals

- No held-out, multi-image, multi-seed, native-5328, equal-byte, actual-rate, downstream,
  production, novelty, renderer-default, or mask-containment claim.
- No change to the maintained conversion pipeline or HIER-028's frozen historical protocol.

## Acceptance criteria

- [x] Mask-restricted projection and pursuit preserve historical unmasked behavior and have
      focused deterministic tests.
- [x] All eight 2x4 diagnostic cells complete or remain visibly represented as errors.
- [x] The report contains manifest/config plus JSON/JSONL/CSV metrics and portable native-size
      source, mask, reconstruction, foreground, error, crop, field, and history links.
- [x] `python scripts/check_report_bundle.py RESULTS_DIR --allow-dirty` passes.
- [x] Results receive a producer-side adversarial audit and remain scoped as diagnostic.
- [x] Task/Index/session brief/ARA synchronize and `./scripts/verify.sh` passes.

## Interfaces touched

`src/structsplat/residual_pursuit_additive.py`,
`src/structsplat/endpoint_appearance_projection.py`, one driver under `scripts/experiments/`,
focused tests, `scripts/check_report_bundle.py`, this task/Index/session brief, and result-driven
ARA/evidence notes. No maintained public default.

## Depends on

HIER-028/024, CORE-010/011, BENCH-002, ADR-0003/0006/0017/0019

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest
  `b5c772fd2ab972b6169f3e08f720d8369797bf8b4d63ed6ee593b21b023aa52d`

### Handoff log

Producer execution and adversarial outcome audit are complete without a distinct prospective
reviewer. The immutable checker-valid report contains all eight cells, full/foreground metrics,
raw reconstructions, and fixed-scale errors. A first driver attempt passed the full-frame cells but
passed a CUDA tensor across the NumPy mask-geometry boundary; that failed bundle is retained at
`results/hier029_janelle_c0001_s1200_mask_factorial_s0_diagnostic_2026-08-11_failed_driver_mask_tensor`.
The corrected run changed only that driver boundary, reran the preregistered matrix unchanged, and
is the evidence-bearing bundle below. Distinct review remains pending.

### Outcome

HIER-028 does **not** scale as a normalized replacement on this exposed full-resolution Janelle
image. In `full_frame`, pursuit N=1024 reaches `13.64684 dB`, only `+0.00476 dB` over its exact
N=960 base, `-0.35093 dB` below separately fitted cold N=1024, and `-2.85597 dB` below normalized
N=640. Native review shows pervasive additive support dots/holes across the high-resolution
background; the normalized control is coarse/blocky but globally closer. Pursuit lowers the base's
displayed worst-pixel/7x7 maxima from `0.94516/0.90712` to `0.89027/0.84028`, but the improvement
is too sparse to repair the frame-scale failure.

The mask is useful for its intended foreground objective but does not reverse that conclusion.
Masked pursuit reaches `22.61468 dB`, `+0.25367 dB` over its N=960 base and `+0.19945 dB` over
masked cold N=1024, while reducing their worst residuals. It remains `-2.60041 dB` below masked
normalized N=640 (`25.21509 dB`), with worse MS-SSIM/LPIPS (`0.91182/0.22454` versus
`0.93974/0.20279`). Its isolated pixel maximum is slightly better (`0.61809` versus `0.64110`),
but its 7x7 maximum is worse (`0.50567` versus `0.46469`). Relative to the same arms trained on
the full frame, mask supervision improves foreground PSNR by `+8.96455`, `+1.54501`, `+1.79868`,
and `+0.85143 dB` for normalized, base, pursuit, and cold N=1024 respectively. Full-frame metrics
for masked arms are intentionally poor because their black background is outside the objective.

All projection proposals fail closed. Full-frame PCG lowers raw MSE but worsens MS-SSIM and LPIPS;
masked PCG selects step zero (or has only float-scale MSE movement) and fails strict-MSE/LPIPS
safety. Every additive endpoint contains exactly four arrays, the pursuit bases match their exact
N=960 rows, mask state is absent from fields, and internal/cold/repeated parity is below `1e-6`.
The mask has 87,639 active pixels (`7.0359%`). Best-checkpoint selection is a major scaling
limitation: normalized selects update 1 and masked additive selects update 26 despite all arms
attempting 500 updates. The masked loss is active-pixel weighted, but inherited checkpoint
selection uses global PSNR against the black-matted raster, so outside-mask spill may affect the
selected state. This diagnostic therefore exposes horizon/optimizer and checkpoint-domain
mismatches and is not an equal-effective-optimization comparison.

Evidence:
`ara/evidence/hier029-janelle-full-resolution-mask-2026-08-11/run.md` and
`results/hier029_janelle_c0001_s1200_mask_factorial_s0_diagnostic_2026-08-11/index.html`.

### Review

#### Verdict

Complete negative scaling diagnostic; retain mask-aware research hooks default-off

#### Self-reviewed

Yes

#### Correctness

Focused mask/projection/pursuit tests pass, the maintained report checker accepts the 242-file
bundle, all eight endpoints cold-replay within `4.77e-7`, and exact source/mask/count/work/base/
payload receipts pass. The partial-mask projector now reports active-domain operator parity while
returning the actual maintained full-crop replay; the historical full-mask path is unchanged.

#### Evidence quality

The report is unusually complete for a diagnostic, but the source was already exposed, there is
one image/seed/device, the worktree is dirty, and review is producer-only. The 1200x1038 raster is
the project full-resolution regime, not the 5328x4608 native camera raster.

#### Simplicity

Both optional mask hooks are encoder-only and preserve the ordinary endpoint. The result does not
add containment, a second renderer pass, or a side payload.

#### Missing cases

Native 5328x4608, fresh multi-image/multi-seed evidence, count/resolution scaling, learning-rate
or horizon scaling, equal bytes/rate, downstream response, and distinct review.

#### Required changes

Do not generalize HIER-028 beyond max-side 160 or promote additive/pursuit as the full-resolution
default. Keep normalized rendering and the maintained conversion pipeline unchanged.

#### Optional improvements

Any successor should scale count/work with pixel area and freeze a resolution-aware optimizer
schedule on new sources. A masked successor should keep common full/foreground metric domains and
separate foreground modeling from exact mask containment.

### Handoff

#### Objective

Determine whether HIER-028's N=960+64 pure-additive result survives the established 1200x1038
Janelle regime and whether encoder-only foreground masking changes the answer.

#### Changes

Added optional mask-restricted coefficient projection and residual selection, corrected partial-
mask projection replay/parity accounting, added focused tests and a frozen eight-cell driver, and
produced a portable report plus synchronized research evidence. Maintained defaults are unchanged.

#### Evidence

The 242-file corrected bundle passes the maintained report checker. Fifty-five focused tests pass;
all fields/counts/work/hashes/base-prefix/payload/parity receipts pass. Native source,
reconstruction, foreground, and error sheets were reviewed at report resolution.

#### Assumptions

“Full resolution” follows the repository's existing Janelle max-side-1200 convention. The encoder
knows the source and mask, and the masked objective deliberately does not enforce containment.

#### Uncertainties

One exposed image/seed/device, dirty sources, producer-only review, checkpoint-domain mismatch,
unequal rows/work, and no native-5328/equal-rate/downstream evidence.

#### Review focus

Check the active-domain projection receipt, full raw replay outside a partial mask, encoder-only
endpoint purity, exact per-mode N=960 base sharing, foreground metric domain, early selected
checkpoints, and whether the visible additive stipple supports the negative interpretation.

#### Protected actions not taken

No downscale after failure, schedule/count/threshold retune, containment arm, result-bundle
rewrite, default change, or claim promotion. The failed driver bundle remains preserved.

#### Recommended next action

Obtain distinct review. If research continues, preregister resolution-scaled capacity/work and an
optimizer/checkpoint schedule on new sources rather than tuning this exposed image.

## Notes

The output is an immutable diagnostic bundle once `COMPLETED` is written. If the 1200x1038 run
cannot fit the available 8 GiB device, retain the error rows and report the resource boundary; do
not silently reduce resolution.
