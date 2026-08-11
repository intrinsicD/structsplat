# HIER-015--021 exact-7k portfolio diagnostic

## Disposition

Supported only at the measured diagnostic scope: an unchanged direct normalized 7,000-Gaussian
field plus a target-known, explicit SPT1 RGB8 exception stream passed the four-image HIER-021
prospective screen and a frozen no-refit replay over 40 persisted fields, including all 16
`tests/test_images` fields. It repaired all nine recorded direct-versus-HIER-005 local failures.

This is **not** a pure-Gaussian repair, held-out confirmation, production-codec result, actual-rate
claim, default decision, or evidence that the method works on arbitrary images. The run used one
seed, a dirty but source-snapshotted tree, an RTX 3050, and no distinct prospective or outcome
reviewer. The source image is used by the encoder to choose and populate exact RGB8 exceptions;
decode uses only the unchanged field and stored SPT1 bytes.

## Causal sequence retained

| Task | Frozen outcome |
|---|---|
| HIER-015 | Direct normalized fitting is visually clean and much stronger globally, but one worst-pixel clause fails; additive geometry relaxation remains visibly latticed. |
| HIER-016 | Fixed-geometry color-tail optimization cannot repair the isolated failure without violating its frozen rules. |
| HIER-017 | Lowering normalized-render epsilon repairs low-coverage pixels but regresses whole-image/local metrics. |
| HIER-018 | Sixty-four counted broad-background rows create coverage but regress all four fresh images. |
| HIER-019 | Same-field twice-scale recovery repairs the local error but raises LPIPS, so the global transaction rolls it back. |
| HIER-020 | Pointwise-safe SST1 passes fresh/lineage screens but LPIPS rolls back required repairs on `000009` and `000034`; 14/16 repository-test fields pass. |
| HIER-021 | Coherent radius-3 source-RGB patches pass the fresh screen and all 40 frozen replays, including the two HIER-020 counterexamples. |

HIER-021 does not erase these negative results: they establish that the general fix came from an
extra source-derived residual representation, not from epsilon, counted coverage, color-only
optimization, or a same-field Gaussian prior.

## Frozen HIER-021 representation

- Ordinary field: HIER-015 direct normalized fit, exact `N=7,000`, `eps=1e-8`, seed 0,
  max-side 512, 750 steps.
- Seeds: normalized denominator `D < 1e-8`.
- Support: clipped Chebyshev radius 3, hence a 7x7 square around each seed.
- Records: only sites where source RGB8 differs from displayed baseline and exact source RGB8
  strictly improves raw RGB SSE.
- SPT1: 16-byte `<4sIII` header followed by sorted unique `<IBBB>` records, seven bytes each.
- Selection: candidate must be canonical, finite, exact outside records, parity-clean, and
  noninferior to ordinary MSE/pixel/7x7/MS-SSIM/LPIPS; a non-material or unsafe candidate selects
  the zero-byte ordinary interpretation.

The format/parser tests include malformed magic/length/dimensions, duplicate/unsorted/out-of-range
indices, invalid colors, exact round trip, boundary clipping, field immutability, normalized-only
decode, and CPU/CUDA selection/decode parity.

## Executed bundles

All listed bundles pass `python scripts/check_report_bundle.py`. Reviewed bundles copy the raw
cells and change only review/config/report/manifest records.

| Bundle | Decision SHA-256 | Manifest SHA-256 | Outcome |
|---|---|---|---|
| `results/hier021_coco_source_patch_2026-08-10` | `b5a4685f30c3207cfed72d57d2acf00244ea3d9c927445d90c7b70124c301d19` | `5a77ce3cc1b278740e9c661c12496a7b45c6623d30576f7c4d0207ed39f60bd9` | 8/8 attempts; numeric pass; visual pending |
| `results/hier021_coco_source_patch_reviewed_2026-08-10` | `8371e0e1e84efe9692660f084cf9dc3daea597e83d63006ca8cbbbbb9e42125c` | `413e56064a228184cec423e22c33bb097577f10e5aec3449c9fa47055edb273d` | review-only visual pass; bounded pass |
| `results/hier021_replay_40_fields_2026-08-10` | `86dc2a66b8ef5acd93414c8be8f9fd5e56870db002c828a8fc0d90a4d4fe912c` | `f308dd4e839a201e9ef0849019ac90585d6b5a9f390228254e56e8fdf2d99566` | 80/80 attempts; numeric pass; visual pending |
| `results/hier021_replay_40_fields_reviewed_2026-08-10` | `3dafcc9a5d62f4ff02b643a27434a5835b017bd5d9689eac59e7b7645b92be69` | `9d0e1ba945b885c9105377a85266f66dbff624549403b1fa6e6b6ff398cb4418` | review-only visual pass; bounded pass |

The replay inputs are the gate-clean reviewed HIER-020 H15--H19 replay (20 fields), reviewed
HIER-020 fresh bank (4 fields), and complete HIER-020 `tests/test_images` bundle (16 fields). The
HIER-021 task's earlier phrase “40 prior fields, then 16 tests” was a documentation count defect;
the snapshotted executable protocol and actual immutable ledger are 24 prior fields plus 16 tests,
40 total. The task was corrected without changing the method or rerunning cells.

## Independent recalculation

### Fresh four-image bank

- 4/4 exact-count field hashes unchanged; maximum ordinary/candidate replay parity
  `3.5762787e-7`.
- 3/4 images select SPT1; 2,722 records and 19,102 selected side bytes.
- SPT1 records cover `0.3674%` of all four rasters and add `1.6979%` relative to the four raw
  Gaussian NPZ files. These are reference-file ratios, not actual codec rate.
- Median candidate-construction timing ratio recorded by the raw decision: `1.00290x` direct fit;
  median selected decode: `1.00802x` ordinary render; maximum decode ratio: `1.25858x`.
- Audit correction: the raw pipeline ratio omitted separately logged target-known metric/RDO time.
  Adding it gives median `1.00656x` and maximum `1.00693x`, still inside the frozen `1.25x` gate.
- The fresh bank contains no direct-versus-HIER-005 local failure, so it validates prospective
  safety/selection but does not prospectively confirm repair of the known failure mechanism.

### Frozen 40-field replay

- 40/40 exact-count field hashes unchanged; maximum ordinary/candidate replay parity
  `4.7683716e-7`; 80/80 copy/evaluation attempts succeeded.
- 24/40 images select SPT1; 20,137 selected records and 141,343 selected side bytes. Selected
  records are `0.2748%` of all replay raster pixels and the selected side bytes are `1.2563%` of
  the 40 raw Gaussian NPZ files. These are not complete-stream or entropy-coded rates.
- All nine recorded direct-versus-HIER-005 pixel/7x7 failures are repaired, including HIER-020's
  `000009` and `000034` failures.
- All 16 `tests/test_images` pairs gain at least `3.07308 dB` over HIER-005. Worst across-image
  selected comparisons remain noninferior: MSE ratio at most `0.492825`, MS-SSIM delta at least
  `+0.0643423`, LPIPS delta at most `-0.228884`, pixel-max delta at most `-0.128668`, and 7x7-max
  delta at most `-0.0470035`.
- Median recorded pipeline/decode ratios are `1.00345x`/`1.06281x`; maximum decode is `1.52647x`.
  Adding logged metric/RDO work gives encoder-pipeline median `1.00684x` and maximum `1.16212x`,
  still passing `1.25x`.
- Native full-frame review of all 24 selected candidates, plus native inspection of `000009` and
  `000034`, found no visible square seam, speckle, ringing, checker/lattice pattern, wash, or new
  blur. This is producer self-review, not independent acceptance.

Payload audit reparsed every raw `candidate.spt1`, checked `size = 16 + 7*K`, row SHA-256, selected
`tail.spt1` presence/content, and absence of a selected payload for ordinary-mode rows; zero
discrepancies were found in both raw bundles.

## Results-audit claim table

| Claim | Disposition | Reason |
|---|---|---|
| Direct normalized exact-7k plus frozen SPT1 passes every recorded metric on the 16 repository test images. | **Confirm, bounded diagnostic scope.** | Per-image frozen gates and native review pass; test images were already consumed and two shaped HIER-021, so they are not held out. |
| HIER-021 fixes the 7,000-Gaussian field itself. | **Refute.** | Field hashes and row count are unchanged; quality comes from 7-byte source-RGB exception records outside the field. |
| HIER-021 is a production codec or compression win. | **Refute.** | NPZ+raw SPT1 is only reference accounting; no complete selected stream, entropy coding, or actual bpp exists. |
| HIER-021 works everywhere. | **Narrow.** | It passes 4 fresh and 40 replay fields at one seed/device; arbitrary-image, full-resolution, multi-seed, and independent confirmation remain untested. |
| HIER-021 has negligible measured overhead. | **Narrow.** | Encoder/RDO and render-only decode ratios pass the frozen bounds, but file I/O and a production decoder are outside scope. |
| Lower epsilon, counted background rows, or same-field recovery alone solves the failure. | **Refute on the measured banks.** | HIER-017--020 retain their frozen negative/counterexample bundles. |

## Exact commands

```bash
python scripts/experiments/hier021_source_patch_tail.py \
  --images /home/alex/Documents/datasets/train2014 \
  --out results/hier021_coco_source_patch_2026-08-10

python scripts/experiments/hier021_source_patch_tail.py \
  --review-from results/hier021_coco_source_patch_2026-08-10 \
  --out results/hier021_coco_source_patch_reviewed_2026-08-10 \
  --visual-disposition pass

python scripts/experiments/hier021_replay_persisted.py \
  --source-results \
    results/hier020_consumed_h15_h19_reviewed_2026-08-10 \
    results/hier020_coco_sparse_tail_recovery_v2_2026-08-10 \
    results/hier020_tests_test_images_2026-08-10 \
  --development-decision \
    results/hier021_coco_source_patch_reviewed_2026-08-10/decision.json \
  --out results/hier021_replay_40_fields_2026-08-10

python scripts/experiments/hier021_replay_persisted.py \
  --review-from results/hier021_replay_40_fields_2026-08-10 \
  --out results/hier021_replay_40_fields_reviewed_2026-08-10 \
  --visual-disposition pass
```

## Open boundary

No maintained pipeline/default changes. A separately reviewed integration task would need a
complete stream contract, production encode/decode accounting, an explicit policy for source-known
RDO, broader multi-seed/source evaluation, and independent confirmation. A pure-Gaussian successor
would need to reproduce the coherent repair without storing target RGB exceptions.
