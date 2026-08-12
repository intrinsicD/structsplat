# HIER-030 Janelle 7k capacity and contained-mask diagnostic

## Evidence class

Frozen, exposed-source, dirty-worktree, producer-reviewed diagnostic on canonical Janelle C0001.
It separates HIER-029's count confound from the user's hard mask-containment requirement at the
repository's max-side-1200 Janelle evaluation regime. It is not a confirmation, a native-camera
run, a density-matched test, or evidence for a maintained default.

## Bound source and protocol

- RGB: native 5328x4608 `C0001.jpg`, SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b`.
- Mask: native 5328x4608 `mask_C0001.png`, SHA-256
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`.
- Raster: deterministic Pillow LANCZOS/nearest max-side 1200, yielding 1200x1038 and 1,245,600
  pixels. The thresholded mask has 87,639 active pixels (`7.035886%`).
- Device/seed: RTX 3050 8 GiB, CUDA, seed 0, required LPIPS, 256-row render chunks.
- Counts per mode: normalized N=4,375; projected cold additive N=6,562; exact projected N=6,562
  plus 438 fixed 0.35-pixel pursuit rows; projected cold additive N=7,000.
- Fit: 500 attempted Adam updates, unchanged HIER-028 learning rates and L1 + 0.3 SSIM objective,
  best-PSNR/final-count checkpoints every 25. This is count-scaled but not pixel-density- or
  optimizer-horizon-scaled.
- Contained mode: masked initialization/target/loss; `mask_contain=True`; certified anisotropic
  caps; margin 0.75 px; C0 support fade; pursuit selection eroded by
  `0.75 + 3 * 0.35 = 1.80 px`; final scale materialization and four-array endpoint reduction.
- Frozen integrity thresholds: all centres inside; unit coverage outside `<=1e-7`; reconstruction
  outside `<=1e-7`; exact count/payload/base/projection/parity/work receipts.

Command:

```bash
PYTHONPATH=src python scripts/experiments/hier030_janelle_7k_contained_diagnostic.py \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png \
  results/hier030_janelle_c0001_s1200_7k_contained_s0_diagnostic_2026-08-11 \
  --max-side 1200 --seed 0 --device cuda --lpips
```

## Metrics

The primary columns use complete RGB for `full_frame` and the black-matted foreground crop for
`masked_contained`.

| mode | arm | N | PSNR | MS-SSIM | LPIPS | pixel max | 7x7 max | selected step |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full | normalized | 4,375 | 33.65272 | 0.974791 | 0.165820 | 0.68133 | 0.37569 | 101 |
| full | additive base | 6,562 | 34.65036 | 0.978183 | 0.171042 | 0.51563 | 0.20925 | 500 |
| full | pursuit | 7,000 | 35.00091 | 0.979014 | 0.154966 | 0.20265 | 0.11716 | 500 |
| full | cold additive | 7,000 | 35.05745 | 0.980458 | 0.157884 | 0.52445 | 0.17125 | 500 |
| contained | normalized | 4,375 | 22.78911 | 0.960964 | 0.138523 | 1.00000 | 0.44293 | 26 |
| contained | additive base | 6,562 | 20.65644 | 0.918300 | 0.150921 | 0.98562 | 0.46413 | 376 |
| contained | pursuit | 7,000 | 20.97452 | 0.919361 | 0.146646 | 0.98562 | 0.45781 | 376 |
| contained | cold additive | 7,000 | 21.57449 | 0.940046 | 0.134944 | 0.87855 | 0.46045 | 476 |

Full pursuit gains `+21.35407 dB` over the immutable literal-count HIER-029 N=1,024 pursuit,
`+1.34819 dB` over normalized N=4,375, and `+0.35055 dB` over its exact N=6,562 base. Cold
N=7,000 is `0.05654 dB` higher in PSNR and has higher MS-SSIM, while pursuit has lower LPIPS and
substantially lower pixel/7x7 maxima. Thus count fixes the global support failure; residual pursuit
does not clearly beat an ordinary same-count field at this scale.

Contained pursuit gains `+0.31809 dB` over its base but loses `0.59997 dB` to cold N=7,000 and
`1.81459 dB` to normalized N=4,375. Cold N=7,000 is the strongest additive/perceptual contained
arm; normalized remains strongest in contained PSNR/MS-SSIM.

## Exact containment and endpoint audit

Every contained endpoint reports:

- `centres_outside_mask = 0` and `centres_inside_mask = N`;
- `unit_coverage_outside_abs_max = 0.0`;
- `reconstruction_outside_abs_max = 0.0`;
- exactly `means`, `log_scales`, `rotations`, and `colors` in its field payload;
- no mask, scale-cap, mass, denominator, optimizer, or auxiliary-RGB payload;
- passing direct/internal/cold/repeated parity and finite reconstruction receipts.

There are 24,937 centre instances across the four contained endpoints. This sum is descriptive;
the N=6,562 field is also the exact base prefix of its N=7,000 pursuit sibling. The pursuit tail's
438 centres are selected only from the 1.80-pixel eroded mask, and its C0-faded analytic update
matches the ordinary renderer.

## Preserved failed attempt and correctness repair

The first complete execution placed every centre inside but correctly failed support containment.
Maximum outside unit coverage was `0.00326563`, `0.00047007`, `0.00047007`, and `0.00142278` for
contained normalized, base, pursuit, and cold N=7,000. The largest outside reconstruction was
`0.514975`. The immutable failed bundle remains at:

`results/hier030_janelle_c0001_s1200_7k_contained_s0_diagnostic_2026-08-11_failed_stale_anisotropic_checkpoint_caps`.

Diagnosis found that anisotropic containment caps refresh on a cadence during optimization, while
terminal and restored best-checkpoint states previously reused whatever caps were last certified.
A checkpoint captured between refreshes could therefore restore moved means with stale caps.
Fresh recertification made the same fields exactly zero outside. The fitter now forces
`refresh=True` both before terminal checkpointing and after restoring a selected checkpoint, and a
focused regression forces an early best checkpoint and verifies exact-zero replay. The unchanged
frozen experiment was then rerun from scratch; no count, schedule, margin, threshold, or metric
changed.

## Post-hoc boundary audit

This descriptive audit was performed after sealing the report, using only each manifest-bound
`analysis.npz`. Euclidean distance to the outside of the raw mask defines one-, two-, and four-
pixel interior boundary bands. It did not affect selection or rewrite the bundle.

| arm | uncovered active pixels | max uncovered depth | SSE <=1 px | SSE <=2 px | SSE <=4 px | interior >4 px PSNR |
|---|---:|---:|---:|---:|---:|---:|
| normalized N=4,375 | 1,679 | 3.162 px | 87.50% | 94.67% | 95.70% | 35.908 dB |
| additive base N=6,562 | 883 | 2.236 px | 75.99% | 90.20% | 94.56% | 32.750 dB |
| pursuit N=7,000 | 808 | 1.414 px | 81.76% | 91.84% | 94.33% | 32.893 dB |
| cold additive N=7,000 | 869 | 2.236 px | 82.37% | 92.96% | 96.27% | 35.313 dB |

The foreground has 87,639 pixels; the four-pixel band has 10,395. The low aggregate contained
score is consequently dominated by a narrow silhouette ring. HIER-029's loss-only mask could
place or spread support across the raw boundary and therefore had an easier foreground objective,
but did not satisfy the user's containment semantics.

## Native-size visual audit

The N=7,000 full-frame reconstructions remove HIER-029's pervasive stipple/hole lattice. The
normalized field remains more patch-like; additive fields are smoother. All still lose high-
frequency face, lace, hair, and machinery detail relative to the source. Pursuit shows a few
small bright tail speckles along the diagonal rail; cold N=7,000 is slightly smoother, consistent
with its PSNR/MS-SSIM edge, while pursuit's lower local maxima and LPIPS are also visible.

Contained reconstructions have clean black outside support, and placement sheets show red centres
only inside the silhouette. Foreground error is strongest on hair, lace, fine facial/clothing
detail, and the mask edge. Outside-support visualizations are identically black. No visual/report
or metric-domain mismatch was found.

## Density interpretation

Seven thousand rows over 1,245,600 pixels is one row per about 178 pixels. HIER-028 used 1,024 rows
at max-side 160; preserving that row-per-pixel density at max-side 1200 requires approximately
57,600 rows. HIER-030 therefore proves that literal N=1,024 was grossly under-capacity, but 7k is
still not a density-matched full-resolution test. The report's `overall_pass=true` means the
diagnostic completed with passing integrity and containment; `formal_claim_ready=false` remains
the scientific disposition.

## Execution and validation

The corrected report contains 268 manifest-bound files (256,710,088 bytes) and passes:

```bash
python scripts/check_report_bundle.py \
  results/hier030_janelle_c0001_s1200_7k_contained_s0_diagnostic_2026-08-11 \
  --allow-dirty
```

Focused pursuit, mask/checkpoint, and driver coverage passed before the evidence rerun. The final
repository gate passes Ruff, 1,985 portable tests with 26 skips and 514 deselections, and every
docs, ARA, task-policy, script-layout, and agent-workflow checker.

## Decision

Accept the capacity and containment diagnoses, not a method/default promotion. HIER-029's poor
full-frame result was chiefly a literal-count error. Exact mask-contained fields now satisfy the
requested placement and support contract. Do not select pursuit over cold additive from this one
image, and do not tune the exposed boundary result. A new-source successor should test explicit
contained boundary allocation; a separate approximately 57.6k or native-camera experiment is
needed for density/full-resolution evidence.

## Receipts

- Report:
  `results/hier030_janelle_c0001_s1200_7k_contained_s0_diagnostic_2026-08-11/index.html`
- Manifest SHA-256: `80c9c64a3b59730e9ab3fdd4dddbd3b55cfd8f50088054f5416b9764fe83f840`
- Metrics SHA-256: `4cd91f2395e2b74d1ca559c7dbd7af3f803ae802a7362917b41461cb76fb030b`
- Decision SHA-256: `445b1cbf37dff322aec31152e30dc92e719f94ada815bbe318d5aebe82d44124`
- Index SHA-256: `1cd74c1d2236ef2d22f819c952fc60af1fdad7d708cd7861a0365342dc2d68c3`
- Config SHA-256: `a87bb6b87968fd63a1ed225f687b3b4317213398b844c1ecc47011f005b77d24`

## Limitations

One exposed image; one seed; one RTX 3050; dirty sources; producer-only review; 1200x1038 rather
than native 5328x4608; 7k rather than density-matched 57.6k; unequal counts/work; target-known
pursuit; post-hoc boundary decomposition; no boundary-aware schedule, equal bytes, actual rate,
downstream behavior, broad-corpus evidence, or distinct review.
