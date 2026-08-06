# HIER-005 all-active error-weighted Janelle diagnostic

## Scope and verdict boundary

This is a dirty-worktree, one-exposed-image, downscaled implementation diagnostic requested to
test recovery over every active Gaussian, with row updates weighted by a spatially smoothed error
field. It compares the new recovery scope with a freshly executed touched-only control under the
same source snapshot and attempted optimizer-step schedule. It is not preregistered, independently
reviewed, held out, equal-FLOP/equal-wall-time, a maintained benchmark bundle, a semantic/default
decision, a convergence-speed result, or a compression result.

The implementation, driver, focused test, two architecture documents, and core skill are bound in
that path order by source-set SHA-256
`60eff11c57e3975966e209743529e1e633c4b7eec1c2e02866ed93adef6cdf6b`. HIER-005 remains
`in-review` with a distinct numerical/scientific reviewer required.

## Source and evaluation raster

- RGB source:
  `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg`
- RGB SHA-256: `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b`
- Supplied RGB bytes: `14,268,226`; native dimensions: `5,328x4,608`
- Mask:
  `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png`
- Mask SHA-256: `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`
- Evaluation dimensions: `512x443`, with logged Pillow LANCZOS RGB and nearest mask resampling
- Evaluation active pixels: `15,929`; same-raster evaluation PNG bytes: `29,263`
- PSNR/MSE use the thresholded foreground mask. SSIM, MS-SSIM, and LPIPS use the complete
  black-matted evaluation raster.

The native JPEG numerator is resolution-mismatched to the resized field. Only the same-raster PNG
comparison is relevant to this diagnostic's storage sanity check, and neither denominator makes an
uncoded float field an actual compressed package.

## Method

The new `all_error_weighted` scope makes every currently active row trainable. At each recovery
checkpoint it computes RGB residual MSE, performs mask-aware Gaussian smoothing as
`blur(error * mask) / blur(mask)`, and zeroes the result outside the mask. One additive-renderer
color VJP computes each Gaussian's support-averaged exposure to that field; a second mask VJP is
the normalization denominator. This is matrix-free and does not allocate a pixel-by-Gaussian
matrix.

The score is raised to power `0.5`, normalized to approximately mean one, and clipped to
`[0.05, 4.0]`. Its multiplier is applied to each row's actual post-Adam parameter update. Applying
the multiplier only to raw gradients would be largely canceled by Adam's second-moment
normalization. The selected smoothing sigma is `1.5 px`. Means, scales, rotations, and RGB
coefficients retain the touched-only learning rates and trust regions. Each checkpoint keeps the
best strict masked-SSE improvement and accepted geometry rebuilds the topology frontier.

Both full rows use 16 progress checkpoints and at most 50 Adam steps per checkpoint, for 800
attempted steps. This matches attempted iterations, not work: all-active recovery trains roughly
15k rows at early checkpoints, whereas touched-only recovery trains at most 1.3k--4.0k rows across
these targets.

## Commands

Touched-only control:

```bash
PYTHONPATH=src python scripts/experiments/hier005_pixel_contraction.py \
  --images /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg \
  --mask /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png \
  --out results/hier005_janelle_c0001_touched_control_progress16_current_2026-08-05 \
  --target-gaussians 2048 4096 8192 12000 \
  --max-side 512 --device cuda --renderer cuda_additive --lpips \
  --recovery-steps 50 --recovery-scope touched
```

All-active error-weighted arm used the same command and output
`results/hier005_janelle_c0001_all_error_weighted_progress16_2026-08-05`, with
`--recovery-scope all_error_weighted`.

## Probe controls

A 256-pixel-maximum-side, N=1,024 probe used the same 16x50 recovery schedule:

| scope / weight | run 1 PSNR | run 2 PSNR |
|---|---:|---:|
| touched-only | 31.9148 | -- |
| all-active, uniform weight | 35.8391 | -- |
| all-active, raw error (`sigma=0`) | 40.9619 | 41.6449 |
| all-active, smoothed error (`sigma=1.5`) | 41.1608 | 40.2558 |

The probe isolates a large all-active effect and an additional error-weighting effect on this
raster. It does not establish that smoothing improves raw-error weighting: the sigma ordering
reverses across CUDA repeats. Sigma `1.5` is retained because it is the requested candidate, not
because this post-hoc probe selected it. Sigma `3` and weight power `1` single runs reached
39.6892 and 40.2343 dB respectively.

## Matched full-raster outcomes

| N | touched PSNR | all-active PSNR | delta dB | touched LPIPS | all-active LPIPS | touched -> all SSE | total-time ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 29.2189 | 34.2336 | +5.0147 | 0.0162563 | 0.0053528 | 57.2025 -> 18.0279 | 1.150x |
| 4,096 | 30.5215 | 40.9777 | +10.4562 | 0.0227493 | 0.0017695 | 42.3803 -> 3.8154 | 1.231x |
| 8,192 | 52.3369 | 46.0264 | -6.3105 | 0.00001632 | 0.0011299 | 0.2790 -> 1.1931 | 1.404x |
| 12,000 | 57.2413 | 59.2835 | +2.0422 | 0.00000555 | 0.00000082 | 0.09019 -> 0.05636 | 1.556x |

All 16 checkpoints were individually accepted in every row. The new scope strongly reduces the
low-count square/grid imprint and improves the 2k, 4k, and 12k terminal metrics on this run, but it
is not a monotone replacement for touched-only recovery. At N=8,192 it changes the accepted
geometry enough that later contraction proposals follow a worse terminal path. A separate
all-active N=8,192 repeat reached 47.9699 dB and SSE 0.7626, still 4.3670 dB below the touched-only
control. Checkpoint-local strict improvement therefore does not imply dominance of the final
topology path.

The uncoded payload and exact row count are unchanged by the recovery scope. Every field remains
larger than the 29,263-byte same-raster PNG, so this experiment provides no compression gain.

## Localized-artifact follow-up

After the global-metric comparison, the preserved 8-bit source/reconstruction PNGs were rescored
with per-pixel RGB RMSE and maximum pooled-patch RMSE. Pixel quantiles use the same resized binary
foreground mask; patch maxima use squared error averaged over the complete black-matted raster and
then square-rooted. These are post-hoc morphology diagnostics, not calibrated perceptual or
artifact-free thresholds.

| N | scope | pixel q99 | pixel q99.9 | pixel max | pixels > 0.05 | max 7x7 patch RMSE |
|---:|---|---:|---:|---:|---:|---:|
| 4,096 | touched | 0.1018 | 0.1311 | 0.2073 | 12.010% | 0.0707 |
| 4,096 | all-active weighted | 0.0381 | 0.0708 | 0.0904 | 0.483% | 0.0465 |
| 8,192 | touched | 0.0075 | 0.0106 | 0.0148 | 0.000% | 0.0053 |
| 8,192 | all-active weighted | 0.0214 | 0.0517 | 0.0686 | 0.157% | 0.0359 |

Thus the 8k regression is strongly localized: the weighted field's maximum 7x7 patch error is
6.8x the touched field's even though every interleaved checkpoint reduced global masked SSE. At
4k, all-active weighting suppresses much of the cell imprint but still leaves substantially larger
local maxima than the visually clean 8k touched control. Global PSNR/SSE alone is therefore an
unsafe acceptance authority for the stated no-visible-artifact priority.

Code inspection identifies a concrete, unisolated renderer hypothesis. These runs use an AABB
support rectangle, `sigma_cutoff=3`, and `support_fade_alpha=0`. A nominal Gaussian still has
weight `exp(-4.5)=0.0111` at 3 sigma before the rectangle truncates it. The visible cell-aligned
edges and sparse square holes are consistent with support discontinuities, but this does not prove
causation because contraction topology and later geometry also align spatially. A matched
support-fade/cutoff factorial with refitting is required; re-rendering an already fitted field under
different semantics would not isolate the mechanism.

## Artifact receipts

- Touched metrics / manifest / HTML SHA-256:
  `cda33647a131ca1c15339dd91e7a4be4f0b8db02696aca645e6ca1f80cc51326`,
  `eaf9be4bbff2ba356e8dd2f6a8b9216bcb593b6ceef3e6cdb357d45baefe7566`,
  `cb62ed9bac67d0ffec13683d1a1609c8b557d576dba35763660f3003e4c67a2d`
- All-active metrics / manifest / HTML SHA-256:
  `d9fe98c9fe9d7a0f59697bd248c823666b9eec4b2f547f1cdb266685a24a1b47`,
  `b66a3bd4cb99267b78e4df64fa03215e0d32f39983643eadf666201b92cd6306`,
  `ac761337ce3b0eb43219db1c0998e50c37885271a470f582d0689fee8c7ded13`
- Both report source snapshots have identical core/driver SHA-256:
  `76dea386d061dc6d4257072fad1684475b530c864809de77671d0857ca27d0a2` and
  `5577b6c063d408566183176de19b97bc9027bbc3f44814e4638bf5b5ba3de906`.
- Each manifest verifies all `80/80` listed files; each HTML resolves `61/61` local links; each
  report contains 44 standalone SVG outcome curves.

## Verification and required next evidence

- Focused HIER-005 suite: `27 passed`.
- Pixel/Field-V2/render regression slice: `93 passed`.
- `./scripts/verify.sh`: `1,587 passed`, `4 skipped`, `514 deselected`; lint and every structural
  checker passed.

A distinct reviewer must audit the VJP attribution, post-Adam multiplier, strict rollback, and
topology-path interaction. The next bounded comparison should test touched-only interleaving plus a
terminal all-active weighted polish under a matched total-work protocol, and first screen
continuous support against the current hard AABB cutoff. Multi-image, predeclared repeats and
deterministic controls are required before a general quality or convergence claim; COMP-013/FIT-030
complete cold streams are required before a rate or compression claim.
