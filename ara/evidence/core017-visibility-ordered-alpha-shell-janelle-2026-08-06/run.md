# CORE-017 visibility-ordered alpha-shell exposed diagnostic

## Scope

This note binds the ignored immutable CORE-017 bundle to its tracked task and the CORE-016 results
audit. It is one exposed `frame_00009`, one seed, packet inputs at calibrated/undistorted downscale
4, and common source targets at downscale 8. The same 23 cameras build/train all arms; `C0004`,
`C0025`, and `C1004` are reporting-only but not sealed confirmation. No realtime-gs source,
maintained StructSplat path, packet grammar, renderer, or default changed.

## Method and result

All four fixed-topology arms reuse byte-identical 970,310-byte quality-92 WebP/512-structure packet
sets and finish at exactly 5,000 Gaussians after 1,000 gsplat steps. Sparse structure proposes rays.
The alpha-shell backend reuses exact packet alpha to select the first of 48 ordered samples attaining
maximum multiview support, while codec appearance supplies radiance. Optional realtime-gs surface
cover replaces covariance/opacity only.

Interior/inherited reaches 20.7469 dB reporting PSNR, 0.901271 MS-SSIM, 0.163428 LPIPS, 0.016095
gradient MAE, 0.790721 alpha IoU, and 0.021650 outside alpha. Shell/inherited reaches 22.3342 dB
(+1.5873), 0.941373, 0.131568, 0.015708, 0.921256, and 0.006991. Shell/cover reaches 22.1278 dB
(+1.3809), 0.938626, 0.104433, 0.014749, 0.931089, and 0.005634. It passes every scalar gate.
Cover alone loses 0.7035 dB and 0.0161 alpha IoU, so covariance repair is not sufficient before
surface placement.

Shell placement evaluates zero sparse-index pairs for depth scoring versus 212,517,051 for ordinary
placement and takes 5.810--6.246 seconds versus 8.951--9.420. Shell/inherited first reaches the
interior baseline's terminal PSNR at step 200/2.064 seconds; shell/cover does so at step 500/5.599;
baseline reaches it at step 1,000/9.831. These are exposed same-run mechanism timings, not general
production-speed claims.

## Visual and decision disposition

The numerical gate passes and the mandatory visual gate fails. Shell placement removes most broad
glow and scattered floaters, but every reporting view retains visible directional trailing smear or
double-silhouette structure, with especially clear errors around the feet/body in `C0004` and
`C1004`; fine detail remains soft. Surface cover improves LPIPS, gradient, and alpha localization but
does not eliminate those artifacts. The tested alpha-shell route is not advanced to variable
topology and is not retuned on this frame.

The immutable bundle is
`results/core017_visibility_surface_janelle_frame00009_2026-08-06_v1/`; manifest SHA-256 is
`bf14b8e8d08609bdf89dd3c4474422a7ea8c0281c45cb81de8ba50e17252be2e`.
Independent replay checks 222 unique path/byte/SHA receipts, identical packet hashes across all
arms, exact counts, metric envelopes, and gate arithmetic. `scripts/check_report_bundle.py` rejects
the task-local schema with the four documented CORE-016 envelope errors, so this is internally
audited diagnostic evidence rather than a maintained portable report.

## Reproduction

```bash
PYTHONPATH=src:/home/alex/Documents/realtime-gs/src \
  /home/alex/Documents/realtime-gs/.venv/bin/python \
  scripts/experiments/core016_multiview_downstream.py \
  --profile surface2x2 \
  --frame /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00009 \
  --out results/core017_visibility_surface_janelle_frame00009_2026-08-06_v1
```

Focused compatibility checks use the realtime-gs environment and
`tests/test_codec_native_field.py tests/test_core017_surface_driver.py`. The bundle snapshots exact
executed StructSplat/realtime-gs sources because execution began from a dirty diagnostic worktree.
