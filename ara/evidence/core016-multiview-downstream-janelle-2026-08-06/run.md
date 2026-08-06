# CORE-016 exposed multiview downstream diagnostic

## Scope

This evidence note binds the ignored immutable v1--v6 Janelle result directories to the tracked
CORE-016 task, ADR, and adversarial results audit. It is one exposed frame, one nominal seed,
packet inputs at calibrated/undistorted downscale 4, and common training/reporting targets at
downscale 8. `C0004`, `C0025`, and `C1004` are reporting-only, but were inspected and reused for
post-hoc development; they are not confirmation data. No maintained pipeline or default changes.

## Retained development result

The retained run is
`results/core016_multiview_full23_matched10k_janelle_2026-08-06_v4/`, manifest SHA-256
`9c546d5c7b65f483326e525d8080b0a1c0928ff0805f50255b1191c4ff2d651c`.

At a common 10,000 final 3D Gaussians, its 23 quality-92 WebP / 512-structure packets total 956,301
bytes versus 3,850,647 bytes for 23 existing RTGSV containers (`4.02660564x`). Candidate versus
control reporting metrics are 25.1880 versus 24.0119 dB foreground PSNR, 0.967342 versus 0.960839
MS-SSIM, 0.079733 versus 0.099172 LPIPS, 0.012926 versus 0.014792 gradient MAE, and 0.949653 versus
0.928680 alpha IoU. Candidate first reaches control-final PSNR at step 500 / 5.148 native seconds
versus control step 1,400 / 13.266 seconds.

This is not an end-to-end speed or artifact-free result. Candidate lift/full native training/peak
VRAM are 9.467 s / 20.740 s / 0.366 GiB versus 4.894 s / 18.521 s / 0.310 GiB. Production time for
the pre-existing control inputs is unavailable. Native review retains soft silhouette halos,
fine-detail blur, and sparse floaters. The v4 rate plot also has a preserved packaging-only x-axis
typo (“eight-view”); its numeric 23-view byte row is correct.

## Negative cleanup results

- V5 strong exact-mask supervision, manifest
  `724af9c20b02d1ffa6a0c00674d4a59d21f7aa634ec38e275846ea7173a59209`, raises candidate alpha
  IoU to 0.960395 but loses 0.445559 dB and worsens gradient MAE by 0.000269 versus v4.
- V6 250-step fixed-topology low-rate mask polish, manifest
  `7da107f0bbaf8882d18011bc95c4ce8c3d1dc0eb4e48f56261e95133ed5a036b`, raises candidate alpha
  IoU to 0.954845 but loses 0.316067 dB and worsens gradient MAE by 0.000194 versus v4.

Both fail their frozen v4-retention preferences. Their executed custom reports only encoded the
within-run gates and therefore display `scalar_pass: true`; independent replay returns false for
the missing cross-run PSNR and gradient guards, and the driver is corrected for future runs without
rewriting either artifact.

## Integrity replay

The audit independently validated all path/byte/SHA receipts (108 unique in v2; 213 in each v3--v6
bundle), byte-identically cold-resaved all 100 v2--v6 candidate packets, recomputed packet/input
ledgers and decision deltas, recreated every source tensor hash, proved reporting IDs absent from
packet inputs, checked all checkpoint cardinalities, and loaded finite final NPZ arrays at declared
counts. `scripts/check_report_bundle.py` rejects the custom schema with four expected errors, so
these are internally audited diagnostics rather than maintained portable reports.

CUDA density control is not bitwise deterministic despite declared seeds; nearby repeated recipes
can differ slightly in count/quality. V5 timing was also contaminated by concurrent CPU compilation
and is excluded from speed interpretation. Full methods, numbers, commands, visual disposition, and
claim limits are in
[`docs/research/2026-08-06-codec-native-dual-plane-results-audit.md`](../../../docs/research/2026-08-06-codec-native-dual-plane-results-audit.md)
and [`tasks/CORE-016-codec-native-dual-plane-field.md`](../../../tasks/CORE-016-codec-native-dual-plane-field.md).
