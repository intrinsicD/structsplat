# Native Image-GS Comparison

This artifact executes the pinned official Image-GS checkout in an isolated environment. Metrics are centrally recomputed from float reconstructions; upstream-reported metrics, analytical payload estimates, synchronized wall timing, and synchronized end-to-end render timing are retained separately.

Profile `matched_steps_fixed_n`: Common final N and requested steps; Image-GS progressive allocation is disabled, so it starts at full N while StructSplat retains its pinned growth policy.

Image-GS emits no packed codec stream. `analytical_bpp` follows its documented attribute-bit formula and omits headers/min-max metadata; `actual_bpp` remains blank. Native trajectory samples use Image-GS's evaluation cadence rather than adding per-step GPU synchronization. Target hits are interval-censored at that cadence. Final Image-GS fields are terminal-step selections. `proxy_ms_ssim` is the shared small-image adaptive proxy, not the paper's fixed five-scale native MS-SSIM.

Official environment reproduction: yes.

| Image | Profile | Side | Cap | Start | Seed | Steps | PSNR | Proxy MS-SSIM | LPIPS | AUC | Sync fit s | Native self s | Render FPS | Analytical bpp | Commit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| COCO_train2014_000000000009 | matched_steps_fixed_n | 160 | 640 | 640 | 0 | 500 | 23.8286 | 0.96198 | 0.1818 | 21.6242 | 2.951 | 2.387 | 2030.9 | 8.533 | 03088368d426 |
| COCO_train2014_000000000009 | matched_steps_fixed_n | 160 | 640 | 640 | 1 | 500 | 23.7494 | 0.96357 | 0.1865 | 21.6617 | 2.264 | 1.717 | 2255.5 | 8.533 | 03088368d426 |
| COCO_train2014_000000000025 | matched_steps_fixed_n | 160 | 640 | 640 | 0 | 500 | 22.9163 | 0.94700 | 0.5051 | 21.5832 | 1.860 | 1.268 | 2352.7 | 9.660 | 03088368d426 |
| COCO_train2014_000000000025 | matched_steps_fixed_n | 160 | 640 | 640 | 1 | 500 | 22.8436 | 0.94661 | 0.5022 | 21.5419 | 2.365 | 1.902 | 2355.7 | 9.660 | 03088368d426 |
| COCO_train2014_000000000030 | matched_steps_fixed_n | 160 | 640 | 640 | 0 | 500 | 29.2729 | 0.98691 | 0.1304 | 26.8788 | 1.537 | 1.130 | 2234.9 | 9.570 | 03088368d426 |
| COCO_train2014_000000000030 | matched_steps_fixed_n | 160 | 640 | 640 | 1 | 500 | 29.2706 | 0.98665 | 0.1264 | 26.8345 | 1.872 | 1.421 | 276.5 | 9.570 | 03088368d426 |
| COCO_train2014_000000000034 | matched_steps_fixed_n | 160 | 640 | 640 | 0 | 500 | 20.3236 | 0.91590 | 0.2825 | 18.5850 | 1.624 | 1.175 | 2286.3 | 9.660 | 03088368d426 |
| COCO_train2014_000000000034 | matched_steps_fixed_n | 160 | 640 | 640 | 1 | 500 | 20.0790 | 0.90974 | 0.2909 | 18.4722 | 1.499 | 1.103 | 2197.3 | 9.660 | 03088368d426 |

## Paired Image-GS vs `structsplat_best_default`

Positive is an Image-GS gain; timing and LPIPS signs are inverted so positive always means better. Displayed intervals are marginal 95% image-bootstrap intervals; a final-quality relation uses PSNR and proxy MS-SSIM with Bonferroni-adjusted 95% familywise bounds. LPIPS is reported separately. AUC is diagnostic only because the native histories use different render clamping/cadence semantics. Paired rows require identical run-recorded decoded-pixel hashes. Final N and requested steps match, but Image-GS starts at full N while the pinned StructSplat row starts at half N and grows. Image-GS synchronized fit wall includes its terminal image logging and checkpoint write, while StructSplat fit timing does not; timing deltas are therefore diagnostic and the displayed relation is not a strict implementation-dominance test.

| Pairs / images | PSNR gain [95% CI] | Proxy MS-SSIM gain [95% CI] | LPIPS gain [95% CI] | Diagnostic AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | Final-quality relation | Familywise 95% relation |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 8 / 4 | -3.6639 [-4.3839, -2.7583] | -0.01907 [-0.02937, -0.00812] | -0.1773 [-0.2592, -0.1099] | -2.7060 [-3.2294, -1.9944] | -0.6599 [-0.9280, -0.3917] | -0.6646 [-0.9456, -0.3835] | structsplat dominates | structsplat dominates |

## Paired Image-GS vs `structsplat_best_checkpoint`

Positive is an Image-GS gain; timing and LPIPS signs are inverted so positive always means better. Displayed intervals are marginal 95% image-bootstrap intervals; a final-quality relation uses PSNR and proxy MS-SSIM with Bonferroni-adjusted 95% familywise bounds. LPIPS is reported separately. AUC is diagnostic only because the native histories use different render clamping/cadence semantics. Paired rows require identical run-recorded decoded-pixel hashes. Final N and requested steps match, but Image-GS starts at full N while the pinned StructSplat row starts at half N and grows. Image-GS synchronized fit wall includes its terminal image logging and checkpoint write, while StructSplat fit timing does not; timing deltas are therefore diagnostic and the displayed relation is not a strict implementation-dominance test.

| Pairs / images | PSNR gain [95% CI] | Proxy MS-SSIM gain [95% CI] | LPIPS gain [95% CI] | Diagnostic AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | Final-quality relation | Familywise 95% relation |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 8 / 4 | -3.6334 [-4.3008, -2.8440] | -0.01854 [-0.02816, -0.00797] | -0.1774 [-0.2622, -0.1088] | -2.6845 [-3.2047, -1.9637] | -0.7709 [-1.1781, -0.4046] | -0.7771 [-1.2070, -0.3960] | structsplat dominates | structsplat dominates |
