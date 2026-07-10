# Native Image-GS Comparison

This artifact executes the pinned official Image-GS checkout in an isolated environment. Metrics are centrally recomputed from float reconstructions; upstream-reported metrics, analytical payload estimates, synchronized wall timing, and synchronized end-to-end render timing are retained separately.

Profile `matched_steps_fixed_n`: Common final N and requested steps; Image-GS progressive allocation is disabled, so it starts at full N while StructSplat retains its pinned growth policy.

Image-GS emits no packed codec stream. `analytical_bpp` follows its documented attribute-bit formula and omits headers/min-max metadata; `actual_bpp` remains blank. Native trajectory samples use Image-GS's evaluation cadence rather than adding per-step GPU synchronization. Target hits are interval-censored at that cadence. Final Image-GS fields are terminal-step selections. `proxy_ms_ssim` is the shared small-image adaptive proxy, not the paper's fixed five-scale native MS-SSIM.

Official environment reproduction: no; algorithm/build provenance is pinned, but the recorded Python/Torch/CUDA versions differ from the official environment.

| Image | Profile | Side | Cap | Start | Seed | Steps | PSNR | Proxy MS-SSIM | LPIPS | AUC | Sync fit s | Native self s | Render FPS | Analytical bpp | Commit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| COCO_train2014_000000000009 | matched_steps_fixed_n | 160 | 640 | 640 | 0 | 500 | 23.8240 | 0.96195 | 0.1828 | 21.6145 | 1.174 | 0.967 | 2252.4 | 8.533 | 03088368d426 |
| COCO_train2014_000000000009 | matched_steps_fixed_n | 160 | 640 | 640 | 1 | 500 | 23.7598 | 0.96375 | 0.1863 | 21.6613 | 1.726 | 1.448 | 2373.2 | 8.533 | 03088368d426 |
| COCO_train2014_000000000025 | matched_steps_fixed_n | 160 | 640 | 640 | 0 | 500 | 22.8885 | 0.94726 | 0.5171 | 21.5779 | 1.183 | 0.983 | 2483.9 | 9.660 | 03088368d426 |
| COCO_train2014_000000000025 | matched_steps_fixed_n | 160 | 640 | 640 | 1 | 500 | 22.8422 | 0.94662 | 0.5087 | 21.5491 | 1.204 | 1.010 | 2347.6 | 9.660 | 03088368d426 |
| COCO_train2014_000000000030 | matched_steps_fixed_n | 160 | 640 | 640 | 0 | 500 | 29.2083 | 0.98663 | 0.1397 | 26.8363 | 1.241 | 1.038 | 2175.7 | 9.570 | 03088368d426 |
| COCO_train2014_000000000030 | matched_steps_fixed_n | 160 | 640 | 640 | 1 | 500 | 29.2802 | 0.98648 | 0.1313 | 26.8358 | 1.398 | 1.152 | 2329.9 | 9.570 | 03088368d426 |
| COCO_train2014_000000000034 | matched_steps_fixed_n | 160 | 640 | 640 | 0 | 500 | 20.1926 | 0.91344 | 0.2858 | 18.5517 | 1.226 | 1.012 | 2165.8 | 9.660 | 03088368d426 |
| COCO_train2014_000000000034 | matched_steps_fixed_n | 160 | 640 | 640 | 1 | 500 | 20.0542 | 0.91084 | 0.3061 | 18.4648 | 1.136 | 0.943 | 2267.4 | 9.660 | 03088368d426 |

## Paired Image-GS vs StructSplat Default

Positive is an Image-GS gain; timing and LPIPS signs are inverted so positive always means better. Displayed intervals are marginal 95% image-bootstrap intervals; a final-quality relation uses PSNR and proxy MS-SSIM with Bonferroni-adjusted 95% familywise bounds. LPIPS is reported separately. AUC is diagnostic only because the native histories use different render clamping/cadence semantics. Paired rows require identical run-recorded decoded-pixel hashes. Final N and requested steps match, but Image-GS starts at full N while the pinned StructSplat row starts at half N and grows. Image-GS synchronized fit wall includes its terminal image logging and checkpoint write, while StructSplat fit timing does not; timing deltas are therefore diagnostic and the displayed relation is not a strict implementation-dominance test.

| Pairs / images | PSNR gain [95% CI] | Proxy MS-SSIM gain [95% CI] | LPIPS gain [95% CI] | Diagnostic AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | Final-quality relation | Familywise 95% relation |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 8 / 4 | -3.6011 [-4.3059, -2.7527] | -0.01879 [-0.02937, -0.00822] | -0.1842 [-0.2658, -0.1135] | -2.6909 [-3.2317, -1.9494] | +0.0487 [-0.0020, +0.0889] | +0.0505 [-0.0087, +0.0971] | structsplat dominates | structsplat dominates |
