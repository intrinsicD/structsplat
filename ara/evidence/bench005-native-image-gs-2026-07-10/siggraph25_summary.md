# Native Image-GS Comparison

This artifact executes the pinned official Image-GS checkout in an isolated environment. Metrics are centrally recomputed from float reconstructions; upstream-reported metrics, analytical payload estimates, synchronized wall timing, and synchronized end-to-end render timing are retained separately.

Profile `siggraph25`: Paper-aligned 5000-step, constant-LR, 16-bit analytical-payload algorithm profile with native progressive allocation, applied at the requested benchmark resolution.

Image-GS emits no packed codec stream. `analytical_bpp` follows its documented attribute-bit formula and omits headers/min-max metadata; `actual_bpp` remains blank. Native trajectory samples use Image-GS's evaluation cadence rather than adding per-step GPU synchronization. Target hits are interval-censored at that cadence. Final Image-GS fields are terminal-step selections. `proxy_ms_ssim` is the shared small-image adaptive proxy, not the paper's fixed five-scale native MS-SSIM.

Official environment reproduction: no; algorithm/build provenance is pinned, but the recorded Python/Torch/CUDA versions differ from the official environment.

| Image | Profile | Side | Cap | Start | Seed | Steps | PSNR | Proxy MS-SSIM | LPIPS | AUC | Sync fit s | Native self s | Render FPS | Analytical bpp | Commit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| COCO_train2014_000000000009 | siggraph25 | 160 | 640 | 320 | 0 | 5000 | 25.9400 | 0.97961 | 0.0891 | 24.2145 | 12.432 | 8.874 | 1302.0 | 4.267 | 03088368d426 |
| COCO_train2014_000000000025 | siggraph25 | 160 | 640 | 320 | 0 | 5000 | 24.4747 | 0.96699 | 0.2867 | 23.2102 | 12.320 | 8.674 | 1315.3 | 4.830 | 03088368d426 |
| COCO_train2014_000000000030 | siggraph25 | 160 | 640 | 320 | 0 | 5000 | 32.8472 | 0.99412 | 0.0325 | 30.0930 | 13.016 | 9.145 | 1260.2 | 4.785 | 03088368d426 |
| COCO_train2014_000000000034 | siggraph25 | 160 | 640 | 320 | 0 | 5000 | 23.2680 | 0.94891 | 0.1493 | 21.3783 | 13.302 | 9.175 | 1269.7 | 4.830 | 03088368d426 |

## Paired Image-GS vs StructSplat Default

Positive is an Image-GS gain; timing and LPIPS signs are inverted so positive always means better. Displayed intervals are marginal 95% image-bootstrap intervals; a final-quality relation uses PSNR and proxy MS-SSIM with Bonferroni-adjusted 95% familywise bounds. LPIPS is reported separately. AUC is diagnostic only because the native histories use different render clamping/cadence semantics. Paired rows require identical run-recorded decoded-pixel hashes. Start N, final N, requested steps, and target pixels match; native loss, renderer, growth policy/wave count, and final 16-bit Image-GS quantization remain algorithm-specific. Analytical bpp is not a byte-matched rate constraint. Image-GS synchronized fit wall includes its terminal image logging and checkpoint write, while StructSplat fit timing does not; timing deltas are therefore diagnostic and the displayed relation is not a strict implementation-dominance test.

| Pairs / images | PSNR gain [95% CI] | Proxy MS-SSIM gain [95% CI] | LPIPS gain [95% CI] | Diagnostic AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | Final-quality relation | Familywise 95% relation |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 4 / 4 | -0.3840 [-2.3698, +1.1997] | +0.01608 [+0.00074, +0.03142] | -0.0443 [-0.0652, -0.0243] | -0.5929 [-1.7871, +0.5845] | +0.9545 [-0.5534, +2.4623] | +0.9653 [-0.5431, +2.4738] | tradeoff | tradeoff |
