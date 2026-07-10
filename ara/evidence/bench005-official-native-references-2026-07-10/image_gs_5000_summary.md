# Native Image-GS Comparison

This artifact executes the pinned official Image-GS checkout in an isolated environment. Metrics are centrally recomputed from float reconstructions; upstream-reported metrics, analytical payload estimates, synchronized wall timing, and synchronized end-to-end render timing are retained separately.

Profile `siggraph25`: Paper-aligned 5000-step, constant-LR, 16-bit analytical-payload algorithm profile with native progressive allocation, applied at the requested benchmark resolution.

Image-GS emits no packed codec stream. `analytical_bpp` follows its documented attribute-bit formula and omits headers/min-max metadata; `actual_bpp` remains blank. Native trajectory samples use Image-GS's evaluation cadence rather than adding per-step GPU synchronization. Target hits are interval-censored at that cadence. Final Image-GS fields are terminal-step selections. `proxy_ms_ssim` is the shared small-image adaptive proxy, not the paper's fixed five-scale native MS-SSIM.

Official environment reproduction: yes.

| Image | Profile | Side | Cap | Start | Seed | Steps | PSNR | Proxy MS-SSIM | LPIPS | AUC | Sync fit s | Native self s | Render FPS | Analytical bpp | Commit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| COCO_train2014_000000000009 | siggraph25 | 160 | 640 | 320 | 0 | 5000 | 25.1393 | 0.97449 | 0.1151 | 23.8359 | 12.845 | 8.738 | 1260.7 | 4.267 | 03088368d426 |
| COCO_train2014_000000000025 | siggraph25 | 160 | 640 | 320 | 0 | 5000 | 24.7613 | 0.96871 | 0.2795 | 23.3761 | 12.469 | 8.556 | 1259.0 | 4.830 | 03088368d426 |
| COCO_train2014_000000000030 | siggraph25 | 160 | 640 | 320 | 0 | 5000 | 33.3636 | 0.99438 | 0.0277 | 30.3015 | 15.331 | 10.727 | 1178.3 | 4.785 | 03088368d426 |
| COCO_train2014_000000000034 | siggraph25 | 160 | 640 | 320 | 0 | 5000 | 23.3662 | 0.95231 | 0.1373 | 21.4661 | 16.411 | 11.536 | 1251.7 | 4.830 | 03088368d426 |

## Paired Image-GS vs `structsplat_best_default`

Positive is an Image-GS gain; timing and LPIPS signs are inverted so positive always means better. Displayed intervals are marginal 95% image-bootstrap intervals; a final-quality relation uses PSNR and proxy MS-SSIM with Bonferroni-adjusted 95% familywise bounds. LPIPS is reported separately. AUC is diagnostic only because the native histories use different render clamping/cadence semantics. Paired rows require identical run-recorded decoded-pixel hashes. Start N, final N, requested steps, and target pixels match; native loss, renderer, growth policy/wave count, and final 16-bit Image-GS quantization remain algorithm-specific. Analytical bpp is not a byte-matched rate constraint. Image-GS synchronized fit wall includes its terminal image logging and checkpoint write, while StructSplat fit timing does not; timing deltas are therefore diagnostic and the displayed relation is not a strict implementation-dominance test.

| Pairs / images | PSNR gain [95% CI] | Proxy MS-SSIM gain [95% CI] | LPIPS gain [95% CI] | Diagnostic AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | Final-quality relation | Familywise 95% relation |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 4 / 4 | +0.2201 [-1.4386, +1.8787] | +0.01959 [+0.00030, +0.03887] | -0.0369 [-0.0575, -0.0164] | +0.0831 [-0.7453, +0.9749] | -0.6787 [-3.9531, +2.9488] | -0.6708 [-3.9601, +2.9695] | native dominates | inconclusive |

## Paired Image-GS vs `structsplat_best_checkpoint`

Positive is an Image-GS gain; timing and LPIPS signs are inverted so positive always means better. Displayed intervals are marginal 95% image-bootstrap intervals; a final-quality relation uses PSNR and proxy MS-SSIM with Bonferroni-adjusted 95% familywise bounds. LPIPS is reported separately. AUC is diagnostic only because the native histories use different render clamping/cadence semantics. Paired rows require identical run-recorded decoded-pixel hashes. Start N, final N, requested steps, and target pixels match; native loss, renderer, growth policy/wave count, and final 16-bit Image-GS quantization remain algorithm-specific. Analytical bpp is not a byte-matched rate constraint. Image-GS synchronized fit wall includes its terminal image logging and checkpoint write, while StructSplat fit timing does not; timing deltas are therefore diagnostic and the displayed relation is not a strict implementation-dominance test.

| Pairs / images | PSNR gain [95% CI] | Proxy MS-SSIM gain [95% CI] | LPIPS gain [95% CI] | Diagnostic AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | Final-quality relation | Familywise 95% relation |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 4 / 4 | -0.3601 [-1.4527, +0.7325] | +0.01038 [-0.00179, +0.02254] | -0.0566 [-0.0756, -0.0260] | +0.2528 [-0.3848, +0.9620] | -2.3808 [-4.4171, -0.3445] | -2.3754 [-4.4189, -0.3320] | tradeoff | tradeoff |
