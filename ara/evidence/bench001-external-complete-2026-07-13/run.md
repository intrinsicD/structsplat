# Complete External Benchmark Run

Date: 2026-07-13

Scope:
- StructSplat/common-harness fixed-storage sweep over the repositories present in `/home/alex/Documents/Deeplearning3/external`: StructSplat plus GaussianImage-style and Image-GS-style analogues.
- Native AIR inference benchmark.
- Native official GaussianImage benchmark in an isolated, pinned environment.

External repositories found:
- `/home/alex/Documents/Deeplearning3/external/AIR`
- `/home/alex/Documents/Deeplearning3/external/GaussianImage`
- `/home/alex/Documents/Deeplearning3/external/structsplat`

`Instant-GI` was not present under `/home/alex/Documents/Deeplearning3/external`, so `instant_gi_quadtree_fixed` was excluded from the fixed-storage sweep.

## StructSplat/common fixed-storage sweep

Report: [results/storage_budget_168k_external_present/index.html](../../../results/storage_budget_168k_external_present/index.html)

Artifacts:
- [summary.md](../../../results/storage_budget_168k_external_present/summary.md)
- [metrics.json](../../../results/storage_budget_168k_external_present/metrics.json)
- [metrics.csv](../../../results/storage_budget_168k_external_present/metrics.csv)
- [convergence_curves.csv](../../../results/storage_budget_168k_external_present/convergence_curves.csv)
- [storage_method_summary.csv](../../../results/storage_budget_168k_external_present/storage_method_summary.csv)
- [resource summary](../../../results/storage_budget_168k_external_present/storage_budget_168k_external_present.resources.json)
- [GPU memory samples](../../../results/storage_budget_168k_external_present/storage_budget_168k_external_present.gpu_memory.csv)

Protocol:
- 4 COCO images, max side 160
- 2 seeds
- 168 KiB storage budget, 5,376-Gaussian cap at 32 bytes/Gaussian
- 10,000 requested iterations
- PSNR, SSIM, MS-SSIM, LPIPS, AUC/convergence, time, storage status, target-hit rates

Result:
- 320/320 cells completed successfully.
- 40 methods, 4 images, 2 seeds.
- Storage status: 296 exact rows, 24 overfilled rows. Overfilled rows were scored but marked as capacity-mismatched.

Resource telemetry:
- Wall time: 4,941.746 s
- Max child RSS: 2,240,064 KiB
- Peak GPU memory: 6,667 MiB on NVIDIA GeForce RTX 4090

Selected aggregate means:
- Top PSNR: `SS best + cosine LR`, PSNR 51.4881, MS-SSIM 0.999878, AUC 46.135, total 15.075 s.
- Top MS-SSIM/LPIPS: `SS best + final color solve`, PSNR 50.9996, MS-SSIM 0.999880, LPIPS 0.000036, total 14.594 s.
- Top AUC/convergence: `GaussianImage fixed`, AUC 46.398, PSNR 47.3401, total 13.785 s.
- Fastest exact method: `SS best + L1 only`, total 11.361 s, PSNR 47.8830.

## Native official GaussianImage

Report: [summary.md](../../../results/native_gaussianimage_168k_coco4_s160_10k/summary.md)

Artifacts:
- [metrics.json](../../../results/native_gaussianimage_168k_coco4_s160_10k/metrics.json)
- [metrics.csv](../../../results/native_gaussianimage_168k_coco4_s160_10k/metrics.csv)
- [paired native vs StructSplat summary](../../../results/native_gaussianimage_168k_coco4_s160_10k/paired_native_vs_structsplat_summary.csv)
- [resource summary](../../../results/native_gaussianimage_168k_coco4_s160_10k/native_gaussianimage_168k_coco4_s160_10k.resources.json)
- [environment provenance](../../../results/native_envs/gaussianimage_official_retry2/provenance/verify.json)

Protocol:
- Official GaussianImage checkout `d53393bee7c9fbb24e3510614e3ff2c85b8fbbc1`
- Official `gsplat` submodule `bcca3ecae966a052e3bf8dd1ff9910cf7b8f851d`
- Isolated Python 3.10 environment with torch 2.0.0+cu118
- 4 COCO images, max side 160
- 2 seeds
- 5,376 Gaussians, 10,000 requested iterations
- Centrally recomputed PSNR, SSIM, MS-SSIM, LPIPS, AUC, fit/render timing

Result:
- 8/8 cells completed successfully.
- Mean PSNR 35.6571.
- Mean MS-SSIM 0.996462.
- Mean LPIPS 0.009539.
- Mean AUC 32.6264.
- Mean fit time 6.392 s.
- Median render FPS mean 4,412.3.
- Float32 parameter bpp mean 78.591.

Paired against `structsplat_best_default`:
- Native GaussianImage PSNR delta: -13.1420.
- Native GaussianImage MS-SSIM delta: -0.003289.
- Native GaussianImage AUC delta: -12.0421.
- Native GaussianImage fit-time gain: +8.2882 s.
- Relation: tradeoff.

Resource telemetry:
- Wall time: 64.642 s
- Max child RSS: 1,281,376 KiB
- Peak GPU memory: 3,229 MiB on NVIDIA GeForce RTX 4090

Environment setup telemetry:
- Wall time: 160.635 s
- Max child RSS: 2,626,584 KiB
- Peak GPU memory: 2,488 MiB

## Native AIR

Report: [summary.json](../../../results/native_air_coco4_s256_gsplat100/output/air_coco4_s256_inputs/summary.json)

Artifacts:
- [resource summary](../../../results/native_air_coco4_s256_gsplat100/native_air_coco4_s256_gsplat100.resources.json)
- [stdout log](../../../results/native_air_coco4_s256_gsplat100/native_air_coco4_s256_gsplat100.stdout.log)
- [stderr log](../../../results/native_air_coco4_s256_gsplat100/native_air_coco4_s256_gsplat100.stderr.log)

Protocol:
- Native AIR checkpoint `AIR/checkpoints/checkpoints/ps_7.pt`
- 4 COCO inputs resized to max side 256
- AIR was not run at max side 160 because its MS-SSIM implementation asserts a minimum small-side size above that setting.

Result:
- 4/4 images completed successfully.
- Final-stage mean PSNR 25.254.
- Final-stage mean MS-SSIM 0.96039.
- Final-stage mean LPIPS 0.21825.
- Mean Gaussian count 3,511.25.
- Mean inference time 37.007 ms.
- Quantized PSNR 24.838, MS-SSIM 0.957, LPIPS 0.227, bpp 4.328.

Resource telemetry:
- Wall time: 9.851 s
- Max child RSS: 3,772,856 KiB
- Peak GPU memory: 4,784 MiB on NVIDIA GeForce RTX 4090

## Verification

Targeted tests:

```text
python -m pytest -q tests/test_storage_budget_compare.py tests/test_fair_density_control_compare.py
36 passed in 1.28s
```

Successful benchmark wrappers all recorded `returncode: 0` and `interrupted: false`.

Known caveats:
- AIR is reported separately at max side 256 because native AIR rejected the 160-side input lane via its MS-SSIM size assertion.
- Native GaussianImage exports a terminal float parameterization, not a codec bitstream; bpp is float32 parameter bpp.
- The common-harness GaussianImage/Image-GS rows are repo-inspired analogues inside StructSplat's fitter/renderer, not native external-repo execution.
- Installing native AIR dependencies changed the active base environment's editable `gsplat` to AIR's vendored `gsplat 1.0.0`; the official GaussianImage run used its isolated environment and was unaffected.
