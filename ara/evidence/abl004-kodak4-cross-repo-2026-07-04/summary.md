# Cross-Repo Matrix Comparison

Matched executable comparison using StructSplat's current fitter and exact CUDA renderer for all rows.
External repositories are represented by local repo-inspired placement/growth policies, not native codec pipelines.
Seeds: 0
Aggregate rows report mean and population std over image x seed runs.

## Images

| Image | Difficulty | Source |
|---|---|---|
| kodim01 | regular | `results/datasets/abl004/kodak24/kodim01.png` |
| kodim02 | regular | `results/datasets/abl004/kodak24/kodim02.png` |
| kodim03 | regular | `results/datasets/abl004/kodak24/kodim03.png` |
| kodim04 | regular | `results/datasets/abl004/kodak24/kodim04.png` |

## Method Mapping

- **StructSplat current**: best-searched StructSplat policy in this harness: aniso_flanking, scharr/rgb tensor, WSE, two-sided colors, feature12 cap, residual_tensor_add
- **StructSplat shipped defaults**: public InitConfig/FitConfig defaults with only benchmark-control fields overridden (iterations, renderer, logging, target tracking)
- **GaussianImage**: fixed random GaussianImage-style baseline
- **GaussianImage++**: random half-budget init plus high-residual additions
- **Image-GS**: gradient-density init plus residual progressive additions
- **Instant-GI qt**: Instant-GI quadtree/Delaunay fallback; no learned checkpoint

## Overall Means

| Method | Runs | PSNR | PSNR Std | SSIM | MS-SSIM | MS-SSIM Std | LPIPS | AUC | MAE | Edge MAE | Fit s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| StructSplat current | 4 | 31.6934 | 2.7796 | 0.88172 | 0.97767 | 0.01299 | - | 28.584 | 0.01797 | 0.02473 | 0.348 | 0.474 |
| StructSplat shipped defaults | 4 | 31.6323 | 2.5915 | 0.89479 | 0.98046 | 0.00990 | - | 29.163 | 0.01699 | 0.02470 | 0.180 | 0.296 |
| GaussianImage | 4 | 28.8334 | 1.8800 | 0.85222 | 0.97133 | 0.01218 | - | 25.969 | 0.02244 | 0.03915 | 0.195 | 0.196 |
| GaussianImage++ | 4 | 26.6340 | 2.0874 | 0.77467 | 0.94307 | 0.02698 | - | 23.984 | 0.02920 | 0.05225 | 0.193 | 0.194 |
| Image-GS | 4 | 27.0637 | 2.0453 | 0.78508 | 0.94621 | 0.02546 | - | 24.516 | 0.02812 | 0.04879 | 0.197 | 0.201 |
| Instant-GI qt | 0 | - | - | - | - | - | - | - | - | - | - | - |

## Means By Resolution And Iterations

| Max side | Iterations | Method | Budget cap | PSNR | PSNR Std | MS-SSIM | MS-SSIM Std | LPIPS | AUC | Fit s | Total s |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 160 | 80 | GaussianImage | 640 | 28.8334 | 1.8800 | 0.97133 | 0.01218 | - | 25.969 | 0.195 | 0.196 |
| 160 | 80 | GaussianImage++ | 640 | 26.6340 | 2.0874 | 0.94307 | 0.02698 | - | 23.984 | 0.193 | 0.194 |
| 160 | 80 | Image-GS | 640 | 27.0637 | 2.0453 | 0.94621 | 0.02546 | - | 24.516 | 0.197 | 0.201 |
| 160 | 80 | StructSplat current | 640 | 31.6934 | 2.7796 | 0.97767 | 0.01299 | - | 28.584 | 0.348 | 0.474 |
| 160 | 80 | StructSplat shipped defaults | 640 | 31.6323 | 2.5915 | 0.98046 | 0.00990 | - | 29.163 | 0.180 | 0.296 |

## Difficult-Image Split

| Difficulty | Method | Runs | PSNR | PSNR Std | MS-SSIM | MS-SSIM Std | LPIPS | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| regular | GaussianImage | 4 | 28.8334 | 1.8800 | 0.97133 | 0.01218 | - | 0.195 |
| regular | GaussianImage++ | 4 | 26.6340 | 2.0874 | 0.94307 | 0.02698 | - | 0.193 |
| regular | Image-GS | 4 | 27.0637 | 2.0453 | 0.94621 | 0.02546 | - | 0.197 |
| regular | StructSplat current | 4 | 31.6934 | 2.7796 | 0.97767 | 0.01299 | - | 0.348 |
| regular | StructSplat shipped defaults | 4 | 31.6323 | 2.5915 | 0.98046 | 0.00990 | - | 0.180 |

## Winners

| Max side | Iterations | Best PSNR | Best MS-SSIM | Best LPIPS | Fastest fit |
|---:|---:|---|---|---|---|
| 160 | 80 | StructSplat current (31.693) | StructSplat shipped defaults (0.98046) | - | StructSplat shipped defaults (0.180s) |

## Errors

| Row | Error |
|---|---|
| kodim01 160px 80it Instant-GI qt | `RuntimeError: Instant-GI methods require STRUCTSPLAT_INSTANT_GI=/path/to/quard_image.py; unset -> this method is skipped.` |
| kodim02 160px 80it Instant-GI qt | `RuntimeError: Instant-GI methods require STRUCTSPLAT_INSTANT_GI=/path/to/quard_image.py; unset -> this method is skipped.` |
| kodim03 160px 80it Instant-GI qt | `RuntimeError: Instant-GI methods require STRUCTSPLAT_INSTANT_GI=/path/to/quard_image.py; unset -> this method is skipped.` |
| kodim04 160px 80it Instant-GI qt | `RuntimeError: Instant-GI methods require STRUCTSPLAT_INSTANT_GI=/path/to/quard_image.py; unset -> this method is skipped.` |

## Caveat

This is a matched reference-policy benchmark. It does not exercise each repository's native CUDA renderer, entropy coder, or checkpointed learned components.
Visual grids are under `grids/`; per-row reconstructions are under `reconstructions/`.
