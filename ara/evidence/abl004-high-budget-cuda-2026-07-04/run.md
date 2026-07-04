# ABL-004 high-budget exact-CUDA check

Date: 2026-07-04

Purpose: run the previously tiny two-iteration high-budget smoke as a real 1500-iteration
single-image check, keeping the same image, two arms, budget, and seed.

Command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
python -m benchmarks.ablation results/datasets/abl004/kodak24/kodim01.png \
  --budgets 20000 \
  --strategies aniso_flanking floyd_steinberg \
  --seeds 0 \
  --iters 1500 \
  --target-psnr 35 \
  --max-side 768 \
  --renderer cuda \
  --device cuda \
  --outdir results/abl004_high_budget_cuda_1500 \
  --no-plots
```

Plots were generated afterwards from the completed `ablation.json` rows without rerunning the
fit.

## Result

| strategy | budget | seed | PSNR | SSIM | MS-SSIM | init s | fit s | init+fit s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aniso_flanking | 20000 | 0 | 30.7510 | 0.90900 | 0.98311 | 8.7713 | 35.1502 | 43.9215 |
| floyd_steinberg | 20000 | 0 | 30.7803 | 0.91051 | 0.98379 | 0.5295 | 31.3819 | 31.9113 |

Delta, `floyd_steinberg - aniso_flanking`:

- PSNR: +0.0293 dB
- SSIM: +0.00151
- MS-SSIM: +0.00068
- init+fit time: -12.0102 s

Neither arm reached the 35 dB target within 1500 iterations.

## Interpretation

This is a single-image, single-seed high-budget check, so it is not decision-grade ABL-004
evidence by itself. It weakens the expectation that anisotropic WSE flanking will dominate
the Floyd-Steinberg killer control at high budget: on this cell, Floyd-Steinberg is slightly
better on all recorded quality metrics and faster overall. The staged ABL-004 protocol should
therefore keep Floyd-Steinberg as a required comparator through confirmation.
