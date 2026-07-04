# ABL-004 Exact-CUDA Calibration

Date: 2026-07-04

Command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
python -m benchmarks.ablation results/datasets/abl004/kodak24/kodim01.png \
  --budgets 2000 \
  --strategies random aniso_flanking floyd_steinberg density_random random_relocate \
  --seeds 0 \
  --iters 1500 \
  --target-psnr 35 \
  --max-side 768 \
  --renderer cuda \
  --device cuda \
  --outdir results/abl004_cuda_calibration \
  --max-new-cells 5 \
  --no-plots
```

Environment note: the exact CUDA extension loads in this workspace only with the system
`libstdc++.so.6` preload; without it, conda's `libstdc++` lacks `CXXABI_1.3.15`.

Results:

| Arm | Init s | Fit s | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|
| random | 0.009 | 22.240 | 21.9858 | 0.85278 |
| aniso_flanking | 0.407 | 20.589 | 22.5123 | 0.85336 |
| floyd_steinberg | 0.513 | 22.309 | 22.3843 | 0.85971 |
| density_random | 0.043 | 20.983 | 22.5100 | 0.85058 |
| random_relocate | 0.003 | 21.910 | 22.2888 | 0.85202 |

Mean fit time: 21.606 s. Mean init+fit time: 21.801 s.

Planning estimate:

- Current full matrix: 28 images x 4 budgets x 3 seeds x 11 arms = 3,696 cells.
- Flat 2k extrapolation: about 0.93 GPU-days.
- Linear budget-scaled extrapolation over {2k, 5k, 10k, 20k}: about 4.31 GPU-days.
- This is now feasible as a scheduled/job-queue run, but still too long for interactive work.
