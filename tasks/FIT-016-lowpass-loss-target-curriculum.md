# FIT-016: Coarse-to-full loss-target curriculum

## Status

Implemented and screened; rejected by the preregistered 500-step guard. Keep default-off and do
not spend the planned 5,000-step or difficult-Kodak confirmation budgets on this exact schedule.

## Motivation

LIG/Level-of-Gaussian orders learning from a coarse image level to a fine residual level. Its
separate additive residual fields do not transfer directly to StructSplat's normalized compositor,
but the frequency-ordering idea is compatible with the existing fitter. Local pyramid evidence
also favors a short coarse phase rather than equal stage horizons.

Primary references: [LIG paper](https://ojs.aaai.org/index.php/AAAI/article/download/33193/35348)
and [official implementation](https://github.com/HKU-MedAI/LIG).

## Candidate

`structsplat_best_checkpoint_lowpass2x_f10` differs from
`structsplat_best_checkpoint` in exactly two `FitConfig` fields:

- `loss_target_downsample=2`
- `loss_target_full_frac=0.10`

For full target `I`, precompute `B = Up_bilinear(Down_area_2(I))`. At global iteration `t` in a
nominal schedule of length `T`:

`u = clamp(t / (0.10*T), 0, 1)`, `w = 0.5*(1-cos(pi*u))`,
`loss_target = (1-w)*B + w*I`.

Only the pixel and SSIM objective use `loss_target`. Full-resolution `I` remains authoritative for
PSNR/AUC/target hits, early stopping, checkpoint scoring, residual/tensor growth, relocation,
adaptive decisions, final metrics, and the exported representation. Tensor loss weights, when
used, are also prepared from `I`.

## Safety and bookkeeping

- Factor 1 plus fraction 0 is the exact neutral behavior.
- The weight uses `sched_offset/sched_total`, so pyramid stages do not restart the curriculum.
- History records `loss_target_full_weight`; outputs record both resolved fields.
- Geometry consistency and color solving fail closed until their scheduled-target semantics are
  defined.
- Prune/split/relocate/adaptive events, finer pyramid-level insertions, and early stopping fail
  closed if they can occur before the full target boundary. The pinned five-wave schedule first
  grows near one-sixth of the horizon, after the 10% transition.
- The fair harness keys the resolved curriculum/checkpoint fields and writes a direct causal audit:
  `lowpass_vs_checkpoint.csv` plus an image-clustered bootstrap summary. The ordinary default
  dominance table remains useful but conflates curriculum and checkpoint effects.

## Preregistered screen

1. Short guard: COCO4, max-side 160, N=640/start 320, 500 steps, seeds 0/1, checkpoint control
   versus candidate (16 cells).
2. Long proxy: same axes at 5,000 steps (16 cells).
3. Only if the proxy survives: difficult Kodak4, max-side 768, N={2k,5k}, 1,500 steps, seeds 0/1
   (32 cells).

Both proxy arms run in the same process on `renderer=cuda`, `device=cuda`, with LPIPS explicitly
enabled. Runtime provenance includes Python, Torch/CUDA/cuDNN, metric-package versions, CUDA device
UUID/properties, and NVIDIA driver; resume therefore cannot mix hardware or LPIPS environments.

```bash
ITERS=500
LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m benchmarks.fair_density_control_compare \
    --images \
      tests/test_images/COCO_train2014_000000000009.jpg \
      tests/test_images/COCO_train2014_000000000025.jpg \
      tests/test_images/COCO_train2014_000000000030.jpg \
      tests/test_images/COCO_train2014_000000000034.jpg \
    --outdir "results/fit016_lowpass_coco4_${ITERS}" \
    --methods \
      structsplat_best_checkpoint \
      structsplat_best_checkpoint_lowpass2x_f10 \
    --budgets 640 --max-side 160 --iters "$ITERS" --seeds 0 1 \
    --start-fraction 0.5 --growth-waves 5 --renderer cuda --device cuda \
    --render-chunk 512 --pixel-loss l1 --ssim-weight 0.3 \
    --target-psnr 35 --target-psnrs 22 24 26 28 30 32 --lpips --resume
```

The planned long stage used the same command with `ITERS=5000`; it was gated off by the short
result.

Proxy survival requires long-proxy selected PSNR gain >= +0.10 dB, no material MS-SSIM regression
(>0.001), LPIPS regression (>0.005), or AUC loss (>0.10 dB), fit-time overhead <=3%, and no more
than 0.05 dB PSNR loss in the 500-step guard. Report selected and terminal endpoints separately;
promotion still requires the project's stricter multi-metric default-dominance gate.

## Evidence (2026-07-10)

Artifact: `results/fit016_lowpass_coco4_500/`.

All 16 short-guard cells completed under the exact command above. Candidate gain over the
checkpoint control (positive is better) was:

| Endpoint | PSNR | MS-SSIM | AUC | LPIPS gain | Fit-time gain |
|---|---:|---:|---:|---:|---:|
| selected checkpoint | -0.1645 dB | -0.00068 | -0.0716 dB | -0.0030 | +0.065 s |
| unselected terminal | -0.8949 dB | -0.00064 | n/a | -0.0028 | n/a |

The image-clustered 95% CI for selected PSNR was `[-0.2856,-0.0677]` dB, wholly below zero;
selected MS-SSIM and AUC intervals were also wholly negative. The terminal PSNR mean was distorted
by a severe low-pass-arm collapse on one image/seed; terminal-count checkpoint selection rescued
most of it, but the selected endpoint still exceeded the allowed 0.05 dB short-guard loss.

## Decision

Reject `lowpass2x_f10` and stop at stage 1. The mechanism orders frequencies as intended but harms
early convergence and endpoint quality under StructSplat's already structured initialization and
five-wave growth. Do not tune the transition fraction on the same four-image guard; that would
turn the preregistered test into an adaptive search. A materially different multiscale mechanism
needs a new hypothesis and independent screen.

## Interfaces

`src/structsplat/config.py`, `src/structsplat/fit.py`, `src/structsplat/pyramid.py`,
`src/structsplat/cli.py`, `benchmarks/fair_density_control_compare.py`,
`tests/test_fit_dynamics.py`, `tests/test_pyramid.py`, `tests/test_cli.py`,
`tests/test_fair_density_control_compare.py`.

## Depends on

FIT-015, HIER-003/004, ABL-004, BENCH-002.
