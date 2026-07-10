# FIT-014 generation-density covariance filter (2026-07-10)

Implemented the GaussianImage++ birth-cohort rule
`s=min(300, H*W/(alpha*N_after)) px^2` across field lifecycle, normalized/custom-CUDA/gsplat
render paths, relocation/pruning/growth, NPZ, codec materialization, QAT/rate, pyramid, and CLI.
Default mode remains `none`.

The paired screen used COCO4, max-side 160, cap 640/start 320, 500 steps, seeds 0/1, exact CUDA,
and the pinned StructSplat best-default geometry/growth/loss. Candidate gains over default:

| alpha | PSNR | proxy MS-SSIM | AUC | LPIPS gain |
|---:|---:|---:|---:|---:|
| `9*pi` | -2.3587 dB | -0.01087 | -1.5616 | -0.1033 |
| `18*pi` | -1.0799 dB | -0.00371 | -0.7372 | -0.0406 |
| `36*pi` | -0.4362 dB | -0.00106 | -0.2883 | -0.0127 |

All persistent-filter strengths lost every aggregate quality/convergence metric; weaker filters
approached the unfiltered baseline monotonically. The native rule does not transfer as a default
to StructSplat's normalized compositor/current WSE+feature-cap recipe. Keep it opt-in and do not
fund larger confirmation of the persistent form.

Local HTML artifact: `results/structsplat_generation_caf_proxy/index.html`.
