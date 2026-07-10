# FIT-016 low-pass loss-target curriculum screen

Date: 2026-07-10

## Protocol

- Four pinned COCO images, max-side 160, seeds 0/1, exact CUDA renderer, and LPIPS.
- Final/start counts 640/320, five growth waves, 500 optimizer steps.
- Control: `structsplat_best_checkpoint`.
- Candidate: the same checkpoint policy and fit configuration, except
  `loss_target_downsample=2` and `loss_target_full_frac=0.10`.
- The candidate initially optimizes a 2x area-low-pass target and cosine-blends to the full target
  during the first 10% of the global fit horizon. All reporting, target hits, early stopping,
  checkpoint scoring, residual growth, and final export use the full-resolution target.
- The direct audit rejects a pair unless checkpoint policy, requested/actual counts, and all
  `FitConfig` fields except the two treatment fields match exactly. Intervals resample source
  images after averaging correlated seeds.

## Result and decision

The candidate failed the preregistered short-horizon guard. Relative to checkpoint control it lost
0.1645 dB selected PSNR (95% image-bootstrap CI [-0.2856,-0.0677]), 0.00068 MS-SSIM, and 0.0716
PSNR AUC. LPIPS gain was -0.0030 with an interval crossing zero. Its unselected terminal endpoint
lost 0.8949 dB PSNR. Stop before the planned 5,000-step confirmation, leave the curriculum default
off, and do not tune another warmup on this same guard.

## Durable files

- `lowpass_vs_checkpoint_summary.csv`: exact direct-treatment aggregate and intervals.
- Full local artifact: `results/fit016_lowpass_coco4_500/`.
- Source/task: `src/structsplat/fit.py`, `benchmarks/fair_density_control_compare.py`, and
  `tasks/FIT-016-low-pass-loss-target-curriculum.md`.

Repository commit recorded by the run: `4b1212958f5b19ed6e016e87decb5099df4920aa` with the
FIT-016 implementation present as a tracked diff. Runtime: Python 3.12.9, Torch 2.9.0+cu128,
CUDA 12.8, NVIDIA driver 590.48.01, RTX 3050.
