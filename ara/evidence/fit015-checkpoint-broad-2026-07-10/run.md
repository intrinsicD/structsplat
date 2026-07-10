# FIT-015 broad checkpoint-selection confirmation

Date: 2026-07-10

## Protocol

- Kodak `kodim01`, `kodim07`, `kodim13`, and `kodim19`.
- Max-side {160,240,320}; final counts {1280,2560,5120}; seeds {0,1}; 5,000 steps.
- Final/start count ratio 2:1 with five growth waves, exact CUDA renderer, LPIPS, and the pinned
  StructSplat long-fit recipe.
- Seventy-two successful trajectories. Each comparison is selected versus terminal state inside
  that same trajectory; both endpoints have the same final Gaussian count.
- Means and 95% intervals first average repeated resolution/count/seed rows within source image,
  then bootstrap the four source-image clusters with the benchmark's fixed seed.

## Result and decision

Forty of 72 trajectories selected an earlier state. Pooled gain was +0.4884 dB PSNR (95% CI
[+0.4167,+0.5304]), +0.00316 SSIM, +0.00433 MS-SSIM, and +0.00736 LPIPS (positive means lower).
The benefit is density-dependent: +1.0380 dB at N=1280, +0.3812 dB at N=2560, and +0.0458 dB at
N=5120. The max-side-240/N=5120 stratum selected no earlier state. This fails the preregistered
universal-default rule requiring at least +0.10 dB in every resolution/count stratum, while
confirming checkpoint selection as a strong sparse/moderate-density long-horizon option. Keep
`terminal` as the compute-minimal universal default.

## Durable files

- `checkpoint_selection_summary.csv`: pooled, per-resolution, and per-count image-clustered
  aggregates and intervals.
- `checkpoint_selection_matrix.csv`: all nine resolution/count cell means.
- Full local artifacts: `results/fit015_checkpoint_kodak4_r{160,240,320}_5000/`.
- Source/task: `benchmarks/fair_density_control_compare.py` and
  `tasks/FIT-015-full-count-checkpoint-selection.md`.

Runs recorded repository commit `4b1212958f5b19ed6e016e87decb5099df4920aa`; runtime was Python
3.12.9, Torch 2.9.0+cu128, CUDA 12.8, NVIDIA driver 590.48.01, RTX 3050.
