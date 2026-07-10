# FIT-015: Full-count checkpoint selection

## Status
Implemented and confirmed across three resolutions and three density budgets. Keep the shipped
policy `terminal`; use `best_psnr_final_count` as an opt-in long-horizon quality policy,
particularly at sparse/moderate density. Saturated high-density cells do not justify making its
extra checkpoint evaluations the unconditional default.

## Problem
At 5,000 iterations the pinned growth recipe often reaches its best full-budget reconstruction
before the terminal optimizer step. A naive argmax over the existing history is invalid because
that history records pre-step renders/counts, while the returned field is post-step and may have
passed through a growth/prune/relocation transition.

## Implementation

- Added `FitConfig.checkpoint_policy={terminal,best_psnr_final_count}` and CLI exposure.
- The opt-in policy evaluates a separate post-transition stream at `log_every`, after the optimizer
  step and all count-changing/on-split transitions.
- It retains the best state per Gaussian count and may restore only a state whose count equals the
  trajectory terminal count. The terminal state is always a candidate.
- Legacy convergence history, AUC, iteration count, target hits, and fit timing remain unchanged.
- Results expose terminal and selected iteration/count/PSNR/SSIM/MS-SSIM/LPIPS separately.
- Scheduled support fade, final-only color solve, and multi-stage schedule offsets fail closed for
  this policy until their state semantics are explicitly defined.
- The fair harness adds `structsplat_best_checkpoint` and writes `checkpoint_selection.csv`, a
  within-trajectory/same-count audit that is robust to independent CUDA trajectory divergence.

## Evidence (2026-07-10)

- COCO4 x seeds 0/1, max-side 160, N=640, 5,000 steps:
  `results/structsplat_checkpoint_5000_two_seed/`.
  Seven of eight runs selected an earlier full-count state. Same-trajectory mean gains were
  +0.7702 dB PSNR, +0.00669 SSIM, +0.00892 MS-SSIM, and +0.0076 LPIPS (positive means lower).
- Matching 500-step guard: `results/structsplat_checkpoint_500_two_seed/`.
  Seven of eight runs retained the terminal state; mean gain was only +0.0066 dB PSNR, with tiny
  SSIM/MS-SSIM/LPIPS tradeoffs. The policy therefore fixes a long-horizon failure rather than the
  short-budget proxy.
- Independent default-vs-candidate cells remain nondeterministic under the atomic CUDA renderer;
  they are reported but are not used to attribute the selection effect. The within-trajectory
  table is the causal evidence.
- Broader Kodak4 confirmation: 4 images x max-side {160,240,320} x N={1280,2560,5120} x seeds
  {0,1}, 5,000 steps (72 trajectories), under:
  `results/fit015_checkpoint_kodak4_r{160,240,320}_5000/`.
  Forty of 72 trajectories selected an earlier terminal-count state. Image-clustered pooled mean
  gains were +0.4884 dB PSNR (95% CI [+0.4167,+0.5304]), +0.00316 SSIM, +0.00433 MS-SSIM, and
  +0.00736 LPIPS (positive means lower). Every pooled metric interval was positive.

| Stratum | Earlier selected | PSNR gain | 95% image-bootstrap CI | MS-SSIM gain | LPIPS gain |
|---|---:|---:|---:|---:|---:|
| max-side 160 | 10/24 | +0.0977 dB | [+0.0307,+0.1647] | +0.00023 | +0.00065 |
| max-side 240 | 11/24 | +0.6337 dB | [+0.5258,+0.7309] | +0.00490 | +0.00959 |
| max-side 320 | 19/24 | +0.7337 dB | [+0.5873,+0.8779] | +0.00787 | +0.01184 |
| N=1280 | 22/24 | +1.0380 dB | [+0.8806,+1.2154] | +0.01046 | +0.01636 |
| N=2560 | 12/24 | +0.3812 dB | [+0.2688,+0.5109] | +0.00233 | +0.00518 |
| N=5120 | 6/24 | +0.0458 dB | [+0.0007,+0.0914] | +0.00021 | +0.00054 |

The effect shrinks monotonically with density. In the saturated max-side-240/N=5120 stratum the
terminal state won all eight runs; several 160px high-density cells also showed only zero/tiny
PSNR benefit and approximately 1e-5 SSIM tradeoffs. This fails the preregistered >=+0.10 dB gain
in every resolution-density stratum required for a universal default, while strongly validating
the policy as an opt-in sparse/moderate-density long-horizon safeguard.

## Decision
Do not replace `terminal` universally. The broader result resolves the pending question: checkpoint
selection is strongly beneficial at sparse/moderate density and safe but often inactive after
saturation. Recommend `--checkpoint-policy best_psnr_final_count` for long-horizon quality runs
where the extra logged evaluations/snapshots are acceptable; retain `terminal` as the general
compute-minimal default.

## Interfaces
`src/structsplat/config.py`, `src/structsplat/fit.py`, `src/structsplat/cli.py`,
`benchmarks/fair_density_control_compare.py`, `tests/test_fit_dynamics.py`,
`tests/test_fair_density_control_compare.py`.
