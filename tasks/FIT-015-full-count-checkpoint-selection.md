# FIT-015: Full-count checkpoint selection

## Status
Implemented and screened. Keep the shipped/default policy `terminal`; use
`best_psnr_final_count` as an opt-in long-horizon quality policy pending broader confirmation.

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

## Decision
Do not replace `structsplat_best_default` yet. The long-horizon endpoint improvement is strong,
but a general default promotion still needs multiple budgets/resolutions and a policy for
PSNR-selected checkpoints that slightly trade SSIM/LPIPS on individual images. Recommend
`--checkpoint-policy best_psnr_final_count` for current long-horizon quality runs.

## Interfaces
`src/structsplat/config.py`, `src/structsplat/fit.py`, `src/structsplat/cli.py`,
`benchmarks/fair_density_control_compare.py`, `tests/test_fit_dynamics.py`,
`tests/test_fair_density_control_compare.py`.
