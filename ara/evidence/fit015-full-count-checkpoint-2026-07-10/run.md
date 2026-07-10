# FIT-015 full-count checkpoint evidence

Date: 2026-07-10

## Protocol

- Four pinned COCO images, decoded at max-side 160.
- Final cap 640, start 320, five tensor-aware growth waves, exact CUDA renderer.
- Pinned StructSplat recipe: anisotropic on-edge WSE, feature cap `12@160`, L1 + 0.3 SSIM.
- Seeds 0/1 at both 500 and 5,000 optimizer steps; LPIPS enabled.
- Candidate: post-transition `best_psnr_final_count` selection at the logging cadence. Only a
  checkpoint whose Gaussian count equals terminal N may be restored; terminal is always a
  candidate.
- The decisive table is within-trajectory: selected vs terminal metrics from the same run/count.
  Independent CUDA candidate/default fits can diverge before a policy activates and are retained
  only as a non-causal comparison.

## Results

At 5,000 steps, 7/8 runs restored an earlier full-count state. Mean within-trajectory gains were
+0.7702 dB PSNR, +0.00669 SSIM, +0.00892 MS-SSIM, and +0.0076 LPIPS (positive means lower LPIPS).
At 500 steps, 7/8 runs retained the terminal state; the sole restoration produced a +0.0066 dB
mean PSNR gain overall, with tiny SSIM/MS-SSIM/LPIPS tradeoffs.

The policy therefore repairs long-horizon terminal regression without changing count or training
trajectory/AUC. It remains opt-in pending multiple budgets/resolutions and an explicit decision
about per-image PSNR-vs-perceptual tradeoffs.

## Files

- `long_checkpoint_selection.csv`, `long_default_dominance.csv`, `long_summary.md`,
  `long_config.json`
- `short_checkpoint_selection.csv`, `short_default_dominance.csv`, `short_summary.md`,
  `short_config.json`
- Source implementation: `src/structsplat/fit.py`, `src/structsplat/config.py`,
  `benchmarks/fair_density_control_compare.py`
