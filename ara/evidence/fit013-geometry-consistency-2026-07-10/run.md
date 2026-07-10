# FIT-013 geometry-consistent regularization

- Date: 2026-07-10
- Source idea: Structure-Guided Allocation, arXiv:2512.24018
- Implementation: cached channel-wise Sobel target gradients, rendered-gradient discrepancy,
  weights 0.015/0.030/0.060, and intermittent every-two/every-four controls
- COCO proxy: four images x seeds 0/1, cap 640, 500 steps, max-side 160, LPIPS enabled
- Retained dense arm: weight 0.015 gained +0.1887 dB PSNR, +0.00086 MS-SSIM,
  +0.1132 AUC, and +0.0047 LPIPS gain; mean fit-time gain was -0.0888 s
- Kodak check: four held-out images, 768x512, cap 2000, 1500 steps, seed 0; dense 0.015
  gained +0.1998 dB PSNR, +0.01016 MS-SSIM, +0.1756 AUC, and +0.01533 LPIPS gain,
  but cost 2.189 s fit time
- Cadence result: every-two recovered time but was quality-neutral on Kodak; every-four was weaker
- Dead end: explicit sliced Sobel arithmetic was slower than cached grouped convolutions on CUDA
- Decision: keep shipped weight 0; retain dense 0.015 as an opt-in quality candidate pending
  randomized multi-seed/multi-density confirmation and synchronized timing
- Validation: focused group passed 102 tests, changed-file Ruff passed, full suite passed 326 tests

Artifacts are split into `proxy/`, `schedule/`, and `kodak4/`, each with config, metrics,
dominance audit, summary, and HTML index.
