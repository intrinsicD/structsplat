# FIT-006: Frequency-violation densification

**Status: done.** Structure-aware split mode aligned with the existing tensor pipeline.

## Context
Current split/add modes use residual, support activity, footprint, and AbsGrad-style mean-gradient
signals. They do not explicitly ask whether a Gaussian is too large for the local image frequency
along either principal axis. Recent structure-aware densification work combines tensors, Laplacian
scale-space cues, and anisotropic splitting to identify this exact failure mode.

## Goal
Add `split_mode="freq_violation"` that splits Gaussians whose local support violates the
image-frequency scale implied by structure tensor and Laplacian evidence.

## Approach
1. Reuse the structure-tensor machinery to estimate dominant orientation and anisotropic local
   frequency/edge width near each active Gaussian.
2. Add a Laplacian or DoG scale-space probe under each Gaussian support.
3. Score axis-wise violations: parent scale is too large relative to local wavelength or edge
   width, especially along the high-curvature/frequency direction.
4. Split along the offending axis, carrying optimizer state and function-preservation semantics.

## Acceptance criteria
- [x] `split_mode="freq_violation"` implemented behind config/CLI.
- [x] Logs mean/max violation score and selected split-axis histogram for benchmark analysis.
- [x] Unit test verifies the mode selects an oversized edge Gaussian over a same-residual smooth
      Gaussian in a synthetic image.
- [x] Works with fixed final N and the existing budgeted wave protocol.
- [x] Benchmark slice vs `absgrad_wave`, `ranked_wave`, and `residual_tensor_add` on the difficult
      image subset, reporting final PSNR, AUC, and post-split loss dip.
- [x] Stage-search axis added if the slice is competitive.

## Notes

- 2026-07-06: Added `split_mode="freq_violation"`. The score samples residual, target gradient
  magnitude, absolute Laplacian, local gradient-axis alignment, and current Gaussian scale. It
  selects the most violating local Gaussian axis and reuses function-preserving duplication with
  optimizer-state carry. Split events log `freq_violation_score_mean/max`,
  `freq_violation_axis0_count`, `freq_violation_axis1_count`, and
  `freq_violation_freq_mean`.
- 2026-07-06 smoke slice recorded in
  `ara/evidence/fit006-frequency-violation-smoke-2026-07-06/run.md`: on the difficult-four
  targets at max-side 64, initial 64 -> max 80 Gaussians, 60 iterations, `freq_violation` beat
  `ranked_wave` and `absgrad_wave` on mean PSNR/AUC and won individual images, but remained behind
  `residual_tensor_add` on mean final PSNR/AUC. Keep it as a stage-search axis, not a default.

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/tensor.py` or tensor helpers,
`src/structsplat/config.py`, `benchmarks/stage_search.py`, `tests/test_fit_dynamics.py`.

## Depends on
INIT-001, FIT-004, BENCH-002.
