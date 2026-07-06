# FIT-006: Frequency-violation densification

**Status: todo.** Structure-aware split mode aligned with the existing tensor pipeline.

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
- [ ] `split_mode="freq_violation"` implemented behind config/CLI.
- [ ] Logs mean/max violation score and selected split-axis histogram for benchmark analysis.
- [ ] Unit test verifies the mode selects an oversized edge Gaussian over a same-residual smooth
      Gaussian in a synthetic image.
- [ ] Works with fixed final N and the existing budgeted wave protocol.
- [ ] Benchmark slice vs `absgrad_wave`, `ranked_wave`, and `residual_tensor_add` on the difficult
      image subset, reporting final PSNR, AUC, and post-split loss dip.
- [ ] Stage-search axis added if the slice is competitive.

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/tensor.py` or tensor helpers,
`src/structsplat/config.py`, `benchmarks/stage_search.py`, `tests/test_fit_dynamics.py`.

## Depends on
INIT-001, FIT-004, BENCH-002.
