# Task index

Status: `done` (reference implemented + validated) · `partial` · `todo`. Areas: CORE, INIT, FIT,
HIER, BENCH, ABL, FF, COMP, PORT. Work items are picked up via the `task-workflow` skill.

| ID | Title | Status | Depends on |
|----|-------|--------|-----------|
| CORE-001 | Differentiable reference rasterizer (normalized weighted sum) | done | — |
| CORE-002 | RS Gaussian parameterization + conics | done | — |
| INIT-001 | Structure tensor: energy, orientation, flat/edge/corner | done | — |
| INIT-002 | Density field (image + residual) | done | INIT-001 |
| INIT-003 | Anisotropic blue-noise sampling (WSE + metric) | partial | INIT-001, INIT-002 |
| INIT-004 | Flanking vs on-edge placement + threshold study | partial | INIT-003 |
| FIT-001 | Adam fitter (L1+SSIM), PSNR history, iters-to-target | done | CORE-001/002 |
| HIER-001 | Progressive pyramid (residual-driven densification) | partial | INIT-002, FIT-001 |
| BENCH-001 | Metric protocol (PSNR/MS-SSIM/LPIPS + iters-to-target) | done | FIT-001 |
| ABL-001 | Init-strategy x budget sweep (the core experiment + fitness) | partial | INIT-003/004, BENCH-001 |
| ABL-002 | Full stage-combination search | partial | CORE, INIT, FIT, HIER, BENCH |
| FF-001 | Feed-forward init predictor (warm-start) | todo | INIT-003, FIT-001 |
| GEN-001 | Generative 2D Gaussians via SDS distillation (no dataset) | todo | CORE-001, ADR-0006 |
| COMP-001 | Quantization + entropy/VQ codec (rate-distortion) | todo | FIT-001 |
| PORT-001 | CUDA tile rasterizer → IntrinsicEngine RHI pass | todo | CORE-001 |

"done (reference)" means a correct, validated NumPy/PyTorch version exists — not that it is the
performant or final form. Perf and scale live in PORT-001.
