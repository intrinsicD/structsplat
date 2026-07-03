# Task index

Status: `done` (reference implemented + validated) · `partial` · `todo`. Areas: CORE, INIT, FIT,
HIER, BENCH, ABL, FF, COMP, PORT, MERGE, DOCS. Work items are picked up via the `task-workflow`
skill.

| ID | Title | Status | Depends on |
|----|-------|--------|-----------|
| CORE-001 | Differentiable reference rasterizer (normalized weighted sum) | done | — |
| CORE-002 | RS Gaussian parameterization + conics | done | — |
| CORE-003 | Edge-aware render support window (off-image support + tile waste) | done | CORE-001 |
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
| COMP-001 | Quantization + entropy/VQ codec (rate-distortion) | partial | FIT-001 |
| PORT-001 | CUDA tile rasterizer → IntrinsicEngine RHI pass | todo | CORE-001 |
| MERGE-001 | Integrate Claude core optimizations and Codex stage search into main | todo | CORE, INIT, FIT, HIER, BENCH, ABL, COMP |
| CORE-004 | Renderer + GaussianField correctness fixes (CUDA N=0, int-cast UB, aliasing, dilation) | done | CORE-001/002 |
| CORE-005 | Reference renderer memory bound + C0-continuous support cutoff | todo | CORE-003, CORE-004 |
| INIT-005 | Init-math robustness, flanking unification, WSE test coverage | done | INIT-003 |
| INIT-006 | Init-time performance (quadtree, spacing, run-lengths, pair discovery) | todo | INIT-003, INIT-005 |
| FIT-002 | Fitter correctness (split colors, opacity pruning, history pairing) | done | FIT-001, CORE-004 |
| FIT-003 | Fit-loop speed (device-side targets, SSIM hygiene, fused SSIM) | todo | FIT-001, BENCH-001 |
| FIT-004 | Densification & convergence upgrades (fp-growth, relocation, NMS) | todo | FIT-002, BENCH-002 |
| HIER-002 | Pyramid bookkeeping (iteration accounting, budgets, schedules) | done | HIER-001, FIT-001 |
| BENCH-002 | Benchmark harness experimental-validity fixes (**gates all sweeps**) | todo | — |
| BENCH-003 | Benchmark script consolidation + documentation | todo | BENCH-002 |
| ABL-003 | Bisect the undiagnosed −0.794 dB flagship regression | todo | — |
| ABL-004 | Killer controls + full ABL-001 run + committed evidence | todo | BENCH-002, ABL-003, FIT-004 |
| COMP-002 | Codec / metrics / CLI correctness and protocol fixes | done | COMP-001, FIT-001 |
| COMP-003 | Compression-ratio ladder (scale ranges → planes → LSQ → VQ → entropy) | todo | COMP-002, BENCH-002 |
| DOCS-001 | Docs-sync backfill (stale status, missing ADRs, ara scaffold) | todo | — |

"done (reference)" means a correct, validated NumPy/PyTorch version exists — not that it is the
performant or final form. Perf and scale live in PORT-001.

## Suggested order (from the 2026-07-03 repo review)

CORE-004/FIT-002/HIER-002/COMP-002/INIT-005 fix confirmed bugs and are independent of each
other. **BENCH-002 and ABL-003 gate the science** — run them before any new sweep or tuning.
Then ABL-004 (the actual experiment, with evidence committed), and only after that the
improvement tracks: FIT-003/INIT-006 (speed), FIT-004/CORE-005 (quality/convergence),
COMP-003 (rate), BENCH-003 + DOCS-001 (hygiene, anytime).
