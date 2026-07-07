# INIT-008 Feature-Relative Scale Caps

Date: 2026-07-07

## Change

Implemented `scale_cap_mode="feature_rel"` as an opt-in per-Gaussian scale cap. Init-time NumPy
math now carries a local feature scale alongside the existing init spacing:

- WSE/density samplers: density-derived local radius, with the scale-cap estimate not clipped by
  the sampler's large-radius guard.
- Quadtree aggregate samples: quadtree leaf side.
- Fallback/direct helper use: geometric mean of the initialized scales.

The cap is anisotropic: detail rows cap `s_along <= gamma_along * r_i` and
`s_across <= gamma_across * r_i`. Sparse/high-radius rows stay loose by default. Fit-time
enforcement reuses the existing `GaussianField.scale_max` clamp path and densification inherits
nearest caps.

## Tests

```bash
python -m pytest tests/test_init_stages.py tests/test_fit_dynamics.py tests/test_stage_search.py -q
```

Result: 109 passed, 1 CUDA-extension warning.

```bash
python -m pytest tests/test_fair_density_control_compare.py tests/test_init_stages.py tests/test_stage_search.py -q
```

Result: 62 passed.

## Fair-Density Protocol

Exact difficult-four protocol, max-side 768, budgets 2000/5000/10000, seed 0, 1500 iterations,
exact CUDA renderer:

```bash
python -m benchmarks.fair_density_control_compare \
  --images results/datasets/abl004/kodak24/kodim01.png results/datasets/abl004/kodak24/kodim07.png results/datasets/abl004/kodak24/kodim13.png results/datasets/abl004/kodak24/kodim19.png \
  --budgets 2000 5000 10000 \
  --methods \
    structsplat_onedge_residual structsplat_onedge_residual_featurecap structsplat_onedge_residual_feature_rel \
    structsplat_onedge_tensor structsplat_onedge_tensor_featurecap structsplat_onedge_tensor_feature_rel \
    structsplat_quadtree_wse_residual structsplat_quadtree_wse_residual_featurecap structsplat_quadtree_wse_residual_feature_rel \
    structsplat_quadtree_wse_tensor structsplat_quadtree_wse_tensor_featurecap structsplat_quadtree_wse_tensor_feature_rel \
  --max-side 768 --iters 1500 --renderer cuda --render-chunk 512 \
  --outdir results/init008_feature_rel_caps_difficult4_2026_07_07 \
  --resume
```

Result: 144/144 cells completed.

## Feature-Rel vs Matching Uncapped

| Method | Budget | Runs | Mean dPSNR | Min dPSNR | Wins | Mean dAUC | Mean dMS-SSIM | Mean dFit s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| onedge residual | 2000 | 4 | -1.1165 | -1.6800 | 0/4 | -0.2809 | -0.01775 | -0.193 |
| onedge residual | 5000 | 4 | -0.2994 | -0.5501 | 1/4 | +0.0135 | -0.00593 | -0.168 |
| onedge residual | 10000 | 4 | +0.0465 | -0.1990 | 3/4 | +0.0378 | -0.00212 | -0.138 |
| onedge tensor | 2000 | 4 | -0.9041 | -1.5683 | 0/4 | -0.4800 | -0.01782 | -0.074 |
| onedge tensor | 5000 | 4 | -0.0257 | -0.1276 | 1/4 | -0.0820 | -0.00350 | -0.114 |
| onedge tensor | 10000 | 4 | +0.0471 | -0.0200 | 3/4 | -0.0518 | -0.00074 | -0.169 |
| qt-WSE residual | 2000 | 4 | -0.9264 | -2.1119 | 0/4 | -0.2891 | -0.01672 | -0.284 |
| qt-WSE residual | 5000 | 4 | -0.3503 | -0.9568 | 0/4 | -0.0709 | -0.00593 | -0.187 |
| qt-WSE residual | 10000 | 4 | +0.0411 | -0.1829 | 3/4 | +0.0573 | -0.00167 | -0.201 |
| qt-WSE tensor | 2000 | 4 | -0.7764 | -2.0270 | 1/4 | -0.2296 | -0.01584 | -0.323 |
| qt-WSE tensor | 5000 | 4 | -0.2105 | -0.4763 | 1/4 | -0.1388 | -0.00437 | -0.355 |
| qt-WSE tensor | 10000 | 4 | -0.0046 | -0.2009 | 1/4 | -0.0695 | -0.00064 | -0.186 |

Overall feature-rel vs uncapped: 48 paired cells, mean dPSNR -0.3733 dB, minimum -2.1119 dB,
14/48 wins.

## Feature-Rel vs Old Absolute Feature Cap

| Method | Budget | Runs | Mean dPSNR | Min dPSNR | Wins |
|---|---:|---:|---:|---:|---:|
| onedge residual | 2000 | 4 | +2.4383 | +1.9566 | 4/4 |
| onedge residual | 5000 | 4 | +1.7081 | +0.6385 | 4/4 |
| onedge residual | 10000 | 4 | +0.2252 | +0.0491 | 4/4 |
| onedge tensor | 2000 | 4 | +3.0265 | +2.4976 | 4/4 |
| onedge tensor | 5000 | 4 | +1.9100 | +0.4996 | 4/4 |
| onedge tensor | 10000 | 4 | +0.2221 | -0.3278 | 3/4 |
| qt-WSE residual | 2000 | 4 | +2.8218 | +2.4670 | 4/4 |
| qt-WSE residual | 5000 | 4 | +1.5086 | +0.7378 | 4/4 |
| qt-WSE residual | 10000 | 4 | +0.2847 | -0.0077 | 3/4 |
| qt-WSE tensor | 2000 | 4 | +3.3051 | +2.3313 | 4/4 |
| qt-WSE tensor | 5000 | 4 | +1.2590 | +0.6553 | 4/4 |
| qt-WSE tensor | 10000 | 4 | +0.0793 | -0.2853 | 2/4 |

## Decision

`feature_rel` fixes most of the old resolution-scaled cap failure, but it does not meet the
INIT-008 promotion criterion. It loses more than 0.1 dB at 2000 and 5000 in multiple method
families and averages -0.3733 dB versus matching uncapped rows. Keep `feature_rel` searchable
(`scale_cap=feature_rel`) and default off. Close ADR-0012's default candidacy for the current
fair-density protocol.

Artifacts copied here: `config.json`, `metrics.csv`, `metrics.json`, `summary.md`,
`convergence_curves.csv`, `target_hit_rates.csv`, `index.html`, and `plots/*.png`.
