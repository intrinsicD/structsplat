# BENCH-004 Proxy Calibration

Full regime: `ara/evidence/abl004-stage-screen-8img-cuda-2026-07-04/metrics.json`
Proxy regime: `results/bench004_proxy_384_500/ablation.json`
Metric: `psnr`
Sign agreement threshold: `|full delta| > 0.1` dB

## Rank Correlation

| Budget | Strategies | Spearman rho | Full top-1 | Proxy top-1 |
|---:|---:|---:|---|---|
| 2000 | 11 | 0.9182 | aniso_onedge | quadtree_wse |
| 5000 | 11 | 0.8636 | quadtree_wse | quadtree_hybrid |

## Paired-Delta Sign Agreement

| Budget | Compared pairs | Agreeing pairs | Skipped small deltas | Agreement |
|---:|---:|---:|---:|---:|
| 2000 | 49 | 45 | 6 | 91.8% |
| 5000 | 49 | 44 | 6 | 89.8% |

Acceptance target: Spearman rho >= 0.9 and sign agreement >= 90% at each budget.
