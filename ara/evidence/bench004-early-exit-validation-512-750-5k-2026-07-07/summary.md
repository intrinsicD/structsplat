# BENCH-004 Early-Exit Validation

Baseline: `results/bench004_proxy_512_750/ablation.json` filtered to budget 5000.
Early-exit run: `results/bench004_early_exit_512_750_5k/ablation.json`.

Same images, strategies, seed, budget, renderer, loss, max-side, and 750-iteration nominal horizon. Only `--early-exit --early-exit-window 150 --early-exit-min-delta 0.02` changed.

## Savings

| Cells | Stopped cells | Nominal iters | Actual iters | Iter saving | Baseline fit s | Early-exit fit s | Fit saving | Early wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 88 | 13 | 66000 | 65257 | 1.1% | 138.63 | 136.34 | 1.7% | 180.99 |

## Preservation

| Metric | Spearman rho | Baseline top-1 | Early-exit top-1 | Compared pairs | Agreeing pairs | Agreement | Skipped small deltas |
|---|---:|---|---|---:|---:|---:|---:|
| psnr | 0.8636 | quadtree_hybrid | quadtree_hybrid | 43 | 41 | 95.3% | 12 |
| auc_psnr | 1.0000 | quadtree_hybrid | quadtree_hybrid | 43 | 43 | 100.0% | 12 |

AUC uses the BENCH-004 nominal-horizon convention: if a cell exits early, hold the last logged PSNR to 750 iterations. AUC preserves all paired signs above the 0.1 dB threshold; final PSNR keeps top-1 but reorders two close pairs.
