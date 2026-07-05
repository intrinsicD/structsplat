# Fair Density-Control Relocation Difficult Four

- Date: 2026-07-05
- Command: `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. python benchmarks/fair_density_control_compare.py --outdir results/fair_density_control_difficult4 --resume`
- Note: the first no-preload attempt failed the 24 new relocation cells with the known local `libstdc++` / `CXXABI_1.3.15` CUDA extension issue. Those `status=error` JSONL rows were removed and the same resumed run was repeated with the system `libstdc++` preload.
- Result root: `results/fair_density_control_difficult4/`
- HTML overview: `results/fair_density_control_difficult4/index.html`
- Rows: 156/156 ok, 0 errors
- New work: 24 relocation cells added to the previous 132-cell fair density-control run
- Artifacts: `summary.md`, `metrics.jsonl`, `metrics.csv`, `metrics.json`, `convergence_curves.csv`, `target_hit_rates.csv`, 156 reconstructions, 160 diff PNGs, 7 visual grids, plots, and updated `index.html`

## Overall Means

| Method | PSNR | MS-SSIM | AUC | Fit s |
|---|---:|---:|---:|---:|
| SS on-edge + residual | 28.7012 | 0.92726 | 25.914 | 25.774 |
| SS on-edge + residual relocate | 28.7339 | 0.92917 | 25.743 | 30.158 |
| SS qt-WSE + residual | 28.7143 | 0.92545 | 25.947 | 24.309 |
| SS qt-WSE + residual relocate | 28.5922 | 0.92619 | 25.734 | 28.349 |

## Paired Relocation Deltas

Positive means the relocation row beat its matching non-relocation residual row on the same image and budget.

| Pair | Mean dPSNR | PSNR Wins | Mean dMS-SSIM | MS Wins | Mean dAUC | AUC Wins | Mean dFit s |
|---|---:|---:|---:|---:|---:|---:|---:|
| on-edge residual relocate - on-edge residual | +0.0326 | 5/12 | +0.00191 | 6/12 | -0.171 | 0/12 | +4.38 |
| qt-WSE residual relocate - qt-WSE residual | -0.1221 | 3/12 | +0.00074 | 6/12 | -0.212 | 0/12 | +4.04 |

## Budget Breakdown

| Pair | Budget | Mean dPSNR | Mean dAUC | PSNR Wins |
|---|---:|---:|---:|---:|
| on-edge relocation | 2000 | +0.3415 | -0.070 | 3/4 |
| on-edge relocation | 5000 | -0.0510 | -0.180 | 2/4 |
| on-edge relocation | 10000 | -0.1927 | -0.264 | 0/4 |
| qt-WSE relocation | 2000 | -0.1131 | -0.175 | 2/4 |
| qt-WSE relocation | 5000 | -0.1143 | -0.207 | 0/4 |
| qt-WSE relocation | 10000 | -0.1388 | -0.254 | 1/4 |

## Verdict

On this hard-selected seed-0 subset, split-scheduled residual relocation is not a default-worthy improvement. It can improve isolated final-PSNR cells, especially low-budget on-edge rows, but it consistently hurts convergence AUC and increases fit time. The existing non-relocation residual/tensor rows remain the better default candidates unless a broader confirmation run or a tuned relocation schedule changes this.
