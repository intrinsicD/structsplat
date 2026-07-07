# FIT-012 edge-weighted loss

Date: 2026-07-07

Implemented `FitConfig.loss_weighting={none,tensor}` with `loss_weight_beta`, weighting only the
pixel-loss term by `w = 1 + beta * E_norm`. SSIM and all reported metrics remain unweighted.

## Command

```bash
python -m benchmarks.stage_search \
  results/datasets/abl004/kodak24/kodim01.png \
  results/datasets/abl004/kodak24/kodim07.png \
  results/datasets/abl004/kodak24/kodim13.png \
  results/datasets/abl004/kodak24/kodim19.png \
  --mode factorial --budgets 2000 5000 --seeds 0 --iters 1500 --max-side 768 \
  --strategies aniso_onedge quadtree_wse \
  --tensor-operators central --tensor-colors luma \
  --density-modes structure --sampling-modes wse --orientation-modes tensor \
  --color-modes bilinear --scale-modes spacing --scale-cap-modes none \
  --opacity-modes none --renderers cuda --aa-dilations 0.0 \
  --color-basis-modes constant --color-solve-modes none \
  --pixel-losses l1 --loss-weight-modes none tensor \
  --optimizers adam --lr-schedules none \
  --refine-sites residual_tensor --refine-primitives sampled_add --refine-nms-modes off \
  --refine-color-inits target --refine-prune-modes off --refine-relocate-modes off \
  --state-seed-modes off --row-temper-modes off --support-fade-modes off \
  --pyramid-modes single --split-every 300 --split-count 250 \
  --chunk 512 --outdir results/fit012_edge_weighted_loss_difficult4_2026_07_07 \
  --device cuda --resume
```

## Result

The run completed 32/32 cells with `status=ok`, forming 16 tensor-vs-none pairs.

Tensor loss weighting vs matching unweighted rows:

| metric | mean delta | median delta | wins |
|---|---:|---:|---:|
| PSNR | +0.0061 dB | +0.0244 dB | 10/16 |
| MS-SSIM | -0.00068 | +0.00005 | 8/16 |
| edge MAE | +0.000018 | -0.000067 | 10/16 lower-is-better wins |
| PSNR AUC | -0.0107 | -0.0115 | 6/16 |

Strategy split:

| strategy | dPSNR mean | dMS-SSIM mean | dEdge-MAE mean | dAUC mean |
|---|---:|---:|---:|---:|
| aniso_onedge | +0.2661 | +0.00129 | -0.000428 | +0.0507 |
| quadtree_wse | -0.2538 | -0.00266 | +0.000464 | -0.0721 |

Decision: implemented and searchable, but default off. `loss_weight=tensor` is useful enough to
keep as an axis, especially with `aniso_onedge`, but the full matched aggregate does not justify a
global promotion.

## Artifacts

- `stage_search.csv` / `stage_search.json`: raw rows.
- `paired_deltas.csv`: tensor-minus-none pairs by image/budget/strategy.
- `aggregate_deltas.csv`: aggregate paired deltas overall, by budget, and by strategy.
- `summary.md` / `index.html`: generated stage-search report.

## Verification

- `python -m pytest tests/test_fit_dynamics.py tests/test_stage_search.py tests/test_cli.py tests/test_pyramid.py -q`
  passed 96 tests.
- `python -m pytest -q` passed 290 tests.
