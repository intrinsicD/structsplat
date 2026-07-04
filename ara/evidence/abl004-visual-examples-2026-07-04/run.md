# ABL-004 Visual Examples Run

## Command

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m benchmarks.abl004_visual_examples \
  --outdir results/abl004_visual_examples_5img \
  --renderer cuda
```

## Scope

- Images: `kodim01`, `kodim07`, `kodim10`, `kodim13`, `kodim22`
- Levels: `160/240/320 px` x `80/200 iterations`
- Budgets: 640, 1440, 2560 Gaussians from area-scaling a 640 budget at 160 px
- Variants: `aniso_onedge`, `aniso_flanking`, `quadtree_wse`, `quadtree_hybrid`, `iso_blue_noise`, `floyd_steinberg`
- Seed: 0
- Renderer: exact CUDA

## Validation

- Completed 180/180 fitted cells.
- Generated 180 reconstruction PNGs under `results/abl004_visual_examples_5img/reconstructions/`.
- Generated 5 all-level comparison sheets under `results/abl004_visual_examples_5img/grids/`.
- `metrics.csv` contains 180 rows and 0 error rows.
- Visual spot-check of `kodim07_all_levels.png` showed all target/variant cells rendered with readable labels.

## Evidence Contents

- `summary.md`
- `metrics.csv`
- `metrics.json`
- `config.json`
- `grids/kodim01_all_levels.png`
- `grids/kodim07_all_levels.png`
- `grids/kodim10_all_levels.png`
- `grids/kodim13_all_levels.png`
- `grids/kodim22_all_levels.png`
