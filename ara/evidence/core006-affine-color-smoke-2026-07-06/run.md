# CORE-006 Affine Color Smoke - 2026-07-06

Purpose: validate the optional per-Gaussian affine color basis and get a first
constant-vs-affine signal.

Implementation summary:

- `GaussianField` now carries optional `color_grads` with shape `(N, 2, 3)`.
- Reference normalized/additive renderers evaluate
  `c0 + cx * local_x + cy * local_y`, where local coordinates are scale-normalized
  Gaussian coordinates.
- `FitConfig.color_basis` supports `constant` and `affine`; affine fields use Adam
  for base colors and gradients with `color_grad_l2` regularization.
- FIT-005 color solve intentionally rejects affine fields in this first implementation.
- NPZ save/load round-trips affine coefficients.
- Codec v1 rejects affine fields with a clear unsupported-format error.

Validation:

```bash
PYTHONPATH=src:. pytest -q
```

Result: 205 passed, 22 skipped in 8.21 s.

Benchmark smoke:

```bash
PYTHONPATH=src:. python benchmarks/stage_search.py \
  tests/test_images/COCO_train2014_000000000034.jpg \
  tests/test_images/COCO_train2014_000000000025.jpg \
  --mode factorial --budgets 24 32 --seeds 0 --iters 20 --max-side 32 \
  --strategies aniso_flanking --tensor-operators central --tensor-colors luma \
  --density-modes structure --sampling-modes density_random \
  --orientation-modes tensor --color-modes bilinear --scale-modes spacing \
  --scale-cap-modes none --opacity-modes none --renderers normalized \
  --aa-dilations 0 --color-basis-modes constant affine \
  --color-solve-modes none --pixel-losses l1 --optimizers adam \
  --lr-schedules none --refine-modes none --pyramid-modes single \
  --chunk 64 --outdir results/core006_affine_color_smoke --device cpu
```

Result: 8/8 cells completed with status `ok`.
Live overview: `results/core006_affine_color_smoke/index.html`.

Mean over two images x two budgets:

| Color basis | Mean PSNR | Mean AUC | Mean fit s |
|---|---:|---:|---:|
| constant | 20.4391 | 19.1186 | 0.1056 |
| affine | 21.5029 | 19.8572 | 0.1390 |

Paired affine minus constant deltas:

| Image | Budget | Delta PSNR | Delta AUC | Delta fit s |
|---|---:|---:|---:|---:|
| COCO_train2014_000000000025 | 24 | +0.1612 | +0.6191 | +0.0542 |
| COCO_train2014_000000000025 | 32 | +0.5455 | +0.3855 | +0.0242 |
| COCO_train2014_000000000034 | 24 | +1.5145 | +0.8572 | +0.0240 |
| COCO_train2014_000000000034 | 32 | +2.0340 | +1.0928 | +0.0312 |

Interpretation: affine is promising in this tiny low-budget slice and deserves
stage-search coverage. It should not be promoted to a shipped default from this
smoke alone because the run covers only two small images, one seed, and 20
iterations, and uses the slower reference renderer path.
