# FIT-009 refine-axis factorization

- Date: 2026-07-07
- Purpose: validate the factored `refine_site x refine_primitive x refine_nms` interface and
  test the previously inexpressible `residual_tensor x moment_preserving` combination.
- Protocol: difficult Kodak four-image slice (`kodim01`, `kodim07`, `kodim13`, `kodim19`),
  budget 2000, seed 0, 1500 iterations, max-side 768, exact CUDA renderer, `quadtree_wse`
  initialization, equal final N. Adding-refine arms started at 1936 Gaussians and capped at
  2000.

Command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python -m benchmarks.stage_search \
  results/datasets/abl004/kodak24/kodim01.png \
  results/datasets/abl004/kodak24/kodim07.png \
  results/datasets/abl004/kodak24/kodim13.png \
  results/datasets/abl004/kodak24/kodim19.png \
  --mode factorial --budgets 2000 --seeds 0 --iters 1500 --max-side 768 \
  --strategies quadtree_wse --tensor-operators central --tensor-colors luma \
  --density-modes structure --sampling-modes wse --orientation-modes tensor \
  --color-modes bilinear --scale-modes spacing --scale-cap-modes none \
  --opacity-modes none --renderers cuda --aa-dilations 0.0 \
  --color-basis-modes constant --color-solve-modes none --pixel-losses l1 \
  --optimizers adam --lr-schedules none \
  --refine-sites residual residual_tensor \
  --refine-primitives sampled_add moment_preserving \
  --refine-nms-modes off --refine-color-inits target \
  --refine-prune-modes off --refine-relocate-modes off \
  --pyramid-modes single --split-every 750 --split-count 64 \
  --target-psnrs 28 30 32 --log-every 75 --chunk 512 \
  --outdir results/fit009_refine_factor_fair_slice_2026_07_07 --device cuda
```

Aggregate results:

| refine | cells | mean PSNR | mean MS-SSIM | mean AUC | mean fit s |
|---|---:|---:|---:|---:|---:|
| `moment_preserving` | 4 | 24.3194 | 0.88102 | 24.4891 | 4.774 |
| `residual_add` | 4 | 24.2549 | 0.88177 | 24.4723 | 4.881 |
| `residual_tensor_add` | 4 | 24.2213 | 0.87992 | 24.5915 | 4.737 |
| `residual_tensor_moment_preserving` | 4 | 24.0325 | 0.87685 | 24.4541 | 4.803 |

Pairwise deltas for the new cross-product:

| comparison | mean delta PSNR | wins | mean delta AUC | mean delta MS-SSIM |
|---|---:|---:|---:|---:|
| `residual_tensor_moment_preserving` - `residual_tensor_add` | -0.1888 | 1/4 | -0.1374 | -0.00306 |
| `residual_tensor_moment_preserving` - `moment_preserving` | -0.2869 | 0/4 | -0.0350 | -0.00416 |
| `residual_tensor_moment_preserving` - `residual_add` | -0.2225 | 1/4 | -0.0182 | -0.00492 |

Conclusion: the factored interface works and keeps all arms at equal final capacity, but this
slice does **not** support promoting `residual_tensor x moment_preserving`. Plain
`moment_preserving` had the best mean PSNR here, while `residual_tensor_add` had the best mean
AUC. Keep the cross-product searchable rather than treating it as an immediate default candidate.

Artifacts copied here: `config.json`, `stage_search.csv`, `stage_search.json`, `summary.md`, and
`index.html`.
