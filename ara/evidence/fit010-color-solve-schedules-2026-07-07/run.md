# FIT-010 color-solve schedule smoke

- Date: 2026-07-07
- Purpose: test cheap event-based color-solve schedules against `every10`.
- Protocol: two difficult Kodak images (`kodim01`, `kodim07`), CPU, max-side 64, budget 80,
  seed 0, 60 iterations, one residual sampled-add split wave at iteration 30, initial 64 ->
  final 80 Gaussians, normalized renderer, constant colors.

Command:

```bash
PYTHONPATH=. python -m benchmarks.stage_search \
  results/datasets/abl004/kodak24/kodim01.png \
  results/datasets/abl004/kodak24/kodim07.png \
  --mode factorial --budgets 80 --seeds 0 --iters 60 --max-side 64 \
  --strategies aniso_flanking --tensor-operators central --tensor-colors luma \
  --density-modes structure --sampling-modes density_random --orientation-modes tensor \
  --color-modes bilinear --scale-modes spacing --scale-cap-modes none \
  --opacity-modes none --renderers normalized --aa-dilations 0.0 \
  --color-basis-modes constant \
  --color-solve-modes none every10 init on_split init+on_split \
  --pixel-losses l1 --optimizers adam --lr-schedules none \
  --refine-sites residual --refine-primitives sampled_add --refine-nms-modes off \
  --refine-color-inits target --refine-prune-modes off --refine-relocate-modes off \
  --pyramid-modes single --split-every 30 --split-count 16 --log-every 3 \
  --chunk 64 --outdir results/fit010_color_solve_schedule_smoke_2026_07_07 \
  --device cpu
```

Schedule results:

| color solve | cells | mean PSNR | delta PSNR vs none | mean AUC | delta AUC | mean fit s | extra fit s | mean events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 2 | 23.3878 | +0.0000 | 21.6988 | +0.0000 | 0.3459 | +0.0000 | 0.00 |
| `every10` | 2 | 23.8762 | +0.4884 | 22.0057 | +0.3070 | 0.9331 | +0.5872 | 6.00 |
| `init` | 2 | 22.8631 | -0.5247 | 21.8042 | +0.1054 | 0.4590 | +0.1131 | 1.00 |
| `on_split` | 2 | 23.5811 | +0.1933 | 21.9235 | +0.2247 | 0.4646 | +0.1187 | 1.00 |
| `init+on_split` | 2 | 23.0872 | -0.3006 | 21.9854 | +0.2866 | 0.6181 | +0.2723 | 2.00 |

Promotion target: an event schedule needed at least 70% of `every10`'s PSNR delta
(+0.3419 dB) while using at most 30% of `every10`'s extra fit time (+0.1762 s). `on_split` met
the cost target but captured only +0.1933 dB, so no event schedule met the promotion rule.

Split-dip interaction from logged PSNR around the split:

| color solve | mean post-split delta | mean recovery lag |
|---|---:|---:|
| `none` | -0.8055 dB | 4.5 iters |
| `every10` | -1.4500 dB | 4.5 iters |
| `init` | -0.7056 dB | 4.5 iters |
| `on_split` | +0.8418 dB | 0 iters |
| `init+on_split` | +0.6415 dB | 0 iters |

Conclusion: event-based schedules are implemented and useful for split recovery/AUC screening, but
this smoke does **not** justify replacing `every10` as the quality arm. Keep periodic `every<N>` as
the promoted color-solve axis; keep `on_split` searchable for split-dip work.

Artifacts copied here: `config.json`, `stage_search.csv`, `stage_search.json`, `summary.md`, and
`index.html`.
