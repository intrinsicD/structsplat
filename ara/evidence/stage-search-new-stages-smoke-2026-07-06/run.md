# Stage-Search New Stages Smoke - 2026-07-06

Purpose: verify that the newly exposed stage-search axes execute end to end:
`color_solve=every10`, `refine=freq_violation`, and
`refine=moment_preserving`.

Focused unit command:

```bash
PYTHONPATH=src:. pytest -q \
  tests/test_stage_search.py::test_color_solve_is_stage_axis \
  tests/test_stage_search.py::test_refine_kwargs_threads_fit004_modes
```

Result: 2 passed in 1.51 s.

Stage-search smoke command:

```bash
PYTHONPATH=src:. python benchmarks/stage_search.py \
  tests/test_images/COCO_train2014_000000000034.jpg \
  --mode factorial --budgets 32 --seeds 0 --iters 10 --max-side 32 \
  --strategies aniso_flanking --tensor-operators central --tensor-colors luma \
  --density-modes structure --sampling-modes density_random \
  --orientation-modes tensor --color-modes bilinear --scale-modes spacing \
  --scale-cap-modes none --opacity-modes none --renderers normalized \
  --aa-dilations 0 --color-solve-modes none every10 --pixel-losses l1 \
  --optimizers adam --lr-schedules none \
  --refine-modes none freq_violation moment_preserving --pyramid-modes single \
  --split-every 5 --split-count 8 --chunk 64 \
  --outdir results/stage_search_new_stages_smoke --device cpu
```

Result: 6/6 cells completed with status `ok`.
Live overview: `results/stage_search_new_stages_smoke/index.html`.

| Refine | Color solve | PSNR | AUC | Split events | Color events | N | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|
| none | none | 19.7711 | 19.2176 | 0 | 0 | 32 | 0.0473 |
| freq_violation | none | 19.4606 | 18.6553 | 1 | 0 | 32 | 0.0457 |
| moment_preserving | none | 19.4306 | 18.6386 | 1 | 0 | 32 | 0.0451 |
| none | every10 | 21.0691 | 19.2911 | 0 | 1 | 32 | 0.1018 |
| freq_violation | every10 | 21.3015 | 18.7671 | 1 | 1 | 32 | 0.1094 |
| moment_preserving | every10 | 20.9582 | 18.7327 | 1 | 1 | 32 | 0.1097 |

Metadata checks:

- `color_solve=every10` rows recorded one color-solve event.
- `freq_violation` rows recorded one split event and non-null frequency-violation stats.
- `moment_preserving` rows recorded one split event and completed at the requested final
  budget.

Interpretation: this is a plumbing smoke only. It validates that the new stage axes run and
write expected metadata; it is too small to rank methods.
