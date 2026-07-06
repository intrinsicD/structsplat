# FIT-008 Adaptive Count Smoke

Date: 2026-07-06

Purpose: validate the self-adaptive Gaussian-count controller and stage-search metadata. This is a
plumbing/behavior smoke, not a default-promotion benchmark.

## Implementation

Added optional `FitConfig.adaptive_count` controls:

- Stops on `target_psnr`, `target_ms_ssim`, raw-attribute `target_bpp`, `max_gaussians`, `stalled`,
  `no_growth`, or `iteration_limit`.
- Grows with existing residual split/add modes at `adaptive_growth_every` cadence.
- Records `history["adaptive_events"]`, `history["adaptive_stop_reason"]`,
  `history["adaptive_selected_n"]`, row-level `adaptive_*` metadata, and `estimated_bpp`.
- Fixed-N behavior remains unchanged when `adaptive_count=False`.

## Focused Tests

Command:

```bash
PYTHONPATH=src:. pytest \
  tests/test_fit_dynamics.py::test_adaptive_count_stops_when_target_psnr_reached_without_growth \
  tests/test_fit_dynamics.py::test_adaptive_count_stops_at_max_gaussians \
  tests/test_fit_dynamics.py::test_adaptive_count_stops_at_target_bpp_cap \
  tests/test_fit_dynamics.py::test_adaptive_count_grows_then_stops_on_stall \
  tests/test_gaussians.py::test_fitconfig_rejects_negative_dilation \
  tests/test_stage_search.py::test_adaptive_count_metadata_is_recorded -q
```

Result: 6 passed in 1.88 s.

## Smoke Commands

Adaptive 16 -> 32:

```bash
PYTHONPATH=src:. python benchmarks/stage_search.py \
  tests/test_images/COCO_train2014_000000000034.jpg \
  tests/test_images/COCO_train2014_000000000025.jpg \
  --mode factorial --budgets 16 --seeds 0 --iters 17 --max-side 32 \
  --strategies aniso_flanking --tensor-operators central --tensor-colors luma \
  --density-modes structure --sampling-modes density_random \
  --orientation-modes tensor --color-modes bilinear --scale-modes spacing \
  --scale-cap-modes none --opacity-modes none --renderers normalized \
  --aa-dilations 0 --color-basis-modes constant --color-solve-modes none \
  --pixel-losses l1 --optimizers adam --lr-schedules none \
  --refine-modes none --pyramid-modes single \
  --adaptive-count --max-gaussians 32 --target-psnr 99 \
  --adaptive-growth-every 4 --adaptive-growth-count 4 \
  --adaptive-split-mode residual_tensor_add \
  --chunk 64 --outdir results/fit008_adaptive_count_smoke --device cpu
```

Fixed 32 control:

```bash
PYTHONPATH=src:. python benchmarks/stage_search.py \
  tests/test_images/COCO_train2014_000000000034.jpg \
  tests/test_images/COCO_train2014_000000000025.jpg \
  --mode factorial --budgets 32 --seeds 0 --iters 17 --max-side 32 \
  --strategies aniso_flanking --tensor-operators central --tensor-colors luma \
  --density-modes structure --sampling-modes density_random \
  --orientation-modes tensor --color-modes bilinear --scale-modes spacing \
  --scale-cap-modes none --opacity-modes none --renderers normalized \
  --aa-dilations 0 --color-basis-modes constant --color-solve-modes none \
  --pixel-losses l1 --optimizers adam --lr-schedules none \
  --refine-modes none --pyramid-modes single \
  --chunk 64 --outdir results/fit008_fixed32_smoke --device cpu
```

## Result

Both adaptive rows selected the cap (`adaptive_selected_n=32`) and stopped with
`adaptive_stop_reason=max_gaussians_reached`.

| Mode | Rows | Mean PSNR | Mean AUC | Mean fit s | Mean N | Mean raw bpp |
|---|---:|---:|---:|---:|---:|---:|
| adaptive 16->32 | 2 | 19.5006 | 18.2459 | 0.0926 | 32.0 | 12.1905 |
| fixed 32 | 2 | 20.3149 | 19.2296 | 0.0623 | 32.0 | 12.1905 |

Paired adaptive-minus-fixed deltas:

| Image | dPSNR | dAUC | dFit s |
|---|---:|---:|---:|
| COCO_train2014_000000000025 | -1.4479 | -1.1955 | +0.0191 |
| COCO_train2014_000000000034 | -0.1808 | -0.7720 | +0.0415 |

Verdict: FIT-008 controller and metadata are working. This tiny short-run smoke does not support
adaptive growth as a default; fixed 32 starts with full capacity and wins here.

Artifacts:

- `results/fit008_adaptive_count_smoke/index.html`
- `results/fit008_adaptive_count_smoke/stage_search.json`
- `results/fit008_fixed32_smoke/index.html`
- `results/fit008_fixed32_smoke/stage_search.json`
