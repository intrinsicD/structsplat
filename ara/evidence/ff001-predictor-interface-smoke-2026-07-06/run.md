# FF-001 Predictor Interface Smoke

Date: 2026-07-06

Purpose: first FF-001 implementation slice. This validates the executable predictor interface,
`strategy=feedforward` warm-start path, and teacher-field export artifacts. It does not claim a
trained amortized predictor exists yet.

## Implemented

- Added `src/structsplat/predictor.py` with `predict(image, InitConfig/options) -> GaussianField`.
- Added `strategy="feedforward"` to initialization.
- Added optional `InitConfig.predictor_checkpoint` and `predictor_fallback_strategy`.
- Added CLI fit flags `--predictor-checkpoint` and `--predictor-fallback-strategy`.
- Added `benchmarks/feedforward_teacher_export.py` to export fitted teacher fields plus
  manifest/config/summary artifacts.

## Focused Tests

Command:

```bash
PYTHONPATH=src:. pytest \
  tests/test_predictor.py \
  tests/test_cli.py::test_fit_cli_accepts_feedforward_short_refinement \
  tests/test_smoke.py::test_build_field_all_strategies -q
```

Result: 14 passed in 1.65 s.

## Teacher Export Smoke

Command:

```bash
PYTHONPATH=src:. python -m benchmarks.feedforward_teacher_export \
  tests/test_images/COCO_train2014_000000000034.jpg \
  --budget 16 --strategy aniso_flanking --seed 0 --iters 2 --max-side 32 \
  --render-chunk 64 --outdir results/ff001_teacher_export_smoke --device cpu
```

Result row:

| image | budget | N | strategy | iters | PSNR | MS-SSIM | fit s |
|---|---:|---:|---|---:|---:|---:|---:|
| COCO_train2014_000000000034 | 16 | 16 | aniso_flanking | 2 | 18.4464 | 0.48704 | 0.010489 |

Artifacts:

- `results/ff001_teacher_export_smoke/config.json`
- `results/ff001_teacher_export_smoke/teacher_manifest.json`
- `results/ff001_teacher_export_smoke/teacher_manifest.csv`
- `results/ff001_teacher_export_smoke/summary.md`
- `results/ff001_teacher_export_smoke/fields/COCO_train2014_000000000034_n16_seed0.npz`

Verdict: interface/export plumbing is working. FF-001 remains partial because no learned predictor,
training loop, distillation loop, or generalization comparison has been implemented.
