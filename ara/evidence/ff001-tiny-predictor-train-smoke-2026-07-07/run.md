# FF-001 Tiny Predictor Train Smoke

Date: 2026-07-07

Purpose: validate the learned checkpoint path for FF-001. This uses the existing one-image teacher
export smoke and trains a tiny CNN Gaussian regressor. It is an interface/training-contract smoke,
not a speedup, quality, or generalization claim.

## Implemented

- Added `TinyGaussianPredictorNet`, normalized Gaussian target decoding, and safe `.pt` checkpoint
  load/save helpers in `src/structsplat/predictor.py`.
- Added `benchmarks/feedforward_train.py` to train from
  `teacher_manifest.json` produced by `benchmarks/feedforward_teacher_export.py`.
- Extended `strategy="feedforward"` so `predictor_checkpoint=*.pt` predicts Gaussian means, scales,
  rotations, colors, and opacity, then truncates or pads to the requested budget.

## Focused Tests

Command:

```bash
PYTHONPATH=src:. pytest tests/test_predictor.py -q
```

Result: 5 passed in 1.43 s.

## Training Smoke

Command:

```bash
PYTHONPATH=src:. python -m benchmarks.feedforward_train \
  results/ff001_teacher_export_smoke/teacher_manifest.json \
  --outdir results/ff001_tiny_predictor_smoke \
  --image-size 16 --hidden 8 --epochs 20 --lr 0.01 --seed 0 --device cpu
```

Result:

| rows | N | epochs | initial loss | final loss |
|---:|---:|---:|---:|---:|
| 1 | 16 | 20 | 1.623774 | 0.421427 |

Artifacts:

- `results/ff001_tiny_predictor_smoke/predictor.pt`
- `results/ff001_tiny_predictor_smoke/config.json`
- `results/ff001_tiny_predictor_smoke/loss_history.csv`
- `results/ff001_tiny_predictor_smoke/loss_history.json`
- `results/ff001_tiny_predictor_smoke/summary.md`

## Short-Refinement Smoke

Command:

```bash
PYTHONPATH=src:. python - <<'PY'
from benchmarks.common import load_image, target_tensor
from structsplat import init as _init
from structsplat.config import FitConfig, InitConfig
from structsplat.fit import fit

img = load_image('tests/test_images/COCO_train2014_000000000034.jpg', max_side=32)
field = _init.build_field(
    img,
    InitConfig(
        strategy='feedforward',
        num_gaussians=16,
        predictor_checkpoint='results/ff001_tiny_predictor_smoke/predictor.pt',
        predictor_fallback_strategy='grid',
    ),
    device='cpu',
)
out = fit(field, target_tensor(img, 'cpu'), FitConfig(iters=2, render_chunk=64, log_every=1), verbose=False)
print({'n': out['field'].n, 'psnr': round(float(out['psnr']), 4), 'ms_ssim': round(float(out['ms_ssim']), 5), 'fit_seconds': round(float(out.get('fit_seconds', 0.0)), 6)})
PY
```

Result:

```text
{'n': 16, 'psnr': 17.4469, 'ms_ssim': 0.41073, 'fit_seconds': 0.096339}
```

Verdict: the learned checkpoint path trains, saves, loads, emits finite in-bounds Gaussian fields,
and runs through the existing short-refinement fitter. FF-001 remains partial because there is no
multi-image training set, generalization test, image-vs-tensor-prior ablation, distillation loop,
or equal-final-N wall-time comparison yet.
