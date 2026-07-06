# FF-001 Equal-N Evaluation Smoke

Date: 2026-07-07

Purpose: validate the FF-001 equal-final-N evaluation harness for learned, tensor-prior, and
scratch warm starts. This is a one-image smoke using the tiny predictor checkpoint from
`results/ff001_tiny_predictor_smoke/`; it is not a generalization or speedup claim.

## Implemented

- Added `benchmarks/feedforward_eval.py`.
- The evaluator writes:
  - `feedforward_eval.json`
  - `feedforward_eval.csv`
  - `summary.md`
  - `config.json`
- Methods:
  - `learned`: `strategy=feedforward` with a learned `.pt` checkpoint.
  - `tensor_prior`: the configured tensor-prior strategy at equal N.
  - `scratch`: a random or configured scratch initializer at equal N.

## Focused Tests

Command:

```bash
PYTHONPATH=src:. pytest tests/test_predictor.py -q
```

Result: 6 passed in 6.80 s.

## Smoke Command

```bash
PYTHONPATH=src:. python -m benchmarks.feedforward_eval \
  tests/test_images/COCO_train2014_000000000034.jpg \
  --checkpoint results/ff001_tiny_predictor_smoke/predictor.pt \
  --outdir results/ff001_equaln_eval_smoke \
  --budget 16 --iters 2 --max-side 32 --render-chunk 64 --seed 0 --device cpu \
  --prior-strategy aniso_flanking --scratch-strategy random --target-psnr 17.0
```

## Results

| method | N | PSNR | MS-SSIM | AUC | init s | fit s | total s | iters to 17 dB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| learned | 16 | 17.4469 | 0.41073 | 16.7518 | 0.014687 | 0.018336 | 0.033022 | 1 |
| tensor_prior | 16 | 18.4464 | 0.48704 | 17.8798 | 0.012412 | 0.008923 | 0.021335 | 0 |
| scratch | 16 | 18.0079 | 0.42004 | 18.1098 | 0.000339 | 0.008885 | 0.009224 | 0 |

Artifacts:

- `results/ff001_equaln_eval_smoke/config.json`
- `results/ff001_equaln_eval_smoke/feedforward_eval.json`
- `results/ff001_equaln_eval_smoke/feedforward_eval.csv`
- `results/ff001_equaln_eval_smoke/summary.md`

Verdict: the equal-N evaluator is functional and exposes a weak tiny checkpoint honestly. On this
one-image smoke, the learned predictor loses to the tensor-prior and scratch baselines, so FF-001
still needs multi-image training, validation, and ablations before any speedup/quality claim.
