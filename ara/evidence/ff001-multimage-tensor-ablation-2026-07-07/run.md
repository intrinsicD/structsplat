# FF-001 multi-image tensor-prior predictor ablation

Date: 2026-07-07

Goal: close the remaining FF-001 evidence gap with a held-out equal-N comparison and an ablation
between an image-only learned predictor and an image+structure-tensor-prior learned predictor.

## Protocol

Teacher/training split:

- Train/teacher images: `kodim01`, `kodim07`, `kodim13`.
- Held-out evaluation image: `kodim19`.
- Teacher export: `quadtree_wse`, 512 Gaussians, 600 fit iterations, max-side 384,
  `renderer=cuda`, seed 0.
- Predictors: tiny CNN, 512 output Gaussians, image size 96, hidden 128, 800 epochs, seed 0.
- Predictor arms:
  - `learned`: RGB image input only.
  - `learned_tensor`: RGB plus structure-tensor channels
    (`energy`, `coherence`, `cos(2 theta)`, `sin(2 theta)`).
- Held-out refinement/eval: 512 final Gaussians, 200 fit iterations, max-side 384,
  `renderer=cuda`, target PSNR 22.
- Baselines: `tensor_prior=quadtree_wse`, `scratch=random`.

## Teacher Fields

| image | PSNR | MS-SSIM |
|---|---:|---:|
| `kodim01` | 23.9236 | 0.88937 |
| `kodim07` | 22.9810 | 0.90355 |
| `kodim13` | 21.2450 | 0.84649 |

## Held-Out Result

| method | PSNR | MS-SSIM | AUC | total s | iters to 22 dB | seconds to 22 dB |
|---|---:|---:|---:|---:|---:|---:|
| `learned` | 21.0514 | 0.85978 | 20.4061 | 0.949523 | - | - |
| `learned_tensor` | 23.4686 | 0.90109 | 21.7372 | 0.333759 | 69 | 0.128815 |
| `scratch` | 22.9056 | 0.90148 | 21.6377 | 0.308444 | 108 | 0.166847 |
| `tensor_prior` | 25.3249 | 0.91512 | 24.1797 | 0.448505 | 15 | 0.174787 |

The image-only checkpoint remains a measured negative. Adding tensor-prior channels changes the
answer: `learned_tensor` beats random scratch by +0.5630 dB final PSNR and reaches 22 dB in 69
iterations / 0.128815 seconds instead of 108 iterations / 0.166847 seconds. It still loses
clearly to the hand `quadtree_wse` tensor prior (-1.8563 dB final PSNR, -2.4425 AUC), so this is
not a default-promotion or SOTA claim.

## Verdict

FF-001 is complete as a first warm-start system: it has the predictor API, teacher export,
training, learned checkpoint loading, short-refinement evaluation, held-out generalization test,
and image-only vs image+tensor ablation.

Decision: keep `strategy=feedforward` available for experiments, but do not promote it as a
default initializer. Future work should be a new FF task with a larger architecture or
predict-optimize-distill loop if amortized prediction remains important.

## Artifacts

- `teacher/`: teacher manifest, config, summaries, and fitted teacher fields.
- `train_image/`: image-only checkpoint and training logs.
- `train_tensor/`: image+tensor checkpoint and training logs.
- `eval/`: held-out equal-N comparison JSON/CSV/summary.

## Verification

- `python -m pytest tests/test_predictor.py -q` passed 7 tests before this run.
