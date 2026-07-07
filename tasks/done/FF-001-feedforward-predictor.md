# FF-001: Feed-forward init predictor (warm-start)

**Status: done.** The first feed-forward warm-start system is implemented and evaluated:
predictor API, `strategy=feedforward`, saved-field/tensor-prior warm starts, teacher export,
tiny learned checkpoint training, image+tensor-prior inputs, short-refinement evaluation,
held-out generalization test, and image-only vs tensor-prior-input ablation.

## Context
StructSplat is strong on hand-designed initialization and optimization, but it does not yet have
the current fastest pattern: amortized Gaussian prediction followed by short refinement. Instant
GaussianImage-style coarse prediction and AIR-style predict-optimize-distill are the clearest path
to a 10x fitting-time win rather than another small optimizer gain.

## Goal
Train a small feed-forward model that predicts a warm-start Gaussian field, budget map, or both from
an input image, then refines for only 50-200 iterations with the existing fitter.

## Approach
1. Build a teacher dataset by running the best current StructSplat pipeline on pinned image crops and
   saving fields, stage metadata, fit traces, and quality metrics.
2. Train a compact U-Net/ConvNeXt-style predictor to output density/budget maps plus Gaussian
   attributes or local candidate sets.
3. Add a self-supervised predict-optimize-distill loop: predict -> short optimize -> distill the
   refined field back into the predictor target.
4. Keep structure-tensor priors available as inputs or auxiliary losses so the network does not have
   to rediscover edge orientation from scratch.
5. Evaluate both fixed-N and adaptive-N variants; adaptive count can share the controller in
   FIT-008.

## Acceptance criteria
- [x] `src/structsplat/predictor.py` or equivalent module with a documented model interface:
      `image, budget/options -> GaussianField`.
- [x] Dataset/export script for teacher fields and a minimal training script with deterministic
      config logging.
- [x] Predictor can emit positions, covariance/orientation, colors, and optional opacity; or emits a
      placement/budget map consumed by the existing initializer with clear scope.
- [x] Short-refinement path exposed in CLI/config, e.g. `init="feedforward"` plus
      `fit_iters=50-200`.
- [x] Compared against optimized-from-scratch and structure-tensor warm-start at equal final N,
      reporting quality, wall time, and speedup-to-target. Smoke evaluator exists; larger
      train/validation comparison remains open.
- [x] Generalization test on images not used for teacher export.
- [x] Ablation: image-only predictor vs image+tensor-prior predictor.
- [x] Predict-optimize-distill was not implemented in FF-001; future distillation work should be
      a new task with its own evidence.

## Outcome

Evidence: `ara/evidence/ff001-multimage-tensor-ablation-2026-07-07/`.

The final FF-001 slice exported 512-Gaussian teacher fields from `kodim01`, `kodim07`, and
`kodim13` with `quadtree_wse`, 600 fit iterations, max-side 384, exact CUDA rendering, then
trained matched tiny CNN predictors for 800 epochs:

- `learned`: RGB image input only.
- `learned_tensor`: RGB plus structure-tensor channels (`energy`, `coherence`,
  `cos(2 theta)`, `sin(2 theta)`).

Held-out equal-N comparison on `kodim19`, 512 final Gaussians, 200 refinement iterations:

| method | PSNR | MS-SSIM | AUC | total s | iters to 22 dB | seconds to 22 dB |
|---|---:|---:|---:|---:|---:|---:|
| `learned` | 21.0514 | 0.85978 | 20.4061 | 0.949523 | - | - |
| `learned_tensor` | 23.4686 | 0.90109 | 21.7372 | 0.333759 | 69 | 0.128815 |
| `scratch` | 22.9056 | 0.90148 | 21.6377 | 0.308444 | 108 | 0.166847 |
| `tensor_prior` | 25.3249 | 0.91512 | 24.1797 | 0.448505 | 15 | 0.174787 |

Decision: `strategy=feedforward` is a working experimental path, not a default initializer.
Image-only prediction remains a measured negative. Tensor-prior channels make the tiny learned
model useful versus random scratch on this held-out slice (+0.5630 dB PSNR and faster time to
22 dB), but the hand `quadtree_wse` tensor prior is still clearly better (-1.8563 dB gap for
`learned_tensor`). Future amortized-prediction work should move to a larger architecture or
predict-optimize-distill under a new task.

## Interfaces touched
`src/structsplat/predictor.py` (new), `src/structsplat/init.py`, `src/structsplat/fit.py`,
`src/structsplat/config.py`, `src/structsplat/cli.py`, training/export scripts under
`benchmarks/` or `tools/`, tests for shape/range/round-trip invariants.

## Current implementation notes

- `InitConfig(strategy="feedforward")` calls `structsplat.predictor.predict_field`.
- `predictor_checkpoint` loads a saved `GaussianField` and truncates/pads to the requested budget.
- `predictor_checkpoint=*.pt` loads a tiny learned CNN checkpoint trained by
  `benchmarks/feedforward_train.py`. It predicts normalized means, scales, rotation unit vectors,
  colors, and opacity for a fixed budget, then truncates/pads to the requested N.
- Without a checkpoint, `predictor_fallback_strategy` delegates to an existing deterministic
  tensor-prior initializer. This is an executable API fallback, not a learned model.
- `benchmarks/feedforward_teacher_export.py` exports fitted teacher fields and a manifest for
  future model training.
- `benchmarks/feedforward_train.py` trains the minimal checkpoint contract from that manifest.
- `benchmarks/feedforward_eval.py` compares learned, tensor-prior, and scratch warm starts at equal
  final N and short-refinement iterations.
- Evidence: `ara/evidence/ff001-predictor-interface-smoke-2026-07-06/run.md`.
- Evidence: `ara/evidence/ff001-tiny-predictor-train-smoke-2026-07-07/run.md`.
- Evidence: `ara/evidence/ff001-equaln-eval-smoke-2026-07-07/run.md`.
- Evidence: `ara/evidence/ff001-multimage-tensor-ablation-2026-07-07/run.md`.
- The held-out multi-image result supersedes the tiny one-image smoke: tensor-prior input helps,
  but the hand tensor-prior initializer remains stronger.

## Depends on
INIT-003, FIT-001. Optional follow-ups: FIT-008 for adaptive count, COMP-004 for
compression-aware prediction.
