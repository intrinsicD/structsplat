# FF-001: Feed-forward init predictor (warm-start)

**Status: partial.** First implementation slices landed 2026-07-06/07: stable predictor
interface, `strategy=feedforward`, saved-field/tensor-prior warm starts, CLI short-refinement
flags, teacher-field export, and a tiny learned CNN checkpoint trainer. Larger predictor
architecture, distillation, decision-grade equal-N speed/quality comparisons, and generalization
tests remain open.

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
- [~] Compared against optimized-from-scratch and structure-tensor warm-start at equal final N,
      reporting quality, wall time, and speedup-to-target. Smoke evaluator exists; larger
      train/validation comparison remains open.
- [ ] Generalization test on images not used for teacher export.
- [ ] Ablation: image-only predictor vs image+tensor-prior predictor.
- [ ] If predict-optimize-distill is implemented, report teacher-only vs distilled predictor
      quality and refinement iterations saved.

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

## Depends on
INIT-003, FIT-001. Optional follow-ups: FIT-008 for adaptive count, COMP-004 for
compression-aware prediction.
