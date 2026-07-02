# Heuristics

## H01: Matched Cross-Repo Evaluation Protocol
- **Rationale**: Compare image set, resolution, Gaussian budget, iteration or stopping mode, metrics, and seed under one protocol, and report representation quality separately from codec bpp when repos define storage differently.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`benchmarks/coco_fit_compare.py`, `results/coco_fit_compare/summary.md`, `results/coco_fit_compare/delta_after_update.md`]
- **From staging**: O03

## H02: Feature-Adaptive Scale Caps as Search Baseline
- **Rationale**: Use the feature-adaptive `feature12` scale cap as the stage-search baseline after the held-out capped benchmark improved mean PSNR, suppressed final max/p95 scale, won most images, and reduced runtime versus uncapped initialization.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: medium
- **Code ref**: [`src/structsplat/config.py`, `src/structsplat/init.py`, `src/structsplat/fit.py`, `benchmarks/stage_search.py`]
- **From staging**: O08

## H03: Tensor-Aware Residual Densification Candidate
- **Rationale**: Add residual Gaussians with local edge-tangent orientation, anisotropic scales, inherited scale caps, and renderer-dependent target/residual colors so capped initialization can be paired with searchable adaptive density control.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`src/structsplat/fit.py`, `benchmarks/stage_search.py`, `tests/test_fit_dynamics.py`]
- **From staging**: O10
