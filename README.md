# StructSplat

**Hierarchical, feature-aware, anisotropic blue-noise 2D Gaussian image representation.**

A single image is encoded as a set of oriented 2D Gaussians and rendered by a sorting-free,
normalized weighted-sum rasterizer. The research contribution is the **initialization**:

- a **structure tensor** `J = G_ρ * (∇I ∇Iᵀ)` is the single operator for *density* (where to put
  Gaussians), *orientation* (how to elongate them), and *classification* (flat / edge / corner);
- Gaussians are placed by **anisotropic, density-adaptive blue noise** (Weighted Sample Elimination
  with a Mahalanobis metric) — packing across edges, spreading along them, no clumping, no grid;
- edges are **flanked** (not centered on the discontinuity), corners are centered, flats are sparse;
- the layout is built **progressively** (coarse→fine), so prefixes act as a level-of-detail stack.

The pieces exist separately in prior work (anisotropic blue noise; structured 2D-GS init;
error-driven densification) but not combined into a progressive 2D-Gaussian image codec. `structsplat`
is a **placeholder name** — rename freely (see the `docs-sync` skill).

> This is a PyTorch **research reference**. The sampler and rasterizer are the pieces later ported to
> CUDA/Vulkan and into IntrinsicEngine as an RHI pass (`tasks/PORT-001`).

## Install
```bash
pip install -e .                 # torch, numpy, pillow, imageio
pip install -e ".[metrics]"      # optional: lpips, pytorch-msssim
pip install -e ".[dev]"          # pytest, ruff
```

## Quickstart
```bash
# fit one image with the proposed init
structsplat fit photo.png --strategy aniso_flanking --num-gaussians 20000 --iters 2000

# progressive (hierarchical) fit
structsplat fit photo.png --pyramid --num-gaussians 20000

# the core experiment: init strategy x budget sweep (writes results/summary.md)
structsplat ablation ./images --budgets 2000 5000 10000 20000 --iters 1500 --target-psnr 35

# full stage-combination screening: tensor/density/sampling/color/loss/refinement/pyramid
structsplat stage-search ./images --budgets 1024 2048 --iters 300 --outdir results/stage_search

# per-stage influence: one-factor-at-a-time deltas vs the baseline (writes influence.md with
# ΔPSNR / ΔMS-SSIM / ΔAUC / Δiters-to-target / Δseconds per stage option, ADR-0010)
structsplat stage-search ./images --mode influence --budgets 2048 --seeds 0 1 2 \
    --iters 500 --target-psnr 30 --outdir results/stage_influence
```
Strategies: `random`, `grid`, `iso_blue_noise`, `aniso_onedge`, `aniso_flanking`.
Samplers: `wse` (blue noise), `dart_throwing` (Poisson disk), `halton`, `cvt`, `farthest_point`,
`density_random`, `jittered_grid`.

## Agentic workflow (Claude Code)
This repo is built to be implemented *with* Claude Code, mirroring the IntrinsicEngine setup.

- **`CLAUDE.md`** — project guide + a skill-aware routing table.
- **`.claude/skills/`** — six project skills: `core`, `task-workflow`, `review`, `method`,
  `benchmark`, `docs-sync`. They're auto-discovered inside this repo; run
  `scripts/install_skills.sh` to symlink them into `~/.claude/skills` for global use.
- **`tasks/`** — work items (`AREA-NNN-slug.md`) tracked in `tasks/INDEX.md`. Say *"work on
  INIT-003"* and the `task-workflow` skill drives the lifecycle.
- **`docs/adr/`** — architecture decisions the code references by number.

Typical loop: `core` → `task-workflow` → `method` (if adding a component) → `review` → `docs-sync`.

## Layout
```
src/structsplat/   structure_tensor, density, sampling (NumPy) · gaussians, render, metrics,
                   init, fit, pyramid, codec, cli (torch)
tests/             pytest (NumPy tests run anywhere; torch tests skip without torch)
benchmarks/        ablation.py (ABL-001), stage_search.py (ABL-002), rate_distortion.py
                   (COMP-001), fitness hooks
docs/              adr/ · architecture.md · theory.md
tasks/             INDEX.md + task files
```

## The open question this repo is built to answer
The optimizer discovers anisotropy on its own, so flanking/tensor init mainly buys **convergence
speed** and **low-budget quality**. Hypothesis (ABL-001): `aniso_flanking ≥ aniso_onedge >
iso_blue_noise > grid > random` at low budgets, with the gap shrinking as the budget grows. If
flanking never wins, the honest move is to prefer the simpler strategy — the benchmark is designed
to tell you either way.

## Verification status
Init-time math is validated numerically in this environment: structure-tensor orientation/labels,
density concentration, WSE exact-count + blue-noise spacing + density adaptivity, unit-area
anisotropy metric, and the conic inverse-covariance + render compositing formulas (NumPy mirror).
The PyTorch modules compile and are covered by tests that run once `torch` is installed
(`pytest -q`); run the smoke test locally to confirm the fit loop end-to-end on your hardware.

## Selected references
GaussianImage (ECCV 2024) · Image-GS (SIGGRAPH 2025) · AIR, Fast-2DGS (2025) · GaussianVision
(structured init) · Li & Wei, *Anisotropic Blue Noise Sampling* (SIGGRAPH Asia 2010) · Yuksel,
*Sample Elimination* (EGSR 2015) · *Gaussian Blue Noise* (2022).

## License
MIT.
