# StructSplat

**Hierarchical, feature-aware, anisotropic blue-noise 2D Gaussian image representation.**

A single image is encoded as a set of oriented 2D Gaussians and rendered by a sorting-free,
normalized weighted-sum rasterizer. The research contribution is the **initialization**:

- a **structure tensor** `J = G_ρ * (∇I ∇Iᵀ)` is the single operator for *density* (where to put
  Gaussians), *orientation* (how to elongate them), and *classification* (flat / edge / corner);
- Gaussians are placed by **anisotropic, density-adaptive blue noise** (Weighted Sample Elimination
  with a Mahalanobis metric) — packing across edges, spreading along them, no clumping, no grid;
- edges are tensor-aligned and density-aware; flanking remains available as a control, but the
  current evidence favors on-edge/quadtree WSE placement over flanking;
- the layout is built **progressively** (coarse→fine), so prefixes act as a level-of-detail stack.

The pieces exist separately in prior work (anisotropic blue noise; structured 2D-GS init;
error-driven densification) but not combined into a progressive 2D-Gaussian image codec. `structsplat`
is a **placeholder name** — rename freely (see the `docs-sync` skill).

> This is a PyTorch **research reference** with an opt-in exact CUDA extension for the same
> normalized/additive equations. The remaining production port is a tiled CUDA/Vulkan/RHI path
> (`tasks/PORT-001`, ADR-0011).

## Install
```bash
pip install -e .                 # torch, numpy, pillow, imageio
pip install -e ".[metrics]"      # optional: lpips, pytorch-msssim
pip install -e ".[gen]"          # optional: diffusers text-to-Gaussian generation
pip install -e ".[dev]"          # pytest, ruff
```

## Quickstart
```bash
# fit one image with the measured high-budget PSNR winner
structsplat fit photo.png --strategy quadtree_wse --num-gaussians 20000 --iters 2000

# progressive (hierarchical) fit
structsplat fit photo.png --pyramid --num-gaussians 20000

# the core experiment: init strategy x budget sweep (writes results/summary.md)
structsplat ablation ./images --budgets 2000 5000 10000 20000 --iters 1500 --target-psnr 35

# full stage-combination screening, including factored refinement site/primitive/NMS axes
structsplat stage-search ./images --budgets 1024 2048 --iters 300 --outdir results/stage_search

# per-stage influence: one-factor-at-a-time deltas vs the baseline (writes influence.md with
# ΔPSNR / ΔMS-SSIM / ΔAUC / Δiters-to-target / Δseconds per stage option, ADR-0010)
structsplat stage-search ./images --mode influence --budgets 2048 --seeds 0 1 2 \
    --iters 500 --target-psnr 30 --outdir results/stage_influence

# long stage-search runs are resumable/shardable; influence HTML marks best paired-delta variants
structsplat stage-search ./images --mode influence --resume --max-new-cells 64 \
    --outdir results/stage_influence

# text-to-Gaussian MVP: sample raster -> fit -> latent SDS refine -> save .npz + PNGs
structsplat generate "flat red calendar app icon" --n 5000 --steps 200 --outdir runs/icon
```
Strategies: `random`, `grid`, `iso_blue_noise`, `aniso_onedge`, `aniso_flanking`.
Additional quadtree strategies: `quadtree_aggregate`, `quadtree_hybrid`, `quadtree_wse`.
Samplers: `wse` (blue noise), `floyd_steinberg` (density-map error diffusion),
`dart_throwing` (Poisson disk), `halton`, `cvt`, `farthest_point`, `density_random`,
`jittered_grid`.
Renderers: `normalized`, `additive`, `cuda`, `cuda_additive`, `gsplat`. `cuda`/`cuda_additive`
are exact StructSplat semantics; `gsplat` is a GaussianImage++-style alpha/sum comparator.
Scale caps: `none`, `hard`, `feature` (ADR-0012).

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
                   (COMP-001), coco_fit_compare.py, cross_repo_matrix_compare.py,
                   optimization_followup.py, quadtree_init_compare.py, fitness hooks
docs/              adr/ · architecture.md · theory.md
tasks/             INDEX.md + task files
```

## The question this repo was built to answer — and the measured answer
The optimizer discovers anisotropy on its own, so flanking/tensor init mainly buys **convergence
speed** and **low-budget quality**. Hypothesis (ABL-001): `aniso_flanking ≥ aniso_onedge >
iso_blue_noise > grid > random` at low budgets, with the gap shrinking as the budget grows. If
flanking never wins, the honest move is to prefer the simpler strategy — the benchmark is designed
to tell you either way.

**Measured answer (2026-07-04..07): the flanking half of the hypothesis is dead; the
structured-placement half stands.** ABL-006 completed the decision-grade successive-halving
confirmation on Kodak-24 + COCO4 at max-side 768, 1500 iterations, exact CUDA, and 3-seed finalist
confirmation (`ara/evidence/abl006-complete-2026-07-07/`). Final PSNR winners are budget-specific:
`aniso_onedge` has the higher mean at 2000 Gaussians, but its paired PSNR lead over
`quadtree_wse` is not significant; `quadtree_wse` is the clear 5000-Gaussian PSNR winner
(+0.0930 dB, 95% CI [+0.0168, +0.1700]); and `quadtree_wse` has a small non-significant PSNR lead
at 10000 (+0.0357 dB, 95% CI [-0.0041, +0.0778]) while `aniso_onedge` has higher MS-SSIM.

Operational status: prefer `quadtree_wse` for high-budget PSNR work and keep `aniso_onedge` as the
low-budget/MS-SSIM alternative. `aniso_flanking`, `quadtree_hybrid`, `iso_blue_noise`, and
Floyd-Steinberg were eliminated at stage 1 by the frozen CI rule. ADR-0013 updates the shipped init
default to `quadtree_wse`; flanking stays available as an explicit control arm. The cross-repo
caveat stands: these are matched policy analogues inside StructSplat's harness, not native external
pipelines. BENCH-005 now has isolated, provenance-checked native GaussianImage++, Image-GS, and
GaussianImage runners. The official-environment Image-GS fixed-N 500-step slice supports
StructSplat on final PSNR/proxy-MS-SSIM, but differing initialization and timing semantics prevent
a strict implementation-dominance claim. At 5k steps, official Image-GS remains a tradeoff: versus
the full-count-checkpoint StructSplat candidate it has higher proxy MS-SSIM, while StructSplat has
higher PSNR and substantially better LPIPS. Native GaussianImage is much faster: at 500 steps it
has not converged, while at 5k it is roughly PSNR-competitive, higher in proxy MS-SSIM, lower in
AUC, and worse in LPIPS than the checkpoint candidate. Full-resolution, multi-budget/time-envelope,
native codec/RD, and learned Instant-GI tracks remain open.

FIT-015 adds opt-in `checkpoint_policy=best_psnr_final_count`. It selects only post-transition
states with the terminal Gaussian count and writes a same-trajectory audit. On COCO4 x seeds 0/1,
640 Gaussians, and 5k steps, 7/8 runs selected an earlier full-count state and improved their own
terminal means by +0.7702 dB PSNR, +0.00892 MS-SSIM, and +0.0076 LPIPS gain. At 500 steps it kept
the terminal state in 7/8 runs and was effectively neutral. Keep the pinned default unchanged
until broader budget/resolution evidence resolves AUC, speed, and metric tradeoffs; use the
checkpoint policy for long-horizon quality runs. FIT-013's Sobel loss and FIT-014's covariance
filter remain experimental and default-off.

## Verification status
Init-time math is validated numerically in this environment: structure-tensor orientation/labels,
density concentration, WSE exact-count + blue-noise spacing + density adaptivity, unit-area
anisotropy metric, and the conic inverse-covariance + render compositing formulas (NumPy mirror).
The PyTorch modules compile and are covered by tests that run once `torch` is installed
(`pytest -q`); run the smoke test locally to confirm the fit loop end-to-end on your hardware.

**Reproducibility caveat.** Every benchmark writes a `config.json` (resolved args + device +
torch/numpy/structsplat versions + repository commit/dirty diff fingerprint) so a run is
source-bound from its own artifacts. Results are
bit-exact from a seed only on **CPU**: the CUDA renderer accumulates with atomics
(`atomicAdd` / `index_add`), so GPU renders vary run to run — the logged renderer/device/versions
bound that variation. See the `benchmark` skill for the full experimental-validity rules.

## Selected references
GaussianImage (ECCV 2024) · Image-GS (SIGGRAPH 2025) · AIR, Fast-2DGS (2025) · GaussianVision
(structured init) · Li & Wei, *Anisotropic Blue Noise Sampling* (SIGGRAPH Asia 2010) · Yuksel,
*Sample Elimination* (EGSR 2015) · *Gaussian Blue Noise* (2022).

## License
MIT.
