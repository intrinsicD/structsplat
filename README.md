# StructSplat

**Hierarchical, feature-aware, anisotropic blue-noise 2D Gaussian image representation.**

A single image is encoded as a set of oriented 2D Gaussians and rendered by a sorting-free,
normalized weighted-sum rasterizer. The repository's tested structural-prior candidate is:

- a **structure tensor** `J = G_ρ * (∇I ∇Iᵀ)` is the single operator for *density* (where to put
  Gaussians), *orientation* (how to elongate them), and *classification* (flat / edge / corner);
- Gaussians are placed by **anisotropic, density-adaptive blue noise** (Weighted Sample Elimination
  with a Mahalanobis metric) — packing across edges, spreading along them, no clumping, no grid;
- edges are tensor-aligned and density-aware; flanking remains available as a control, but the
  current evidence favors on-edge/quadtree WSE placement over flanking;
- an optional progressive WSE ordering improves audited uniform-set geometric prefixes without
  changing the terminal set; SSPL1 currently Morton-reorders the field, so this is not yet an
  embedded-codec/LOD claim.

The broad ingredients and several close combinations now exist in prior work, including
structure-guided allocation/orientation/precision and progressive Gaussian coding. The unresolved
claim is narrower: whether tensor-metric WSE adds held-out actual-rate value beyond direct
SLIC/Sobel, gradient, uniform, and native controls. `structsplat` is a **placeholder name** —
rename freely (see the `docs-sync` skill).

> This is a PyTorch **research reference** with an opt-in exact CUDA extension for the same
> normalized/additive equations. The remaining production port is a tiled CUDA/Vulkan/RHI path
> (`tasks/PORT-001-cuda-rasterizer.md`, ADR-0011).

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
`local_slic_sobel_control` is a benchmark-only, explicitly local transplant for BENCH-007; its
frozen SLIC assumptions are not presented as upstream Structure-Guided Allocation code.
Samplers: `wse` (blue noise), `floyd_steinberg` (density-map error diffusion),
`dart_throwing` (Poisson disk), `halton`, `cvt`, `farthest_point`, `density_random`,
`jittered_grid`.
Pure-WSE layouts can add `--wse-progressive-order` to permute the identical terminal Gaussian set
into Yuksel-style nested prefixes. It is opt-in because saved row order and GPU reduction order are
part of experimental provenance; the current codec still Morton-sorts the full field. With a
background layer, frozen background rows stay first and only the detail suffix has WSE ordering.
Renderers: `normalized`, `additive`, `cuda`, `cuda_additive`, `gsplat`. `cuda`/`cuda_additive`
are exact StructSplat semantics; `gsplat` is a GaussianImage++-style alpha/sum comparator.
Scale caps: `none`, `hard`, `feature` (ADR-0012).

## Publication method figures

Generate deterministic structure-tensor, tensor-metric sampling, initialized Gaussian, and
normalized-responsibility panels from a real image:

```bash
python scripts/render_paper_figures.py tests/test_images/COCO_train2014_000000000030.jpg \
  --outdir results/paper_method_figure --max-side 256 --num-gaussians 384 --seed 0
```

The bundle includes a vector encoder/decoder overview, individual PNGs, raw NPZ maps, resolved
config, hashes/provenance, and a labeled montage. It is initialization-only explanatory output, not
optimized or comparative evidence. See `docs/publication_figures.md` for panel semantics and the
missing BENCH-007 result-figure queue.

## Agentic workflow (Claude Code)
This repo is built to be implemented *with* Claude Code, mirroring the IntrinsicEngine setup.

- **`CLAUDE.md`** — project guide + a skill-aware routing table.
- **`.claude/skills/`** — seven canonical project skills: `core`, `task-workflow`, `review`,
  `method`, `benchmark`, `docs-sync`, `structsplat-research-ideation`. They're auto-discovered
  inside this repo; run `scripts/install_skills.sh` to symlink them into `~/.claude/skills` for
  global use. `.agents/skills/structsplat-research-ideation` is a relative discovery symlink to
  the same research-ideation tree for Codex/Agent Skills, not a duplicate. The skill is a
  first-party, MIT-licensed adaptation by Alexander Dieckmann of
  `transformational-research-skill-kit` v1.0.0.
- **`tasks/`** — work items (`AREA-NNN-slug.md`) tracked in `tasks/INDEX.md`. Say *"work on
  INIT-003"* and the `task-workflow` skill drives the lifecycle.
- **`docs/adr/`** — architecture decisions the code references by number.

Typical loop: `core` → `task-workflow` → `method` (if adding a component) → `review` → `docs-sync`.
Research discovery starts with `core` → `structsplat-research-ideation`; selected candidates then
enter the normal task/method/benchmark loop.

## Layout
```
src/structsplat/   structure_tensor, density, sampling (NumPy) · gaussians, render, metrics,
                   init, fit, pyramid, codec, visualize, cli (torch)
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

**Current research boundary (2026-07-13).** The completed
`storage_budget_168k_external_present` run is a strong local optimizer/policy diagnostic, but its
nominal payload is 71.68–81.15 bpp at the prepared sizes and its SSPL1 streams are about 22 bpp.
It is not compression-SOTA evidence. Current work already covers broad structure-guided
allocation/orientation/precision, normalized ownership, progressive Gaussian streams, learned
initialization, boundary gating, and clustered quantization. BENCH-007 therefore asks the narrower
question: does tensor-WSE beat SLIC/Sobel, gradient, uniform-WSE, and random controls at
0.25–4.0 **actual** bpp on held-out images? See
`ara/evidence/storage-budget-168k-sota-audit-2026-07-13.md`.

FIT-015 adds opt-in `checkpoint_policy=best_psnr_final_count`. It selects only post-transition
states with the terminal Gaussian count and writes a same-trajectory audit. On COCO4 x seeds 0/1,
640 Gaussians, and 5k steps, 7/8 runs selected an earlier full-count state and improved their own
terminal means by +0.7702 dB PSNR, +0.00892 MS-SSIM, and +0.0076 LPIPS gain. At 500 steps it kept
the terminal state in 7/8 runs and was effectively neutral. A 72-trajectory Kodak4 confirmation
across max-side {160,240,320} and N={1280,2560,5120} gained +0.4884 dB pooled PSNR, but the gain
fell from +1.0380 dB at N=1280 to +0.0458 dB at N=5120. Keep the compute-minimal terminal policy
as the universal default; use checkpoint selection for sparse/moderate-density long-horizon
quality runs. FIT-013's Sobel loss and FIT-014's covariance filter remain experimental and
default-off.

## Verification status
Init-time math is validated numerically in this environment: structure-tensor orientation/labels,
density concentration, WSE exact-count + blue-noise spacing + density adaptivity, unit-area
anisotropy metric, and the conic inverse-covariance + render compositing formulas (NumPy mirror).
The PyTorch modules compile and are covered by tests that run once `torch` is installed
(`pytest -q`); run the smoke test locally to confirm the fit loop end-to-end on your hardware.
The completed available-repository fixed-storage benchmark writes per-image byte sizes,
5,376-Gaussian quality/convergence metrics, cold-decode codec metrics, and explicit completeness
into `results/storage_budget_168k_external_present/index.html`; the portable multi-report entry
point is `results/index.html`. Its analytical and actual rates are reported separately and the
report must not be presented as an actual-rate compression comparison.

**Reproducibility caveat.** Every benchmark writes a `config.json` (resolved args + device +
torch/numpy/structsplat versions + repository commit/dirty diff fingerprint) so a run is
source-bound from its own artifacts. Results are
bit-exact from a seed only on **CPU**: the CUDA renderer accumulates with atomics
(`atomicAdd` / `index_add`), so GPU renders vary run to run — the logged renderer/device/versions
bound that variation. See the `benchmark` skill for the full experimental-validity rules.

## Selected references
GaussianImage (ECCV 2024) · Image-GS (SIGGRAPH 2025) · GaussianImage++ (AAAI 2026) ·
Structure-Guided Allocation (2025) · SAD, SGI, AIR, CGVQ (2026) · P-GSVC · Contour-Aware 2DGS ·
WIPES (ICCV 2025) · Instant-GI · Li & Wei, *Anisotropic Blue Noise Sampling* (SIGGRAPH Asia
2010) · Yuksel, *Sample Elimination* (EGSR 2015).

## License
MIT.
