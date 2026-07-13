# StructSplat — Claude Code project guide

Hierarchical, feature-aware, **anisotropic blue-noise 2D Gaussian research substrate**. A single
image is represented by oriented 2D Gaussians under a normalized compositor. The current
defensible research candidate is narrow: a training-free tensor-metric/WSE structural prior whose
value must be established at actual coded rate against direct SLIC/Sobel, gradient, uniform, and
native controls. Structure-aware allocation, orientation, progressive coding, and generic
Gaussian compression are not blanket novelty claims. See BENCH-007 and the 2026-07-13 SOTA audit.
PyTorch reference plus exact CUDA; the sampler/rasterizer remain CUDA/Vulkan + IntrinsicEngine port
targets.

> `structsplat` is a placeholder project name. If it changes, follow the `docs-sync` skill.

## Skill-aware routing (load the skill, then act)
This repo ships seven project skills in `.claude/skills/`. Load them by task — do not reimplement
their guidance inline.

| When you are… | Load skill |
|---|---|
| Starting any session / orienting / deciding where code goes | **core** (always first) |
| Picking up or closing a task from `tasks/` | **task-workflow** |
| Adding/changing an init strategy, renderer, sampler, hierarchy | **method** |
| Developing novel research directions, cross-domain transfers, or falsifiable research portfolios | **structsplat-research-ideation** |
| Running/extending the ablation, or wiring a fitness signal | **benchmark** |
| Reviewing a diff or self-reviewing before commit | **review** |
| A change touches documented behavior, decisions, or task status | **docs-sync** |

Typical flow: `core` → `task-workflow` (open the task) → `method` (if adding a component) →
`review` (before commit) → `docs-sync` (same commit). Explicit invocation: "use the method skill".
For open-ended research discovery: `core` → `structsplat-research-ideation`; a selected candidate
then re-enters `task-workflow` → `method` → `benchmark` → `review` → `docs-sync`.

## Non-negotiable invariants (full list in the `core` skill)
1. Init-time math (`structure_tensor`, `density`, `sampling`) is **NumPy and importable without
   torch**. Autograd lives in torch modules only.
2. Images `(H,W,3)` float32 in `[0,1]`; positions `(x,y)` pixel coords.
3. The renderer is **normalized** (ADR-0003). Additive/residual compositing requires a new ADR.
4. Everything reproducible from a logged config + `InitConfig.seed`.

## Layout
`src/structsplat/` package · `tests/` pytest · `benchmarks/` ablation + fitness ·
`docs/adr/` decisions · `docs/architecture.md`, `docs/theory.md` · `tasks/` work items + `INDEX.md`.

## Environment
`pip install -e .` (torch, numpy, pillow, imageio). Optional metrics: `pip install -e ".[metrics]"`
(lpips, pytorch-msssim). Dev: `pip install -e ".[dev]"` then `pytest -q`.
Reference code is CPU-correct but slow at large N; use GPU and small budgets while iterating. The
remaining production/tiled CUDA/Vulkan/RHI work is `PORT-001`/002/003; ADR-0011 owns the exact
CUDA research renderer.

## Definition of done (short form)
Acceptance criteria tested · NumPy/torch split intact · ADR for any real decision · docs updated in
the same commit · results reproducible.
