# StructSplat — Claude Code project guide

Hierarchical, feature-aware, **anisotropic blue-noise 2D Gaussian image representation**. A single
image is encoded as a set of oriented 2D Gaussians; the contribution is the *initialization* —
structure-tensor-driven, anisotropic-blue-noise, progressive — for 2D-Gaussian image codecs.
PyTorch reference; the sampler + rasterizer are the CUDA/Vulkan + IntrinsicEngine port targets.

> `structsplat` is a placeholder project name. If it changes, follow the `docs-sync` skill.

## Skill-aware routing (load the skill, then act)
This repo ships six project skills in `.claude/skills/`. Load them by task — do not reimplement
their guidance inline.

| When you are… | Load skill |
|---|---|
| Starting any session / orienting / deciding where code goes | **core** (always first) |
| Picking up or closing a task from `tasks/` | **task-workflow** |
| Adding/changing an init strategy, renderer, sampler, hierarchy | **method** |
| Running/extending the ablation, or wiring a fitness signal | **benchmark** |
| Reviewing a diff or self-reviewing before commit | **review** |
| A change touches documented behavior, decisions, or task status | **docs-sync** |

Typical flow: `core` → `task-workflow` (open the task) → `method` (if adding a component) →
`review` (before commit) → `docs-sync` (same commit). Explicit invocation: "use the method skill".

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
CUDA tile rasterizer is `PORT-001`.

## Definition of done (short form)
Acceptance criteria tested · NumPy/torch split intact · ADR for any real decision · docs updated in
the same commit · results reproducible.
