---
name: structsplat-core
description: Load at the start of any StructSplat work session. Repo map, invariants, naming, and the layered architecture (structure-tensor init -> anisotropic blue-noise sampling -> RS Gaussians -> normalized renderer -> fit -> pyramid). Use when orienting in the codebase, deciding where code belongs, or before editing any module.
---

# StructSplat — core conventions

Feature-aware, hierarchical, anisotropic **blue-noise 2D Gaussian image representation**. Research
reference in PyTorch; the rasterizer + sampler are the pieces later ported to CUDA/Vulkan and into
IntrinsicEngine as an RHI pass.

## Layer map (data flows top to bottom)
- `structure_tensor.py` (NumPy) — J = G_rho * (grad I grad I^T); eigen-analysis gives **density**,
  **orientation**, and **flat/edge/corner** labels. One operator, three jobs (ADR-0004).
- `density.py` (NumPy) — energy -> density pmf; residual density for pyramid levels.
- `sampling.py` (NumPy) — Weighted Sample Elimination: exact-N blue noise, density-adaptive via
  per-point radius, anisotropic via a per-point metric tensor (ADR-0005), with opt-in
  terminal-set-preserving progressive survivor order (INIT-009).
- `gaussians.py` (torch) — `GaussianField`, RS parameterization, conics, radii (ADR-0002).
- `render.py` (torch) — differentiable normalized-weighted-sum rasterizer, no sort (ADR-0003).
- `init.py` (torch bridge) — the five strategies in `STRATEGIES` (the ablation variables).
- `fit.py` (torch) — Adam fitter, L1+SSIM, records PSNR history + iters-to-target; its
  `active_row_count` hook keeps capacity-sized field/moment storage while rendering, projection,
  and Adam updates use a contiguous active prefix (FIT-024/ADR-0021).
- `pool.py` / `safe_schedule.py` (torch) — fixed-capacity storage is independent of topology:
  FIT-021 owns mutable triage/free-list policy, while FIT-024 keeps the transactional proposal
  auction and immutable active-prefix liveness. FIT-025/ADR-0022 separately controls physical
  capacity, the ordinary active ceiling, and an opt-in Pareto-gated late detail reserve; defaults
  preserve the historical single ceiling with no tail. FIT-031/ADR-0029 adds a separate
  default-off terminal error-only tail whose effective-support estimate may grow dynamic storage
  before fixed-topology convergence. FIT-040/ADR-0030 adds a mutually exclusive default-off
  masked fine-detail pursuit tail; `detail_pursuit.py` owns its deep high-pass selector, metrics,
  and inherited-row-frozen exact partial color solve.
- `pipeline.py` / `workflows.py` — freeze the source-bound 2026-07-24 operational profile once,
  keep masked and unmasked execution identical outside boundary-specific work, and power the sole
  conversion CLI (`scripts/convert.py`) plus three report-producing evaluation workflows.
- `pyramid.py` (torch) — progressive densification driven by residual structure tensor.
- `codec.py` / `cli.py` (torch at command time) — native NPZ and self-describing SSPL1 persistence;
  `fit`/`image-to-gaussians2d` saves native fields, while `render`/`gaussians2d-to-image`
  reconstructs NPZ or SSPL1 with optional display-referred error metrics and read-only
  fitted-field overlays.

## Invariants (do not break without an ADR)
1. Init-time math stays **NumPy and importable without torch**. Autograd stays in torch modules.
2. Images are `(H, W, 3)` float32 in `[0, 1]`; positions are `(x, y)` in pixel coords.
3. RS parameterization: `theta` = angle of the `sx` axis; edge Gaussians elongate **along** the
   tangent (`sx = s_along`, `sy = s_across`).
4. The renderer is **normalized** (divides by summed weight). Residual work is densification, not
   additive summation — if you need additive, write ADR-0006 first.
5. Every experiment is reproducible: seed flows InitConfig.seed -> RNG; log the config with results.

## Where things go
- New init strategy -> `init.py` + add to `STRATEGIES` + register in `benchmarks/ablation.py`. See `structsplat-method`.
- New metric -> `metrics.py`, wired through `fit.py` output dict and `BENCH-001`.
- Perf/CUDA -> `PORT-001`; keep the NumPy/torch reference as the correctness oracle.
- Routine conversion/evaluation -> `scripts/convert.py`, `benchmark.py`, `ablation.py`, or
  `stage_search.py`; task-specific historical launchers live in `deprecated_scripts/`.
- New claim / refuted claim -> a row in `ara/logic/claims.md`; a not-yet-promoted finding -> an
  `O<NN>` entry in `ara/staging/observations.yaml`. See "Evidence and claims" in `CLAUDE.md`.
- New task -> `tasks/AREA-NNN-slug.md` **and** a row in `tasks/INDEX.md`, same commit.
- One-off experiment driver -> `scripts/experiments/`, not the top level of `scripts/`.
- New ADR -> `docs/adr/NNNN-title.md`, and cite it as `ADR-NNNN` from the code or task that
  depends on it (an uncited ADR fails `docs_sync`).

## Verification gate
`./scripts/verify.sh` runs `ruff check`, the portable pytest gate
(`-m "not slow and not integration"`), and four structural checkers: `docs_sync.py`,
`check_ara.py`, `check_task_policy.py`, `check_script_layout.py`. CI mirrors it on CPU. The
broader lint/format ratchet is tracked by `DOCS-004`, not deferred indefinitely.

## Naming
`structsplat` is a **placeholder project name** — if it changes, update `pyproject.toml`, imports,
`README`, and this file in one commit (see `structsplat-docs-sync`).
