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
- `observation_field.py` / `pixel_contraction.py` / `progressive_residual_quadtree.py` /
  `artifact_first_quadtree.py` / `overlap_elimination.py` / `contraction_refinement.py` /
  `residual_exchange.py` — default-off Field V2 semantic oracle plus the HIER-005
  reverse-contraction, HIER-006 retained-parent hierarchy, HIER-007 parent-replacing frontier,
  HIER-008 exact-overlap/feature-elimination controls, HIER-009 dynamic overlap contraction,
  HIER-010 guarded coefficient refinement, and HIER-011 exact-count residual exchange.
  Topology remains NumPy-first; coefficient fitting imports torch lazily. HIER-006 is a negative
  exposed-image prefix control because retained ancestors consumed most of its budget. HIER-007
  recovers those active rows, but its frozen 2x2 C0001 screen rejects artifact-first/overlap-local
  reconciliation: all four 8k arms fail the artifact gate, and the combined arm exposes severe
  quadtree-aligned artifacts plus prohibitive reference work. HIER-008 finds stable exact overlap
  prefitting and a large positive overlap factor for quadtree contraction, but every cell still
  fails locally and fixed-scale WSE/Schur survivors produce dot holes. HIER-009's 3x3 recovery halo
  removes the obvious low-count block lattice and protection helps patch error, but it redistributes
  error at 8k and every overlap cell still fails the local gate; only the HIER-005 delta/touched 8k
  fallback passes. HIER-010's fixed residual reserve loses quality, while its touched-row solve is
  safe but negligible. HIER-011 repairs exposed local tails but misses its transfer gain floor.
  HIER-012's large exposed all-row projection gain does not survive HIER-013's 16-image screen:
  most cells fail closed on the frozen coefficient bound and the diagnostic bundle also fails
  renderer parity. None enters the maintained pipeline.
- `codec_native_field.py` / `realtime_gs_adapter.py` / `realtime_gs_surface_lift.py` /
  `realtime_gs_ray_posterior.py` / `realtime_gs_coherent_depth.py` —
  CORE-016/ADR-0032's separate default-off
  dual-plane packet experiment. A charged conventional image payload is decoded into a continuous
  cardinal-prefiltered Gaussian lattice; an independent sparse nonnegative Field V2 measure owns
  lift proposals. The lazy adapter must expose both as a structural field/query-backend pair and
  may keep structural metadata/cameras on CPU while placing indexed structure plus appearance
  queries on CUDA. An exposed 23-view matched-10k follow-up supports downstream development utility
  at lower teacher-input bytes, but residual halos/blur/floaters and worse complete lift/training
  resources prohibit continuous-quality, general compression, speed, artifact-free, or BENCH-019
  claims. CORE-017's placement-only alpha-support wrapper and first-maximum surface lift improve
  exposed fixed-5k quality, alpha localization, early convergence, and query work, but residual
  trailing smear/double silhouettes and blur fail its native visual gate; surface cover alone is
  negative. CORE-018's source-excluded DINOv2/local coarse/fine ray posterior improves its raw
  initial PSNR but has near-maximal entropy and sparse reciprocal support on a disjoint karate
  scene; the full arm fails its frozen support floor, while the no-reciprocal arm remains a visually
  smeared volume and loses the fixed-prefix convergence comparison. Retain it only as a negative
  control: no threshold rescue or maintained integration. CORE-019's pinned-VGGT successor adds
  calibration-grouped coherent depth, known-ray fusion/support, hard feature anchors, dynamic WSE,
  bounded post-selection contraction, and compatible depth-normal cover. Its full arm changes the
  raw-known-ray metric/count tradeoff without a uniform win, then fails step-zero/fixed-prefix/
  terminal-control gates and native review with sheets, streaks, floaters, holes, and erased detail.
  Retain its field/compiler only as another
  default-off negative control. None of these modules enters maintained conversion.

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
- New task -> `tasks/AREA-NNN-slug.md` **and** a row in `tasks/INDEX.md`, same commit; regenerate
  the derived `tasks/SESSION-BRIEF.md`. See `tasks/TEMPLATE.md` and
  `structsplat-task-workflow`.
- One-off experiment driver -> `scripts/experiments/`, not the top level of `scripts/`.
- New ADR -> `docs/adr/NNNN-title.md`, and cite it as `ADR-NNNN` from the code or task that
  depends on it (an uncited ADR fails `docs_sync`).

## Verification gate
`./scripts/verify.sh` runs `ruff check`, the portable pytest gate
(`-m "not slow and not integration"`), and five structural checkers: `docs_sync.py`,
`check_ara.py`, `check_task_policy.py`, `check_script_layout.py`,
`check_agent_workflow.py`. CI mirrors it on CPU. `check_report_bundle.py` is the parameterized
on-demand gate for maintained reports. The broader lint/format ratchet is tracked by `DOCS-004`,
not deferred indefinitely.

## Naming
`structsplat` is a **placeholder project name** — if it changes, update `pyproject.toml`, imports,
`README`, and this file in one commit (see `structsplat-docs-sync`).
