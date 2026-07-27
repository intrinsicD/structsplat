# Repository Context: StructSplat

This is a starting map for ideation, not authority. Verify every claim against
the live tree — especially `CLAUDE.md`, `.claude/skills/{core,method,benchmark,task-workflow}/`,
`docs/architecture.md`, `tasks/INDEX.md`, `src/structsplat/`, `tests/`, and
`benchmarks/` — before relying on it.

## Mission

StructSplat is a hierarchical, feature-aware, anisotropic-blue-noise **2D
Gaussian image representation**. It maps an image to oriented 2D Gaussians,
optimizes the field against the target, and studies initialization, fitting,
hierarchy, generation, and compression as research questions. The repository
is a Python/PyTorch research reference with NumPy initialization math and an
owned exact CUDA extension; Vulkan/IntrinsicEngine RHI integration remains a
port target rather than an implemented backend in this tree (`README.md`,
`pyproject.toml`, `tasks/PORT-001-cuda-rasterizer.md`).

## Substrate (what actually exists to build on)

- **Toolchain:** Python >=3.10, setuptools, PyTorch >=2.1, NumPy >=1.24, Pillow,
  and imageio. The `structsplat` CLI exposes fitting, ablation, stage search,
  and generation. Optional dependencies add perceptual metrics, diffusion
  generation, pytest, and Ruff. The owned C++/CUDA renderer is loaded through
  `torch.utils.cpp_extension`; it is not a CMake or in-repo Vulkan backend.
- **Layered pipeline:** `structure_tensor.py` derives energy, orientation,
  coherence, and feature labels; `density.py` turns image or residual structure
  into a sampling density; `sampling.py` provides exact-N WSE plus alternative
  samplers; `init.py` bridges those NumPy stages into a torch `GaussianField`;
  `fit.py` optimizes through `render.py`; `pyramid.py` adds residual-driven
  coarse-to-fine densification. `codec.py`, `predictor.py`, and `generate.py`
  extend that core toward compression, learned warm starts, and diffusion-guided
  generation (`docs/architecture.md`).
- **Representation contract:** images are `(H,W,3)` float32 in `[0,1]` and
  positions are `(x,y)` pixel coordinates. Gaussians use rotation-and-scale
  parameters; tensor-aligned edge Gaussians elongate along the tangent. The
  shipped renderer is a normalized weighted sum, while additive modes are
  explicit alternatives (`CLAUDE.md`, `.claude/skills/core/SKILL.md`,
  `docs/adr/0002-rs-covariance-parameterization.md`,
  `docs/adr/0003-additive-vs-normalized-renderer.md`).
- **Reference/optimized seam:** the differentiable PyTorch renderer is the
  correctness oracle. `cuda`, `cuda_additive`, `cuda_tiled`, and
  `cuda_tiled_additive` are owned extension modes for the same clipped-support
  equations; tests compare pixels and backward gradients against the reference.
  `gsplat` is deliberately a separate alpha/sum comparator and must not be
  described as renderer-equivalent (`src/structsplat/render.py`,
  `src/structsplat/cuda_render.py`, `tests/test_render.py`, ADR-0011).
- **Measurement substrate:** `benchmarks/ablation.py` supplies the focused
  strategy-by-budget experiment, while `benchmarks/stage_search.py` supplies
  factorial and one-factor-at-a-time influence modes. The harness records
  quality, convergence, timing, count/budget, seed, errors, and resumable
  machine-readable rows. BENCH-005 adds isolated native external-reference
  pipelines; BENCH-006 adds a fixed-storage convergence lane. Treat matched
  policy analogues and native external runs as different evidence classes.
- **Research record:** bounded work lives in `tasks/` and `tasks/INDEX.md`;
  hard-to-reverse decisions live in `docs/adr/`; reproducible run artifacts and
  negative results live under `ara/evidence/`. The existing `structsplat-core`,
  `structsplat-task-workflow`, `structsplat-method`, `structsplat-benchmark`, `structsplat-review`, and `structsplat-docs-sync` skills define
  the handoff from a chosen idea to implementation and evidence.

## High-value research surface (grounded in the active task index)

The active frontier spans unfinished core experiments and stage/fitter evidence
(ABL-001/002/004/005), boundary-gated and hybrid edge primitives (CORE-007/008),
quantization and the compression ladder (COMP-001/003), SDS and VSD/multi-particle
generation (GEN-001/003), exact-CUDA tiling and backward reductions
(PORT-001/002/003), native-reference comparisons (BENCH-005), and fixed-storage
convergence (BENCH-006). Several implemented FIT candidates remain opt-in,
rejected, or promotion-blocked; read their current task status and evidence
instead of assuming that implemented means preferred.

Especially fertile given this substrate:

- new primitives or support rules that improve discontinuities without hiding
  extra capacity or changing the renderer semantics silently;
- convergence mechanisms whose benefit survives equal-final-count, equal-budget,
  and matched-horizon controls;
- diagnostics and experimental designs that distinguish initialization,
  optimization, representation, selection, and implementation effects;
- rate-distortion formulations that connect analytical payload, actual streams,
  perceptual quality, and convergence without mixing those quantities;
- exact or bounded-error GPU formulations whose forward and backward behavior can
  be tested against the reference, including alternatives to atomic reductions;
- generative and learned initialization ideas that can be isolated from the
  pretrained image prior and compared with strong hand-designed starts.

Cross-domain donor fields that map plausibly onto this substrate include spatial
point processes and discrepancy theory, adaptive approximation and multigrid,
inverse problems and optimal transport, rate-distortion and information theory,
optimal experimental design, numerical continuation, and GPU scheduling or
parallel reduction theory. Treat these as search directions, not evidence that a
transfer is new or valid.

## Constraints (respect these or the idea will not land)

- Keep initialization-time `structure_tensor`, `density`, and `sampling` math
  NumPy-based and importable without torch; keep autograd in torch modules.
- Preserve image/coordinate conventions and thread every new random choice from
  logged configuration and `InitConfig.seed`.
- Keep the reference implementation intact as the oracle. Renderer semantics
  must be named explicitly; never report `gsplat` alpha/sum behavior as exact
  normalized-renderer parity.
- Compare methods at equal budgets and state the baseline, horizon, seeds,
  renderer, device, and metric convention. Do not let searched settings stand
  in for shipped defaults or matched-policy analogues stand in for native code.
- GPU accumulation uses atomics and is not bit-reproducible. Record source,
  environment, device, and config fingerprints; do not claim seed-only exactness
  for CUDA runs.
- Treat proxy screens as screening evidence. Promotion/default claims require the
  repository's full fair regime, strong controls, and committed reproducible
  artifacts.
- Never fabricate prior art, citations, results, or novelty. Label ideas
  *candidate*-novel until the prior-art audit clears them.

## Acceptance workflow (where a selected idea goes)

A chosen candidate becomes a short `AREA-NNN-slug.md` task in `tasks/` and a row
in `tasks/INDEX.md`, with a falsifiable goal, interfaces, dependencies, acceptance
criteria, baseline, and abandonment rule. Load `structsplat-core`, then use `structsplat-task-workflow`;
use `structsplat-method` for a swappable initializer, renderer, sampler, hierarchy, primitive,
or optimization method, and `structsplat-benchmark` for the decisive experiment. Implement
the smallest reference path first, add correctness and regression tests under
`tests/`, compare optimized CUDA behavior against the reference when applicable,
and preserve configuration plus seed in the result artifacts. Put reproducible
evidence under `ara/evidence/`, update documentation through `structsplat-docs-sync`, and use
an ADR under `docs/adr/` for a hard-to-reverse representation, renderer, or
architecture decision. The ideation skill itself proposes and audits; it does not
skip this workflow or modify research code automatically.
