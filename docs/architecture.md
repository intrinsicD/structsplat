# Architecture

## Entrypoint (ADR-0025/0028)

`structsplat.pipeline.run_pipeline` is the maintained composition of the current best pipeline,
and `scripts/convert.py` is its sole supported conversion CLI. `PipelineConfig`'s defaults are the
measured recipe (C25/C50/C51/C52) plus the evidence-bound `0.75` px Janelle mask margin (C56).
They are deliberately *not* the conservative library default surface in `config.py`
(ADR-0009/0013). Passing a mask selects the arm and nothing else does:

```
image (+ optional mask)
      │
      ▼
scripts/convert.py → pipeline.run_pipeline
      ├── mask ──► masked arm: quadtree-WSE interior + boundary-tangent rows
      │                         (CORE-011), containment on (ADR-0017/0019)
      └── none ──► full-frame arm: same init, mask machinery degenerate
      │
      ▼
safe_schedule.run_safe_schedule
   bootstrap → coverage growth → detail growth → [boundary/general closure]
   → redistribution → polish
   every optimizer block and topology proposal runs on a detached trial field and is committed
   only if a full-frame metric vector is Pareto-safe (FIT-023/024/025, ADR-0020..0023)
```

The full-frame arm degenerates rather than forks: `mask.signed_distance` clips an empty complement
to the image diagonal, so caps are inert and the boundary band is empty. Boundary initialization,
containment, losses, metrics, and proposals are disabled; count-matched general coverage/detail
proposals occupy the same closure slot and budget (ADR-0027). The full-frame arm has no benchmark
screen yet (BENCH-017).

The layered reference path below is what both arms are built from, and what `structsplat fit`
exposes directly as the knob-level research command.

## Pipeline
```
image (H,W,3) in [0,1]
      │
      ▼
structure_tensor.compute ──► StructureTensor{ lam1,lam2, across_edge_angle, coherence, energy, label }
      │                                   │                 │                         │
      │ energy                            │ eigenvectors    │ eigenvalue pattern      │
      ▼                                   ▼                 ▼                         │
density.py  ── pmf ──►  sampling.eliminate (WSE)  ◄── anisotropy_metric ◄─────────────┘
                          │  exact-N blue noise, density- & anisotropy-adaptive
      ▼
init.build_field ──► GaussianField{ means, log_scales, rotations, colors }   (RS params)
                          │
                          ▼
fit.fit  ──►  render.render (normalized weighted sum, differentiable)  ──►  Adam (L1 + SSIM)
                          │
                          ▼
pyramid.fit_pyramid: level 0 from image density; finer levels add Gaussians where the *residual*
structure tensor has energy (densification); append order = coarse→fine = LOD prefix.

pipeline.run_current_pipeline: frozen safe schedule; masked = boundary specialization,
unmasked = identical counts/stages with general closure and no boundary-specific work.
```

## Module responsibilities
- **NumPy, init-time, no autograd:** `structure_tensor` (selectable central/sobel/scharr operator;
  luma or Di Zenzo rgb color space), `density` (structure/gradient/variance/hybrid/uniform modes +
  the inverse-CDF warp for low-discrepancy samplers), `sampling` (WSE blue noise, Poisson-disk
  dart throwing, farthest-point, CVT/Lloyd, Halton, and opt-in terminal-set-preserving progressive
  WSE order), `config`, `mask` (CORE-010/011: exact separable EDT / signed distance / erosion /
  nearest-inside feature transform / boundary color dilation / smoothed-SDF boundary normals for
  mask-contained fitting and boundary coverage).
- **benchmark-only structural controls:** `structural_controls` lazily calls SLIC and keeps the
  SLIC/Sobel complexity ranking, exact-N 6:2:1 allocation, and unresolved upstream-fidelity
  assumptions explicit. `init` registers `local_slic_sobel_control`, but it is not a shipped
  default or an upstream-paper implementation.
- **torch, autograd:** `gaussians` (RS + optional opacity + optional per-Gaussian scale caps,
  ADR-0012), `render` (normalized default + additive, ADR-0006, exact CUDA variants, ADR-0011,
  and gsplat comparator, sharing one accumulator where semantics match), `metrics`, `init`
  (bridge; `build_masked_field` for CORE-010), `fit` (selectable loss/optimizer/LR-schedule/
  split-mode; opt-in mask containment via `_MaskConstraint` with isotropic ADR-0017 or certified
  anisotropic ADR-0019 caps, under-coverage penalty, boundary tangent densification; opt-in
  FIT-022 coverage-matching regularizer — mass-neutral `(S−c)²` on the raw weight sum with
  detached opacities, feature/boundary/error targets and cosine decay), `pool` +
  `triage` (FIT-021/ADR-0020, opt-in via `triage_every`: fixed-capacity pooled row lifecycle with
  off-image parking, byte-budgeted capacity from `target_file_bytes`, and one in-place
  park→merge→split→spawn event replacing the independent topology timers); `pool` also provides
  FIT-024/ADR-0021's immutable active-prefix storage for `safe_schedule`, where preallocation is
  independent of topology policy, state checkpoints retain full field/Adam capacity, Adam update
  kernels use the active shape, and one terminal compaction restores the ordinary `GaussianField`
  interface. FIT-025/ADR-0022 separates that physical capacity from the ordinary active ceiling
  and adds an opt-in post-color-solve reserve whose covered-interior high-frequency births/splits
  remain transactional and Pareto-gated; FIT-026/ADR-0023 adds the opt-in `geometric` storage
  policy that grows physical capacity by `growth_factor` toward `capacity` on demand instead of
  preallocating it, preserving the live prefix so the fit stays bit-identical to `fixed_capacity`),
  `pyramid`, `pipeline` (CORE-012/ADR-0025: the single current-best recipe and matched
  masked/full-frame arm selection), `workflows` (ADR-0027: four folder/report orchestrators,
  registered ablations/stage variants, and optional native-baseline subprocesses),
  `codec`
  (post-fit quantization, ADR-0007; optional in-container alpha stream for masked inputs,
  ignored by pre-FIT-021 decoders).
- **read-only diagnostics:** `visualize` calls the production NumPy analysis/initialization and
  torch normalized renderer, then exports raw tensor/field/responsibility maps plus deterministic
  explanatory panels. It never fits or changes a field and is not benchmark evidence (DOCS-002).
  
  `viewer` bridges a `GaussianField` to the external igsv browser viewer (optional dependency)
  for live fit inspection via `fit(iteration_observer=..., observer_every=...)`; the embedding
  and its diagnostic-only status are ADR-0018.
- **entry:** `scripts/convert.py` is the sole current-best conversion CLI;
  `scripts/{benchmark,ablation,stage_search}.py` are evaluation workflows. All four write portable
  report bundles. `deprecated_scripts/` retains evidence-bound launchers without presenting them
  as supported interfaces (ADR-0028).
- **entrypoint:** `pipeline` (CORE-012/ADR-0025/0028) owns the maintained best-pipeline recipe and
  the masked/full-frame arm selection; it composes `init`, `fit`'s mask constraint, and
  `safe_schedule`, and holds no fitting mechanism of its own. `safe_schedule` (FIT-023/024/025)
  owns the phase order, the topology auction, and the Pareto-safe commit gate; ADR-0020..0023 own
  its storage policies.
- **entry:** `cli` (`structsplat fit` /
  `image-to-gaussians2d`, `render` /
  `gaussians2d-to-image`, `batch-fit`, `ablation`, `stage-search`); `render` cold-loads a native
  full-precision NPZ or self-describing SSPL1 stream and can emit display-referred error/metrics
  plus a read-only fitted-field ellipse overlay. `batch` (PORT-005) runs the `fit` option surface
  across worker processes with device round-robin and a resumable `metrics.jsonl`. The optional
  `fit --live` path remains diagnostic-only.

- **decision benchmark:** `benchmarks.actual_rate_phase_diagram` owns frozen actual-rate manifests,
  SSPL1 cold scoring, exact-cap RDO/statistics, and result figures for BENCH-007. Its manifest
  distinguishes the normalized weighted-sum equation from the selected implementation; native
  scientific runs may freeze the parity-checked owned exact-CUDA implementation explicitly.
  Persisted-stream parity is checked on decoded field state before a single cold render; two CUDA
  renders are not used as an equality oracle because atomic accumulation is not bit-reproducible.
  Result-figure stream replay uses the validated analysis device, so CUDA-frozen semantics are not
  silently forced through CPU tensors. The completed Stage-1 gate is negative; this substrate is
  reusable, but Stage 2 is not authorized for the current tensor-WSE claim.

## Stage-search (ABL-002, protocol in ADR-0010)
`benchmarks/stage_search.py` sweeps configurations across every swappable stage — tensor operator,
tensor color space, density mode, sampling mode, orientation mode, init strategy, color mode,
scale mode, opacity, renderer, loss, optimizer, LR schedule, factored refinement
(`refine_site`, `refine_primitive`, `refine_nms`, sampled-add score, plus
the opt-in normalized-responsibility mass exponent and color/prune/relocate flags), pyramid — in
two modes:
**factorial** (full product, ranked, for the best complete config) and **influence**
(one-factor-at-a-time paired deltas vs the baseline = first value of each axis; emits
`influence.md` with ΔPSNR/ΔMS-SSIM/ΔAUC/Δiters-to-target/Δseconds per stage option). Configs
whose differing stage is provably inert are canonicalized and deduplicated. Every row records
quality (PSNR/MS-SSIM/LPIPS), convergence (iters-to-target, PSNR-AUC), and speed (init/fit
seconds) so max-quality, max-convergence-rate, and max-speed candidates can be read from the same
run. The shipped defaults (ADR-0009 plus ADR-0013's init-default update) are one named cell in
that space; everything else is a candidate the screening can promote. `benchmarks/ablation.py`
(ABL-001) stays the focused
init-strategy × budget sweep.

## Performance notes (reference is the oracle; these keep it usable at N~20k on CPU)
- `sampling.eliminate` builds the WSE conflict graph vectorized over grid-cell offsets (only the
  greedy heap removal stays in Python); the anisotropic search reach is bounded per receiver by the
  metric's minimum eigenvalue, so no long-range along-edge conflict is missed. ~30x faster than the
  original per-pair Python loops at N=20k.
- `render` evaluates each Gaussian on the axis-aligned bounding box of its `sigma_cutoff` ellipse
  (per-axis radii `(rx, ry)`), laid out as one ragged flat tensor — no padding to a shared square
  tile. Elongated anisotropic Gaussians get a tight rectangle instead of a square sized by the major
  axis (~3x forward speedup on a flanking init). Still fully differentiable; radii stay detached.
- `render`/`conics` take an optional EWA-style `aa_dilation` (Sigma + d·I) low-pass for sub-pixel
  Gaussians — off by default; exact under RS since it only shifts the per-axis variances.
- `renderer=cuda` and `renderer=cuda_additive` call StructSplat's owned exact CUDA extension for
  the same clipped-support equations. The internal `cuda_block_reduce` selector preserves the
  exact forward equation and replaces only the untiled backward reduction; PORT-004 keeps it
  benchmark-only after the frozen all-grid/stability gate failed. `renderer=gsplat` is kept as a
  separate alpha/sum comparator because it is not numerically equivalent to the normalized
  reference.
- `renderer=cuda_tiled` (opt-in, PORT-002/003, locally parity-validated but performance-unmeasured)
  builds its tile index inside the
  extension (CUB radix sort over packed 32-bit keys; stable, so the index is deterministic),
  stages Gaussians through shared memory in both tiled kernels, warp-reduces backward gradients
  before atomics, and — under `support_fade` only — exactly culls (tile, Gaussian) pairs whose
  weight is provably zero via a closed-form conic-over-rectangle minimum. Semantics are
  unchanged; `benchmarks/tiled_render_profile.py` owns the preregistered acceleration gate, and
  `cuda` remains the shipped GPU default until that gate passes on hardware.
- `scale_cap_mode=feature` gives each Gaussian a local support ceiling from the structure tensor's
  feature run length. `scale_cap_mode=feature_rel` instead derives the cap from local density
  radius / quadtree leaf side with separate along/across multipliers. The fitter clamps optimized
  scales to the field-owned cap, preventing long edge spikes without changing the renderer
  equation. Both cap modes are searchable and default off after INIT-008's fair-density negative.

## Extension seams
- Init strategies: `init.STRATEGIES` (the ablation variables).
- Renderer variants (e.g. additive for AIR-style residuals): behind ADR-0006, keep reference oracle.
- Performance: `PORT-001` CUDA tile rasterizer → IntrinsicEngine RHI pass; reference stays the oracle.
- Feed-forward init predictor (`FF-001`) and compression codec (`COMP-001`) attach after the fitter.
