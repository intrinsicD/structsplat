---
name: method
description: Use when adding or changing a StructSplat *method* — a new initialization strategy, a rasterizer variant, a sampler, or a hierarchy scheme — so it stays comparable to existing ones and plugs into the ablation. Trigger on "add a strategy", "new init/renderer/sampler", or editing init.py / render.py / sampling.py / pyramid.py.
---

# Adding a method

A "method" is any swappable component the research compares. Keep the **interface fixed** so the
ablation can hold everything else constant and vary one thing (this is what makes results mean
something and feeds the co-scientist fitness cleanly).

## Add an initialization strategy (most common)
1. Implement a branch in `init.build_field` (or a helper it calls) returning a `GaussianField`.
   Reuse `structure_tensor` + `density` + `sampling`; do not re-detect features ad hoc.
2. Add its name to `init.STRATEGIES`.
3. Register it in `benchmarks/ablation.py` so it enters the sweep automatically.
4. Colors init from the **target** image; positions/orientation from the (possibly residual) tensor.
5. Add a test asserting it returns `num_gaussians` (±grid rounding) Gaussians and finite params.

## Add a renderer / sampler variant
- Keep the **reference** intact as the correctness oracle; add the variant behind a flag or a new
  function. New numeric behavior needs a NumPy mirror or closed-form check (`review`).
- A renderer that changes the compositing model (e.g. additive for AIR-style residuals) needs an
  ADR — it interacts with the pyramid and metrics.

## Comparability rules
- One independent variable per ablation axis. If a method needs a different budget/iters to be
  fair, expose it as config, don't hardcode.
- Log the full config + seed with every result. Same fitter, same metric, same images across a row.
