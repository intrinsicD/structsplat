# GEN-003: VSD / multi-particle distillation

**Status: todo (after SDS baseline).** Modernize the generative path beyond vanilla SDS.

## Context
GEN-001 starts with score distillation because it is the shortest path to generating Gaussian
fields from prompts. Vanilla SDS is known to over-saturate, over-smooth, and collapse diversity.
VSD-style guidance and multi-particle distillation are the more plausible quality path once the
basic renderer-to-diffusion gradient path works.

## Goal
Add VSD guidance and a multi-particle field optimization mode to `structsplat generate`.

## Approach
1. Keep GEN-001's vanilla SDS as the debugging baseline.
2. Add `guidance="vsd"` using a LoRA or lightweight score adapter to model the current particle
   distribution.
3. Optimize multiple Gaussian fields per prompt with diversity-promoting interactions instead of
   one mean-seeking sample.
4. Compare against `raster-sample -> fit -> refine` and vanilla SDS in the same prompt suite.

## Acceptance criteria
- [ ] `structsplat generate` exposes `--guidance sds|vsd` and `--particles K`.
- [ ] VSD path keeps the frozen base diffusion model frozen and trains only the adapter/LoRA.
- [ ] Multi-particle mode saves each field, preview PNG, seed, prompt, and guidance metadata.
- [ ] Metrics include CLIP score, image diversity, saturation statistics, and qualitative contact
      sheets for the prompt suite.
- [ ] Core install/tests still pass without the optional `[gen]` dependencies.
- [ ] Documentation states when SDS is only a baseline and when VSD is expected.

## Interfaces touched
`src/structsplat/generate.py`, `src/structsplat/config.py`, `src/structsplat/cli.py`,
`pyproject.toml`, generation tests/smokes behind optional dependencies.

## Depends on
GEN-001.
