# ADR-0011: Owned exact CUDA renderer and gsplat comparator semantics

## Context

StructSplat originally used the PyTorch reference renderer as the only correctness oracle and
treated CUDA rendering as future PORT-001 work. A local GaussianImage++/gsplat path was later added
as an opt-in renderer stage, but its alpha/sum compositing is not equivalent to StructSplat's
normalized weighted-sum reference. It was useful for speed experiments, but it changed the rendered
image under dense overlap and therefore could not be the production replacement for the reference.

The project needed a CUDA path that preserves StructSplat semantics:

- `normalized`: `sum_i c_i o_i G_i / (sum_i o_i G_i + eps)`.
- `additive`: `sum_i c_i o_i G_i`.
- same clipped support windows as `render.py`;
- gradients for means, conics, colors, and optional opacities.

## Decision

StructSplat owns an exact PyTorch CUDA extension under `src/structsplat/cuda/` and exposes it as:

- `renderer=cuda`: exact normalized StructSplat renderer;
- `renderer=cuda_additive`: exact additive StructSplat renderer;
- `renderer=gsplat`: external GaussianImage++ alpha/sum comparator, deliberately not treated as
  equivalent to either reference mode.

The CUDA extension is opt-in and environment-dependent. CPU/reference rendering remains the oracle
and the portable fallback. Tests compare CUDA against the reference path when CUDA and the extension
toolchain are available.

## Consequences

+ Fit-loop performance can be measured without changing the representation or renderer semantics.
+ Cross-repo comparisons can separate "StructSplat exact CUDA" from "gsplat-style comparator".
+ PORT-001 is now partial rather than future-only: exact CUDA exists, but the remaining production
  work is a tiled/culled kernel, better backward reductions, deterministic accumulation options, and
  the IntrinsicEngine RHI pass.
- The extension requires a compatible CUDA/PyTorch/toolchain stack. Some local environments need a
  system `libstdc++` preload for CUDA extension loading.
- Atomic accumulation is not bit-exact across GPU runs; benchmark artifacts must log renderer,
  device, and software versions.

## Links

Amends ADR-0003 and ADR-0006 by adding exact CUDA implementations of their equations. Distinguishes
the gsplat comparator from StructSplat semantics. Updates PORT-001.
