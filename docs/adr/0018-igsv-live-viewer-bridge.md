# ADR-0018: External igsv live-viewer bridge (diagnostic-only 3D embedding)

## Context

Watching a fit converge required rendering PNGs (`visualize`) or waiting for the run to
finish. The sibling `interactive-gs-viewer` repository provides `igsv`: a Python server
that streams packed splat snapshots over a binary WebSocket to a WebGPU browser client,
built for live 3D Gaussian-splatting training. StructSplat's Gaussians are 2D image-plane
primitives under a *normalized* compositor (ADR-0003), so displaying them in a 3D
alpha-compositing viewer needs an embedding convention and a clear evidential status.

## Decision

Add `structsplat.viewer` (torch-side) bridging `GaussianField` to igsv, plus an opt-in
read-only observer seam in `fit()` (`iteration_observer`, `observer_every`, forwarded by
`fit_pyramid`), wired to `structsplat fit --live`. The embedding: positions
`(x, (H-1)-y, 0)` (y-up, so `theta` maps to a rotation of `-theta` about +Z), RS scales
with `filter_variance` folded in in quadrature and a constant 0.01 px z-thickness,
`sigmoid(opacity logits)` (1.0 when absent), clamped base colors as SH DC.

The browser view is a **diagnostic, never evidence**: the viewer alpha-composites (OVER)
while the product renderer is a normalized weighted sum, and local affine color carriers
(`color_grads`) are not reproduced. Quantitative claims continue to use the exact
renderers and the benchmark protocol. `igsv` stays an optional dependency, imported
lazily; the observer runs under `no_grad`, must not mutate the field or touch RNG state,
and defaults off (`observer_every=0`), leaving the established fit path byte-identical.

## Consequences

- Live convergence/floaters/orientation inspection during a fit at browser frame rates,
  including over LAN, with zero cost when unused (guarded by a cadence check per
  iteration).
- A second compositing model is now visible to users; the divergence from ADR-0003 is
  deliberate and documented — anyone tempted to eyeball-compare renders is redirected to
  the exact rasterizer.
- The observer seam is generic (any `Callable[[GaussianField, int, float], None]`), so
  future tooling (e.g. streaming metrics) reuses it instead of adding more fit hooks.
- Rules out: making igsv a hard dependency, and 2D-native rendering in the browser client
  (would require a normalized-compositor render mode in igsv; revisit only with a use
  case that a diagnostic embedding cannot serve).
