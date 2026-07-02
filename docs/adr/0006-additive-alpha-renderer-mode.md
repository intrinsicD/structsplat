# ADR-0006: Additive / alpha-compositing renderer mode (alongside the normalized default)

## Context
ADR-0003 chose a normalized weighted-sum renderer `I = Σ c_i G_i / Σ G_i` for per-image fitting:
order-independent, sorting-free, fast, a good partition-of-unity for reconstruction. Two directions
we now want do not fit that model:
- **Generation (GEN-001).** Sampling images as Gaussian sets benefits from occlusion / layering
  semantics — a primitive should be able to sit *in front of* another and cover it. Normalized
  blending has no front/back: every overlapping Gaussian contributes proportionally, so it cannot
  express opaque foreground-over-background composition, which is most of what makes vector-style
  images look intentional. It also mismatches the compositing a pretrained pixel model expects,
  which weakens SDS gradients.
- **Residual / stage-wise refinement (HIER-001, AIR-style).** A Gaussian added to *correct* a region
  must override what is underneath (accumulate on top); normalized blending resists this, which is
  why the current pyramid is densification rather than true residual summation.

## Decision
Add an **opt-in additive / alpha-compositing mode** to `render.py`, selected by a flag. The
normalized mode stays the **default** for fitting and ablations. This does **not** supersede
ADR-0003 — it adds an alternative compositing model for tasks that need ordered coverage.

Give each Gaussian an opacity `a_i ∈ (0, 1]` (a 9th parameter; or reuse a color alpha channel) and
composite either:
- **Additive (default new mode):** `C = Σ_i c_i a_i G_i` (optionally tone-mapped). Sort-free,
  cheapest, unblocks residuals + SDS.
- **Front-to-back alpha (`over`):** `C = Σ_i c_i a_i G_i Π_{j<i}(1 − a_j G_j)` with a per-Gaussian
  order/depth key — matches 3DGS compositing and gives true occlusion. Needs a sort (or a
  differentiable/soft order).

Start with additive; add ordered-alpha behind the same flag only if generation needs real occlusion.

## Consequences
+ Generation (GEN-001) can express opaque layering; SDS sees a compositing model closer to the
  pretrained pixel prior.
+ Enables true residual summation for the pyramid (HIER-001) and AIR-style stage-wise coding.
+ Opacity is a natural pruning / cardinality signal (`a_i → 0` ⇒ removable) — useful for generation
  where the Gaussian count is not fixed a priori.
- Two compositing paths to maintain and test; the NumPy render-formula mirror must cover both.
- Ordered-alpha reintroduces a sort (the thing ADR-0003 removed on purpose) — keep it opt-in and off
  the fitting hot path; prefer additive unless occlusion is required.
- Adds a parameter (opacity): touches `GaussianField` packing, `save`/`load`, and the 8-param
  assumption (8 → 9). Update CORE-002 references and `metrics`/init are unaffected.
- Compositing mode becomes a config axis the benchmark must log (BENCH-001).

## Links
Complements ADR-0003 (normalized default) and extends ADR-0002 (RS params → RS + opacity). Required
by GEN-001. Interacts with HIER-001 (residual mode) and BENCH-001 (log the mode).
