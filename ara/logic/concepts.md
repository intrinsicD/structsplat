# Concepts

## Pixel-Gaussian endpoint

- **Definition**: For a sampled 2D RGB image, associate one narrow peak-one Gaussian basis with
  every active pixel center and its RGB sample. With sufficiently small declared width, the sampled
  render is numerically near the source image, while explicit per-row storage is a worst-rate
  endpoint. A compact method can be studied as a sequence of accepted contractions away from this
  endpoint without materializing trainable torch rows for every leaf.
- **Boundary**: Finite-width Gaussian overlap means the endpoint is not automatically a
  mathematical interpolant, and a single sampled image does not define a 3D novel-view field.
  Implicit leaves remove parameter/optimizer storage, not the need to read pixels or maintain
  contraction bookkeeping. Estimated row bytes do not establish compression.
- **Provenance**: user
- **Crystallized via**: artifact-commitment
- **Evidence**: [`src/structsplat/pixel_contraction.py`,
  `tests/test_pixel_contraction.py`,
  `tasks/HIER-005-implicit-pixel-contraction.md`]
- **From staging**: O105

## Opacity-split gauge equivalence

- **Definition**: In StructSplat's normalized renderer, replace one row by co-located copies with
  identical non-opacity attributes and opacity fractions whose sum is one. Numerator, denominator,
  and rendered pixels are preserved up to reduction roundoff.
- **Allocator consequence**: When the responsibility mass clamp is inactive, child score
  `E/M^alpha` scales as `f^(1-alpha)`. Alpha 1 preserves each row value but duplicates top-k
  tickets; alpha 0.7 also changes row magnitude. Aggregating `E` and `M` over a certified exact
  group before the exponent restores representation-invariant group scores.
- **Boundary**: Ordinary moment-preserving split siblings move and shrink, so ancestry alone is not
  an exact gauge class. Commutation is a correctness property; FIT-019 does not establish recovery,
  performance, compression, or expressiveness utility.
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Evidence**: [`ara/evidence/fit019-opacity-gauge-2026-07-15/`,
  `benchmarks/gauge_equivalence_audit.py`, `docs/adr/0014-keep-opacity-gauge-groups-benchmark-only.md`]
- **From staging**: O55
