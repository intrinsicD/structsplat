# Publication figures and visual diagnostics

`structsplat.visualize` and `scripts/render_paper_figures.py` generate a deterministic explanatory
bundle for the existing structure-tensor/WSE initializer and normalized renderer. The default is
`aniso_onedge` because that exposes the narrow tensor-metric mechanism under study; the shipped
high-budget PSNR default remains `quadtree_wse`.

## Regenerate the pinned example

Install the working tree (`pip install -e .`) and run:

```bash
python scripts/render_paper_figures.py \
  tests/test_images/COCO_train2014_000000000030.jpg \
  --outdir ara/evidence/docs002-publication-visual-diagnostics-2026-07-14 \
  --max-side 256 \
  --strategy aniso_onedge \
  --num-gaussians 384 \
  --seed 0 \
  --candidate-oversample 4
```

Run `python scripts/render_paper_figures.py --help` for crop, tensor operator/color space,
density, glyph, ellipse, and montage controls. The implementation is CPU-capable; it does not run
the fitter.

## Bundle contents

The bundle contains lossless individual PNG panels and a labeled montage for:

- input, tensor energy, coherence, flat/edge/corner classes, and tangent/normal glyphs;
- the exact density PMF, initialized sites, fixed-display unit-area tensor-metric shapes, and
  actual one-sigma RS Gaussian ellipses;
- the initial normalized reconstruction and initial absolute error; and
- the renderer denominator, effective contributor count, responsibility entropy, and dominant
  owner.

It also includes `method_overview.svg`, a vector encoder/decoder diagram that marks the structure
tensor, density, and WSE analysis as source-only and the SSPL1 field state as the only transmitted
input to cold decode.

`diagnostics.npz` stores raw maps and field attributes. `config.json` stores all resolved controls.
`manifest.json` records the source hash, hashes of the exact production modules used, environment
and repository provenance, coordinate and angle conventions, display transforms, diagnostic
identity checks, and hashes for every generated non-manifest artifact. Direct module hashes cover
untracked working-tree source that a Git diff hash cannot see. The NPZ member order/timestamps and
PNG settings are fixed so repeated runs from the same source/config/state have identical output
hashes.

## Interpretation rules

- Positions use `(x,y)` pixel coordinates. `theta` is the Gaussian `sx` axis; tensor-aligned `sx`
  follows the edge tangent.
- Energy, density, and denominator PNGs use disclosed robust display scaling. The raw NPZ arrays,
  not PNG colors, carry numerical values.
- White ellipses in the sampling panel show the *shape* of the unit-area tensor metric at a fixed
  display radius. They are not Gaussian supports and are not transmitted geometry.
- Ellipses in the Gaussian panel are the actual initialized RS covariances at one standard
  deviation.
- Effective contributor count is `(sum_i w_i)^2 / sum_i w_i^2`; responsibility entropy is
  `-sum_i r_i log(r_i)` for normalized weights `r_i`.
- These panels describe initialization and the forward operator. They are not optimized
  reconstructions, held-out comparisons, RD evidence, or a method ranking.

## Figures still requiring experiments

The generator intentionally does not synthesize missing results. BENCH-007 must still produce:

1. equal-count/rate causal allocation comparisons and zooms;
2. actual-rate RD curves with per-image points, intervals, and component bytes;
3. edge/texture errors and signed cross-edge bleed at matched rate;
4. convergence/search/encode/decode resource curves; and
5. predeclared success, median, and failure examples.

The complete publication/readiness and panel-level plan is in
`ara/evidence/publication-readiness-research-2026-07-14.md`.
