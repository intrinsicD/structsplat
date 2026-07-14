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

## Experimental status

The generator intentionally does not synthesize missing results. The four-image Stage-0a run has
now exercised every panel below and validated the complete figure pipeline, including missing-cell
handling and the separately scaled conventional-codec inset. Its visible `plumbing_only` banner is
part of the claim guard: those plots are layout/implementation evidence, not paper results.

Publication-grade BENCH-007 evidence must still produce the following on the preregistered Stage-1
pilot and, only if its gate passes, the untouched Stage-2 confirmation set:

1. equal-count/rate causal allocation comparisons and zooms;
2. actual-rate RD curves with per-image points, intervals, and component bytes;
3. edge/texture errors and signed cross-edge bleed at matched rate;
4. convergence/search/encode/decode resource curves; and
5. predeclared success, median, and failure examples.

The complete publication/readiness and panel-level plan is in
`ara/evidence/publication-readiness-research-2026-07-14.md`.

## BENCH-007 result-figure generator

`benchmarks.actual_rate_phase_diagram` now implements the missing F5--F9 computations. It does not
invent panels when cells are absent: a generated placeholder states the exact missing evidence.
With a completed frozen run, `analyze` emits:

- `f5_causal_allocation.png`: same image, Gaussian count, codec mix, and fit horizon; source-only
  initial sites above cold-decoded reconstructions with actual rate/PSNR below;
- `f6_actual_rate_phase_diagram.png`: raw per-image measured envelopes, image-cluster intervals,
  exact SSPL1 header/attribute bytes, and separately styled conventional-codec context when run;
- `f7_mechanism.png`: per-image paired edge/texture MSE, signed target-normal bleed, and effective
  edge contributor deltas at frozen target rates;
- `f8_resources.png`: full equal candidate-search cost, decoder timing, and equal-horizon fit
  trajectories; and
- `f9_qualitative_quantiles.png`: failure/median/success chosen by paired-PSNR quantiles, with crops
  selected from target gradients and one shared error scale.

Every result directory also contains the frozen manifest, append-only journals, raw/selected CSV
and JSON, retained streams/reconstructions, statistical/gate summary, and a relative-link-only
`index.html`. Stage-0 and Stage-1 banners remain visible on every figure so plumbing or killing-pilot
output cannot be mistaken for held-out evidence.

The completed Stage-0a audit is recorded in
`ara/evidence/bench007-stage0a-plumbing-2026-07-14/run.md`; the large streams, reconstructions, and
figures remain under the ignored `results/bench007_stage0a_20260714/` run directory with committed
artifact hashes in that evidence note.
