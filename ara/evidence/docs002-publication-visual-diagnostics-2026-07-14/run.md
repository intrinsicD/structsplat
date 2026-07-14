# DOCS-002 publication visual diagnostics

Generated from the pinned repository image
`tests/test_images/COCO_train2014_000000000030.jpg` with:

```bash
PYTHONPATH=src python scripts/render_paper_figures.py \
  tests/test_images/COCO_train2014_000000000030.jpg \
  --outdir ara/evidence/docs002-publication-visual-diagnostics-2026-07-14 \
  --max-side 256 \
  --strategy aniso_onedge \
  --num-gaussians 384 \
  --seed 0 \
  --candidate-oversample 4
```

The panels expose the production initialization and normalized-renderer equations. They are
explanatory diagnostics only: the field is not optimized, this is not a held-out comparison, and
the bundle provides no rate-distortion or ranking evidence. See `manifest.json` for exact source,
configuration, implementation, environment, repository, and artifact hashes.

Validation at generation time:

- focused visualization/tensor/renderer suite: 92 passed;
- complete repository suite: 449 passed;
- Ruff checks for the new module, CLI, and tests: passed.
