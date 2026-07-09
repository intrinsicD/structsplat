# ABL-005 CLI unblock smoke

Date: 2026-07-09.

Purpose: verify that the public `structsplat stage-search` console script can run a sharded
ABL-005-style influence pass and write the normal benchmark artifact set, including `index.html`.
This is a tiny CPU workflow smoke, not decision-grade fair-regime evidence.

Command:

```bash
structsplat stage-search \
  tests/test_images/COCO_train2014_000000000009.jpg \
  tests/test_images/COCO_train2014_000000000025.jpg \
  --mode influence \
  --budgets 64 \
  --seeds 0 \
  --iters 12 \
  --max-side 48 \
  --strategies quadtree_wse \
  --tensor-operators central \
  --tensor-colors luma \
  --density-modes structure variance \
  --sampling-modes wse \
  --orientation-modes tensor \
  --color-modes bilinear \
  --scale-modes spacing \
  --scale-cap-modes none \
  --background-modes off \
  --opacity-modes none constant \
  --renderers normalized \
  --aa-dilations 0.0 \
  --color-basis-modes constant \
  --color-solve-modes none every10 \
  --pixel-losses l1 charbonnier \
  --loss-weight-modes none \
  --optimizers adam \
  --lr-schedules none cosine \
  --refine-modes none moment_preserving \
  --pyramid-modes single \
  --target-psnrs 28 30 32 \
  --target-psnr 28 \
  --split-every 6 \
  --split-count 8 \
  --chunk 8 \
  --outdir ara/evidence/abl005-cli-unblock-2026-07-09 \
  --device cpu
```

Result:

- 14/14 cells completed successfully: 2 images x 1 budget x 1 seed x 7 influence configs.
- Wrote `config.json`, `stage_search.jsonl`, `stage_search.json`, `stage_search.csv`,
  `summary.md`, `influence.md`, and `index.html`.
- Baseline mean over the smoke cells: PSNR 17.874, MS-SSIM 0.80550, AUC 17.240.
- Largest smoke delta was `color_solve=every10` at +1.033 dB PSNR and +0.275 AUC, with +0.090 s
  fit time. Treat this only as a workflow sanity check; the task still requires exact CUDA,
  max-side 768, 1500 iterations, budgets {2000, 5000, 10000}, seeds {0, 1}.

Remaining blocker:

The full seven-knob ABL-005 run still needs either native CUDA affine-color backward or a split
protocol that measures the six CUDA-native knobs separately from the affine quality-only arm.
