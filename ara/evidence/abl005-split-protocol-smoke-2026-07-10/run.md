# ABL-005 split-protocol smoke — 2026-07-10

Purpose: verify the two ABL-005 shard scripts run through `benchmarks.stage_search`, write
`influence.md`, and write a local `index.html` overview after splitting the CUDA-native and affine
quality-only protocols.

## Commands

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
OUTDIR=results/abl005_cuda_native_influence_smoke BUDGETS="16" SEEDS="0" ITERS=4 \
MAX_SIDE=32 TARGET_PSNRS="10" DEVICE=cuda \
scripts/run_abl005_cuda_native_influence.sh \
  tests/test_images/COCO_train2014_000000000009.jpg \
  tests/test_images/COCO_train2014_000000000025.jpg

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
OUTDIR=results/abl005_affine_quality_influence_smoke BUDGETS="16" SEEDS="0" ITERS=4 \
MAX_SIDE=32 TARGET_PSNRS="10" DEVICE=cuda \
scripts/run_abl005_affine_quality_influence.sh \
  tests/test_images/COCO_train2014_000000000009.jpg \
  tests/test_images/COCO_train2014_000000000025.jpg
```

## Result

- CUDA-native shard completed 14/14 cells and wrote
  `results/abl005_cuda_native_influence_smoke/index.html` plus `influence.md`.
- Affine quality-only shard completed 4/4 cells and wrote
  `results/abl005_affine_quality_influence_smoke/index.html` plus `influence.md`.
- The standalone `benchmarks.stage_search` parser now accepts `--background-modes`, matching the
  public `structsplat stage-search` wrapper and the protocol scripts.
- This is a plumbing smoke only: tiny 32px crops, 16 Gaussians, one seed, four iterations. It is not
  promotion evidence.
