# ABL-004 Confirmation Shard 6

Purpose: continue the settled default ABL-004 confirmation protocol in bounded, resumable exact-CUDA
shards.

Run:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python -m benchmarks.abl004_confirmation run \
  --outdir results/abl004_confirmation --resume --max-new-cells 18
```

Scope: default confirmation manifest, 1,512 expected cells =
28 images x 3 seeds x 3 budgets x 6 variants. This shard added the next 18 cells:
`kodim02`, budget 10000, all six variants, seeds {0,1,2}.

Partial result:

- Completed cells: 108/1,512.
- Missing cells: 1,404.
- Errors: 0.
- The `kodim02` block is now complete across confirmation budgets {2000, 5000, 10000}.
- Mean PSNR at budget 10000 on this single-image shard:
  `aniso_onedge` 35.7905, `quadtree_wse` 35.7700, `iso_blue_noise` 35.7505,
  `quadtree_hybrid` 35.7436, Floyd-Steinberg 35.5835, `aniso_flanking` 35.5701.
- Mean iterations to PSNR 35.0 on this shard:
  `aniso_onedge` 229.7, `quadtree_wse` 233.3, `quadtree_hybrid` 246.3,
  `aniso_flanking` 260.3, `iso_blue_noise` 274.3, Floyd-Steinberg 320.0.
- After this shard, the aggregate 10000-Gaussian confirmation leaderboard spans 6/84
  expected runs per strategy. It ranks `quadtree_wse` first at 31.8131 mean PSNR,
  followed by `iso_blue_noise` 31.7717 and `aniso_onedge` 31.7442.
- Paired against `aniso_onedge` across the current six 10000-Gaussian units,
  `quadtree_wse` is +0.0688 dB with 3/6 wins, `iso_blue_noise` is +0.0274 dB
  with 3/6 wins, and Floyd-Steinberg is -0.0918 dB with 1/6 wins.

Status: partial shard evidence only. The `kodim02` 10000-Gaussian slice itself narrowly favors
`aniso_onedge`, while the two-image aggregate at 10000 Gaussians narrowly favors `quadtree_wse`.
ABL-004 remains open until the remaining confirmation cells are run.

Live artifacts: `results/abl004_confirmation/index.html`,
`results/abl004_confirmation/confirmation_analysis.md`,
`results/abl004_confirmation/summary.md`,
`results/abl004_confirmation/leaderboard.csv`,
`results/abl004_confirmation/paired_deltas_vs_baseline.csv`,
`results/abl004_confirmation/missing_cells.csv`.
