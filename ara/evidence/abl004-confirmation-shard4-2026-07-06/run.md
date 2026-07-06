# ABL-004 Confirmation Shard 4

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
`kodim02`, budget 2000, all six variants, seeds {0,1,2}.

Partial result:

- Completed cells: 72/1,512.
- Missing cells: 1,440.
- Errors: 0.
- Mean PSNR at budget 2000 on this single-image shard:
  `aniso_onedge` 31.2651, `quadtree_wse` 31.1883, Floyd-Steinberg 30.8809,
  `iso_blue_noise` 30.8091, `quadtree_hybrid` 30.7088, `aniso_flanking` 30.5071.
- Mean iterations to PSNR 30.0 on this shard:
  `aniso_onedge` 62.3, `aniso_flanking` 63.3, `quadtree_wse` 65.7,
  `quadtree_hybrid` 66.0, `iso_blue_noise` 106.0, Floyd-Steinberg 124.3.
- After this shard, the aggregate 2000-Gaussian confirmation leaderboard spans 6/84
  expected runs per strategy. It ranks `aniso_onedge` first at 27.3161 mean PSNR,
  followed by Floyd-Steinberg 27.0533 and `quadtree_wse` 27.0492.
- Paired against `aniso_onedge` across the current six 2000-Gaussian units,
  `aniso_flanking` is -0.7015 dB with 0/6 wins, `quadtree_wse` is -0.2669 dB
  with 1/6 wins, and Floyd-Steinberg is -0.2629 dB with 1/6 wins.

Status: partial shard evidence only. The `kodim02` 2000-Gaussian slice again favors
`aniso_onedge`, but ABL-004 remains open until the remaining confirmation cells are run.

Live artifacts: `results/abl004_confirmation/index.html`,
`results/abl004_confirmation/confirmation_analysis.md`,
`results/abl004_confirmation/summary.md`,
`results/abl004_confirmation/leaderboard.csv`,
`results/abl004_confirmation/paired_deltas_vs_baseline.csv`,
`results/abl004_confirmation/missing_cells.csv`.
