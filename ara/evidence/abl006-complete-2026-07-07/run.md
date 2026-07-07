# ABL-006 complete

Date: 2026-07-07

Final stage command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python -m benchmarks.abl004_confirmation halving-run \
  --outdir results/abl004_confirmation \
  --resume \
  --max-new-cells 280 \
  --target-psnrs 28 30 32
```

After recording the final decision, the analysis was regenerated without launching new cells:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python -m benchmarks.abl004_confirmation halving-run \
  --outdir results/abl004_confirmation \
  --resume \
  --max-new-cells 0 \
  --target-psnrs 28 30 32
```

Result: ABL-006 completed 728/728 staged cells with 0 missing cells. The run covers the full
Kodak-24 + COCO4 fixture at max-side 768, 1500 iterations, exact CUDA rendering, and the frozen
successive-halving rule from `abl006_elimination_decisions.json`.

Stage decisions:

- Stage 1, budget 2000, seeds {0,1}: `quadtree_wse` and `aniso_onedge` survived. `aniso_flanking`,
  `quadtree_hybrid`, `iso_blue_noise`, and `floyd_steinberg` were eliminated by strictly negative
  PSNR CIs versus the stage leader.
- Stage 2, budget 5000, seeds {0,1}: `quadtree_wse` led, but `aniso_onedge` was not eliminated
  because the stage-2 CI still overlapped zero.
- Final seed-2 confirmation: both finalists were measured at budgets 2000, 5000, and 10000.

Final PSNR conclusion:

- Budget 2000: `aniso_onedge` has the higher mean PSNR, 26.5552 dB vs 26.5064 dB for
  `quadtree_wse`, but the paired PSNR CI overlaps zero.
- Budget 5000: `quadtree_wse` is the clear PSNR winner, 29.8172 dB vs 29.7243 dB, with paired
  delta +0.0930 dB and 95% CI [+0.0168, +0.1700].
- Budget 10000: `quadtree_wse` has the higher mean PSNR, 32.6211 dB vs 32.5854 dB, but the paired
  PSNR CI overlaps zero: +0.0357 dB, 95% CI [-0.0041, +0.0778].

Secondary metric note: at budget 10000, `aniso_onedge` has higher MS-SSIM. The paired
`quadtree_wse - aniso_onedge` MS-SSIM delta is -0.000768 with 95% CI [-0.001022, -0.000522].

Operational conclusion: use `quadtree_wse` as the primary high-budget PSNR default, especially at
5000 Gaussians. Keep `aniso_onedge` as a documented alternative when MS-SSIM is prioritized or the
budget is very low.

Key artifacts:

- `abl006_elimination_decisions.json`
- `abl006_elimination_trail.csv`
- `abl006_plan.csv`
- `ablation.jsonl`
- `leaderboard.csv`
- `paired_deltas_vs_baseline.csv`
- `pairwise_deltas.csv`
- `rank_stability.csv`
- `confirmation_analysis.md`
- `summary.md`
- `index.html`
- `stage1_remaining_2026_07_07.log`
- `stage2_2026_07_07.log`
- `stage3_2026_07_07.log`
