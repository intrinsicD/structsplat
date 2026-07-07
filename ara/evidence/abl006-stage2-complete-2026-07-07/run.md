# ABL-006 stage 2 complete

Date: 2026-07-07

Command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python -m benchmarks.abl004_confirmation halving-run \
  --outdir results/abl004_confirmation \
  --resume \
  --max-new-cells 112 \
  --target-psnrs 28 30 32
```

After recording the stage-2 decision, the staged analysis was regenerated without launching new
cells:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python -m benchmarks.abl004_confirmation halving-run \
  --outdir results/abl004_confirmation \
  --resume \
  --max-new-cells 0 \
  --target-psnrs 28 30 32
```

Result: ABL-006 is 448/728 staged cells complete. Stage 1 covered all six arms at budget 2000,
seeds {0,1}; stage 2 covered the two survivors at budget 5000, seeds {0,1}. The regenerated plan
now has 280 missing cells: stage 3 at budget 10000 for both finalists, plus seed 2 for both
finalists at budgets 2000, 5000, and 10000.

Stage-2 decision: `quadtree_wse` leads budget 5000 at 29.7977 dB mean PSNR versus
`aniso_onedge` at 29.7097 dB. The paired delta for `quadtree_wse - aniso_onedge` is +0.0881 dB
with 95% CI [-0.0169, +0.1946], so the frozen CI rule does not eliminate `aniso_onedge`.
Finalists for stage 3 are `quadtree_wse` and `aniso_onedge`.

Key artifacts:

- `abl006_elimination_decisions.json`
- `abl006_elimination_trail.csv`
- `abl006_plan.csv`
- `missing_cells.csv`
- `leaderboard.csv`
- `pairwise_deltas.csv`
- `paired_deltas_vs_baseline.csv`
- `confirmation_analysis.json`
- `confirmation_analysis.md`
- `summary.md`
- `index.html`
- `stage2_2026_07_07.log`
