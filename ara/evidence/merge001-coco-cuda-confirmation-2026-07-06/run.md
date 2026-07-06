# MERGE-001 COCO/CUDA Confirmation Run

Purpose: close MERGE-001's remaining large-confirmation gate after CUDA and COCO val2017 became available in this workspace.

Command shape:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. python - <<'PY'
# Calls benchmarks.stage_search.run_stage_search once per finalist config.
# Images: first 20 sorted COCO val2017 JPEGs.
# Budgets: 512, 1024. Seeds: 0, 1, 2. Iters: 40. Max-side: 160. Renderer: cuda.
PY
```

Scope:

- 20 COCO val2017 images listed in `images20.txt`.
- 6 merged/finalist StructSplat configs.
- 2 budgets x 3 seeds x 20 images = 120 cells per config, 720 cells total.
- Exact CUDA renderer with local system `libstdc++` preload.

Result:

- 720/720 cells completed with zero errors.
- Overall mean PSNR winner: `codex_stage_top1` at 27.3443 dB, followed by
  `merged_onedge_fast` at 27.2016 dB and `merged_shipped_flanking` at 27.0827 dB.
- Paired vs `merged_shipped_flanking`, `codex_stage_top1` gained +0.2616 dB PSNR
  over 120 pairs, with 108/120 paired PSNR wins.
- Paired vs `merged_shipped_flanking`, `merged_onedge_fast` gained +0.1189 dB PSNR
  over 120 pairs, with 100/120 paired PSNR wins.
- The older `merged_best_exact_cuda` feature-cap/residual-tensor row lost -0.3822 dB
  paired PSNR vs shipped flanking on this larger slice, so it should not be promoted
  as a default from MERGE-001 evidence.

Artifacts:

- `summary.md`: aggregate table, budget splits, and paired deltas.
- `metrics.json`: protocol, aggregate tables, paired deltas, and error list.
- `images20.txt`: exact COCO val2017 image list.

