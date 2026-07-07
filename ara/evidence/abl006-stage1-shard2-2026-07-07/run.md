# ABL-006 Stage 1 Shard 2

Purpose: continue the ABL-006 successive-halving stage-1 run with another bounded exact-CUDA shard.

Command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python -m benchmarks.abl004_confirmation halving-run \
  --outdir results/abl004_confirmation \
  --resume \
  --max-new-cells 12 \
  --target-psnrs 28 30 32
```

Scope:

- Stage: 1.
- Existing cells skipped: `kodim01`, budget 2000, seeds {0,1}, all six arms.
- Cells added: 12/12 requested.
- Image added: `kodim02`.
- Budget: 2000.
- Seeds: {0, 1}.
- Arms: `aniso_onedge`, `aniso_flanking`, `quadtree_wse`, `quadtree_hybrid`,
  `iso_blue_noise`, `floyd_steinberg`.

Partial result:

- ABL-006 staged cells complete: 24/336.
- Missing staged cells: 312.
- Cumulative budget-2000 mean PSNR ranks:
  `aniso_onedge` 27.3511, `quadtree_wse` 27.2392, Floyd-Steinberg 26.9542,
  `quadtree_hybrid` 26.9497, `aniso_flanking` 26.8131, `iso_blue_noise` 26.4690.
- A local `inotify_add_watch` warning was printed during the run; the bounded shard completed and
  all 24 rows are present in `ablation.jsonl`.

Status: partial shard evidence only. Stage 1 remains open; no elimination decision is justified
until the planned stage-1 paired units are complete or an explicitly documented stopping decision
is made.
