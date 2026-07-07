# ABL-006 Stage 1 Shard 1

Purpose: start the ABL-006 successive-halving confirmation with a bounded, resumable exact-CUDA
stage-1 shard.

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
- Cells added: 12/12 requested.
- Image: `kodim01`.
- Budget: 2000.
- Seeds: {0, 1}.
- Arms: `aniso_onedge`, `aniso_flanking`, `quadtree_wse`, `quadtree_hybrid`,
  `iso_blue_noise`, `floyd_steinberg`.

Partial result:

- ABL-006 staged cells complete: 12/336.
- Missing staged cells: 324.
- On this two-seed `kodim01` slice, mean PSNR ranks:
  `aniso_onedge` 23.4572, `quadtree_wse` 23.3023, Floyd-Steinberg 22.9219,
  `quadtree_hybrid` 22.9047, `aniso_flanking` 22.4775, `iso_blue_noise` 22.3247.
- Paired vs `aniso_onedge`, `quadtree_wse` is -0.1550 dB PSNR with 0/2 wins;
  Floyd-Steinberg is -0.5353 dB with 1/2 wins.

Status: partial shard evidence only. Stage 1 remains open until the remaining 324 cells are run
or restored from a compatible resumable result store. No elimination decision is justified from
this shard alone.
