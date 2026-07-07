# ABL-006 Stage 1 Complete

Purpose: complete ABL-006 successive-halving stage 1 and record the predeclared elimination
decision before stage 2.

Stage-1 execution:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python -m benchmarks.abl004_confirmation halving-run \
  --outdir results/abl004_confirmation \
  --resume \
  --max-new-cells 312 \
  --target-psnrs 28 30 32 \
  > results/abl004_confirmation/stage1_remaining_2026_07_07.log 2>&1
```

Decision regeneration after editing `abl006_elimination_decisions.json`:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python -m benchmarks.abl004_confirmation halving-run \
  --outdir results/abl004_confirmation \
  --resume \
  --max-new-cells 0 \
  --target-psnrs 28 30 32
```

Stage-1 result:

- Completed cells: 336/336.
- Stage-1 units: 28 images x seeds {0,1} x 6 arms at budget 2000.
- Leader by mean PSNR: `quadtree_wse` at 26.5477 dB.
- `aniso_onedge` is statistically tied with the leader: -0.0004 dB, 95% CI
  [-0.1453, 0.1351].

Decision:

- Survivors for stage 2: `quadtree_wse`, `aniso_onedge`.
- Eliminated after stage 1:
  `aniso_flanking` (-0.1796 dB, CI [-0.2977, -0.0696]),
  `quadtree_hybrid` (-0.1295 dB, CI [-0.2483, -0.0102]),
  `iso_blue_noise` (-0.4722 dB, CI [-0.6335, -0.3138]),
  `floyd_steinberg` (-2.9437 dB, CI [-3.7910, -2.1543]).

Next planned stage:

- Stage 2 expected cells: 112 =
  28 images x budget 5000 x seeds {0,1} x 2 survivor arms.
- Current staged analysis after decision: 336/448 cells complete, 112 missing.
