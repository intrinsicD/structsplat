# Benchmarks

All benchmark scripts write machine-readable rows plus enough resolved configuration to make a run
self-describing. See the `benchmark` skill for the full protocol.

For any substantial benchmark or visual-audit result that produces plots, reconstructions, or
comparison grids, also write a local `index.html` overview in the result directory. It should embed
the key diagrams/images, link the raw CSV/JSON artifacts, and state whether visuals are original
saved renders or matched reruns. This keeps ignored `results/` artifacts inspectable without
requiring the reader to hunt through subfolders.

`common.py` is shared benchmark plumbing (`BENCH-003`): image load/save, trajectory PSNR AUC,
JSON/CSV row writing, seed/config helpers, and shared COCO comparison analogue builders. It is not
a CLI; scripts import it to avoid drifting helper behavior.

The canonical four-image COCO fixture used by the matched comparison and regression-bisect
harnesses lives in `tests/test_images/`. Keep those four files there so benchmark reruns do not
depend on ignored `results/` artifacts.

`ablation.py` runs the core experiment (`ABL-001`): `{init strategy} x {budget}` on fixed images,
scored on PSNR / MS-SSIM / LPIPS + iterations-to-target. Caveat: this is the broad init sweep, so
keep image/budget/seed axes explicit in the output config. ABL-004 control labels are available
alongside the core strategies: `floyd_steinberg`, `density_random`, and `random_relocate`.
Long runs write `ablation.jsonl` incrementally; use `--resume` to skip cells already present there.
For the ABL-004 protocol, `scripts/run_abl004_full_ablation.sh` prepares Kodak-24 under
`results/datasets/abl004`, appends the pinned COCO fixtures, and launches the resumable full sweep.
Set `MAX_NEW_CELLS=N` on either wrapper to execute a bounded shard and stop cleanly after `N`
new cells. The ABL-004 wrapper defaults to `RENDERER=cuda` for the owned exact CUDA renderer; set
`RENDERER=normalized` to reproduce the slower PyTorch reference timing.

```
python -m benchmarks.ablation path/to/images --budgets 2000 5000 10000 20000 --iters 1500 --target-psnr 35 --max-side 768 --renderer cuda --resume
```

Outputs `ablation.json`, `ablation.csv`, `summary.md`. `fitness(rows, strategy, budget)` exposes the
scalar a co-scientist loop maximizes over init/sampling variants.

`abl004_confirmation.py` is the decision-grade wrapper for the post-screen confirmation set. It
materializes the expected-cell manifest, runs bounded/resumable shards through `ablation.py`, and
analyzes existing rows into missing-cell reports, leaderboards, pairwise/bootstrap paired deltas,
per-image/seed baseline-loss rows, and rank-stability tables. The default protocol is Kodak-24 plus
the four pinned COCO fixtures, seeds 0/1/2, budgets 2k/5k/10k, and the six current
finalist/control variants.

```
python -m benchmarks.abl004_confirmation plan --outdir results/abl004_confirmation
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  python -m benchmarks.abl004_confirmation run --outdir results/abl004_confirmation \
  --resume --max-new-cells 50
python -m benchmarks.abl004_confirmation analyze --outdir results/abl004_confirmation
```

`stage_search.py` runs `ABL-002`: factorial or influence-mode sweeps across tensor, density,
sampling, orientation, color, scale-cap, renderer, loss, optimizer, refinement, and pyramid stages.
Caveat: factorial marginals are observational when axes co-vary; use `--mode influence` for paired
one-factor deltas around a baseline.
The sampling axis includes `floyd_steinberg` for the ABL-004 placement-control run.
FIT-004 residual densification variants are exposed as refine arms such as `residual_add_nms`,
`residual_tensor_add_nms`, `fp_duplicate`, `ranked_wave`, `relocate`, and
`absgrad_wave`; stretch controls also include `optimizer=adan` and the `aa` stage from
`--aa-dilations`.

```
python -m benchmarks.stage_search path/to/images --mode influence --budgets 2048 --iters 500
```

`cross_repo_matrix_compare.py` is the current matched comparison harness (`ABL-004` controls and
cross-repo evidence): it runs StructSplat-current plus GaussianImage/Image-GS/Instant-GI analogues
over image x resolution x iteration x seed slices. Caveat: the rows are executable policy
analogues under StructSplat's fitter/renderer, not native external CUDA/codec/checkpoint runs.
It also includes `structsplat_shipped_defaults` so searched StructSplat settings are not reported
as if they were the public defaults.

```
python -m benchmarks.cross_repo_matrix_compare --max-sides 160 240 --iters 80 200 --seeds 0 1
```

`coco_fit_compare.py` is the legacy four-image matched comparison harness (`BENCH-003` back-compat
only). Caveat: it is superseded by `cross_repo_matrix_compare.py`; keep it for reproducing older
ARA evidence that names `results/coco_fit_compare`.

```
python -m benchmarks.coco_fit_compare --budget 512 --iters 80 --seeds 0 1
```

`optimization_followup.py` runs bounded follow-up checks after stage-search evidence (`ABL-002`
follow-up): held-out validation, oversampling, early-stop, pyramid/refine, and spatial-render
prototypes. Caveat: it is not a full factorial search; candidates are hand-picked exact configs.

```
python -m benchmarks.optimization_followup --dataset-dir path/to/train2014 --image-count 8 --budget 512 --iters 80
```

`quadtree_init_compare.py` compares quadtree aggregate/hybrid/WSE init variants and scale caps
(`INIT-003/INIT-006` follow-up evidence). Caveat: it reuses optimization-followup candidate
construction, so it is a focused init comparison, not a complete stage search.

```
python -m benchmarks.quadtree_init_compare --dataset-dir path/to/train2014 --image-count 8 --budget 512 --iters 80
```

`init_spectral_analysis.py` is the placement-only INIT-003 calibration harness. It builds initial
fields without fitting, writes radial FFT spectra, pair-correlation/nearest-neighbor spacing,
edge-local anisotropy signatures, and realized coherence -> axis-ratio sweep metrics.

```
python -m benchmarks.init_spectral_analysis path/to/images --num-gaussians 2048 --max-axis-ratios 2 4 6 8 --coherence-powers 0.5 1 2
```

`rate_distortion.py` evaluates the codec/QAT path (`COMP-001/COMP-003`) and records full
codec/render semantics per row. Caveat: QAT rows spend extra optimization; compare them with the
`refine_noste` equal-compute control.

```
python -m benchmarks.rate_distortion path/to/images --budgets 2000 5000 --iters 1500 --qat-iters 150
```

`regression_bisect.py` is the ABL-003 forensic runner. It reads the pinned four-image COCO subset
from `tests/test_images/`, evaluates historical commits in detached worktrees, and writes compact
evidence under `ara/evidence/`. Caveat: the child runner is intentionally self-contained so old
detached worktrees do not need today's benchmark helpers. `--download` only refreshes missing
fixture images in `tests/test_images/`.

```
python -m benchmarks.regression_bisect --device cpu
```
