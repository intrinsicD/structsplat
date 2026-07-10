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
sampling, orientation, color, affine color basis, scale-cap, renderer, pixel loss, optional
pixel-loss weighting, optimizer, factored refinement (`refine_site`, `refine_primitive`,
`refine_nms` plus color/prune/relocate flags), and pyramid stages.
Caveat: factorial marginals are observational when axes co-vary; use `--mode influence` for paired
one-factor deltas around a baseline. Outputs include `stage_search.jsonl`, `stage_search.json`,
`stage_search.csv`, `summary.md`, and a local scalar `index.html` overview.
The sampling axis includes `floyd_steinberg` for the ABL-004 placement-control run.
FIT-004/006/007 densification variants can still be requested with legacy `--refine-modes` aliases
such as `residual_add_nms`, `residual_tensor_add_nms`, `fp_duplicate`, `ranked_wave`, `relocate`,
and `absgrad_wave`, but new sweeps should prefer explicit axes like
`--refine-sites residual residual_tensor --refine-primitives sampled_add moment_preserving`.
FIT-009's difficult-four slice did not promote `residual_tensor x moment_preserving`; keep it as a
searchable combination. Stretch controls also include `optimizer=adan` and the `aa` stage from
`--aa-dilations`. CORE-006 affine colors are exposed as `--color-basis-modes affine`; keep
`constant` as the baseline/default until larger sweeps justify promotion.
FIT-010 adds event color-solve schedules to the `color_solve` axis: `every<N>` remains the
promoted quality arm, while `init`, `final`, `on_split`, and compositions such as
`init+on_split` are available for screening. The FIT-010 smoke did not meet the rule for replacing
`every10`, though `on_split` helped split recovery. FIT-011 adds split-recovery micro-lever axes:
`--state-seed-modes off on`, `--row-temper-modes off warmup<N>`, and
`--support-fade-modes off on until<F>`. The FIT-011 smokes were negative for promotion: state
seeding and young-row tempering did not improve split recovery, and scheduled fade missed
fade-on AUC at 5k/10k despite preserving fade-off final PSNR. INIT-008 adds
`scale_cap=feature_rel`, a local-radius feature cap. The difficult-four fair-density protocol
keeps it searchable but default off: it repaired most old `feature` cap losses but averaged
-0.3733 dB PSNR versus matching uncapped rows. FIT-012 adds
`--loss-weight-modes none tensor` (`tensor_<beta>` accepted) for structure-tensor weighting of the
pixel-loss term only; SSIM and reported metrics remain unweighted. The difficult-four fair-regime
slice keeps it searchable but default off: tensor weighting was PSNR-neutral overall (+0.0061 dB
over 16 pairs), helped `aniso_onedge`, hurt `quadtree_wse`, and lost AUC on average. FIT-008
adaptive count is a global controller rather than a stage axis: add `--adaptive-count`
with `--max-gaussians` and/or `--target-bpp` plus optional `--target-psnr`/`--target-ms-ssim`.
Rows report selected N, raw-attribute bpp, adaptive event counts, and stop reason so fixed-N and
adaptive-N sweeps can be compared fairly.
HIER-003/HIER-004 changed the pyramid read: `pyramid=pyramid` is no longer a final-PSNR loser, and
explicit per-level schedules are available as `--pyramid-level-iters`. On the difficult-four 2k/5k
slice, `--pyramid-level-iters 150 1350` repaired the old 750/750 AUC loss while preserving final
quality (+0.0601 dB vs 750/750 pyramid, +0.0011 AUC vs single). Keep `pyramid=single` as shipped
default until larger confirmation; use 150/1350 as the pyramid quality candidate.
ABL-005 uses two fixed shard scripts to avoid mixing implementation-confounded timing with
decision-grade fitter deltas: `scripts/run_abl005_cuda_native_influence.sh` covers the six
CUDA-native knobs and can support quality/convergence/speed claims, while
`scripts/run_abl005_affine_quality_influence.sh` isolates `color_basis=affine` as
quality/convergence-only until native CUDA affine backward exists.

```
python -m benchmarks.stage_search path/to/images --mode influence --budgets 2048 --iters 500
```

`feedforward_teacher_export.py`, `feedforward_train.py`, and `feedforward_eval.py` are the first
FF-001 learned-predictor data path. The exporter runs a pinned teacher initializer/fitter and saves
fitted `GaussianField` NPZ files plus a manifest. The trainer consumes that manifest, fits a tiny
CNN Gaussian regressor, and writes a `predictor.pt` checkpoint loadable via
`structsplat fit --strategy feedforward`. The evaluator compares learned, tensor-prior, and scratch
warm starts at equal final N and short-refinement iterations.

```
python -m benchmarks.feedforward_teacher_export path/to/images --budget 512 --iters 80 --max-side 160
python -m benchmarks.feedforward_train results/feedforward_teacher_export/teacher_manifest.json
python -m benchmarks.feedforward_eval path/to/images --checkpoint results/feedforward_train/predictor.pt --budget 512 --iters 80
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

`fair_density_control_compare.py` is the density-control-aware matched-policy benchmark. Growth
rows share the same initial Gaussian count, final cap, growth-wave schedule, renderer, fitter,
loss, and target tracking, then vary repo-inspired or StructSplat placement/growth policies. It
always writes a local `index.html` overview. Caveat: this still does not run native external repo
pipelines; it isolates policy differences under one executable fitter.
The default method list starts with `structsplat_best_default`, a pinned Gaussian-image recipe
from the 2026-07-09 matched run: `aniso_onedge` + WSE, feature cap `12@160`, tensor-aware residual
growth, 5 growth waves, and `L1 + 0.3 SSIM`. Keep this row in default comparisons so every run has
the current best-known StructSplat reference even when global CLI loss or growth options change.
Additional default candidate rows explore lower/no SSIM, Charbonnier, tensor-weighted loss, final
color solve, split relocation, and adaptive extra capacity (`1.5x` cap) for reducing absolute diff.
The summary includes a default-promotion check: a candidate must beat the pinned row on paired mean
PSNR, MS-SSIM, AUC, fit seconds, and total seconds before the benchmark default should be updated.

```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  STRUCTSPLAT_INSTANT_GI=/path/to/Instant-GI/quard_image.py \
  python -m benchmarks.fair_density_control_compare --resume
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
`refine_noste` equal-compute control. Rows include the fitted/selected Gaussian count and a
raw-attribute bpp proxy so adaptive-count fits remain auditable in compression tables.

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
