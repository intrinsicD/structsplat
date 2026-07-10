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
quality/convergence-only until native CUDA affine backward exists. The CUDA-native runner accepts
per-axis env overrides (`DENSITY_MODES`, `OPACITY_MODES`, `COLOR_SOLVE_MODES`, `PIXEL_LOSSES`,
`LR_SCHEDULES`, `REFINE_MODES`) so slow arms such as `color_solve=every10` can be run as separate
resumable shards without hand-writing the long stage-search command.

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
color solve, split relocation, LR stabilization, same-final-count checkpoint selection, and
adaptive extra capacity (`1.5x` cap) for reducing absolute diff. The experimental
`structsplat_best_checkpoint_lowpass2x_f10` arm adds frequency-ordered supervision to the
checkpoint control: it trains against a 2x area-lowpass target initially and cosine-blends to the
full target by 10% of the global horizon.
The summary includes a default-promotion check: a candidate must beat the pinned row on paired mean
PSNR, MS-SSIM, AUC, fit seconds, and total seconds before the benchmark default should be updated.
It also writes `default_dominance.csv` and a compact strict-dominance table. Deltas are expressed as
candidate gains over the pinned default (positive is always better), and 95% confidence intervals
bootstrap source images after averaging correlated seeds/budgets within image. The audit labels
candidate dominance, default dominance, tradeoffs, inconclusive evidence, and over-budget rows;
it does not turn repository-inspired analogue rows into native results. Displayed metric intervals
are marginal; a reported dominance relation uses Bonferroni-adjusted bounds for 95% familywise
coverage across the five core metrics and only complete paired cells.
Every fair row also carries the source/decoded-pixel hashes, repository commit/tracked-diff hash,
and hashes of the harness plus critical fit/config/render/metric/init sources. On `--resume`, only
successful rows whose complete cell key and canonical scientific-protocol hash match the current
request. That hash covers every experiment axis, metric request, device/environment version, and
source fingerprint while excluding execution-only sharding controls. Summaries and `metrics.jsonl`
are compacted to current rows so stale reruns cannot be cross-paired or attributed to the newly
written `config.json`.

`structsplat_best_checkpoint` sets `checkpoint_policy=best_psnr_final_count`. Checkpoint scoring is
post-transition (after the optimizer step and any prune/grow/relocate/on-split solve), and only a
state with the terminal Gaussian count may be restored. The legacy pre-step convergence history,
AUC, iteration count, and fit timing remain intact. `checkpoint_selection.csv` is the causal audit:
it compares selected and terminal states from the same trajectory/count, avoiding false
attribution from nondeterministic CUDA trajectories. On COCO4 x seeds 0/1, N=640, 5k steps, this
policy selected earlier states in 7/8 runs and gained +0.7702 dB PSNR, +0.00892 MS-SSIM, and
+0.0076 LPIPS on average. At 500 steps it selected an earlier state only once, for a negligible
+0.0066 dB mean PSNR gain with small SSIM/LPIPS tradeoffs. It is therefore a long-horizon quality
option, not the pinned general default.

FIT-016 keeps the low-pass image strictly inside the differentiable pixel/SSIM objective. All
reported metrics, target hits, early stopping, checkpoint scores, and residual/tensor growth use
the original full target. `history.loss_target_full_weight` makes the changing objective explicit,
and stage offsets preserve one global schedule. Ambiguous combinations (geometry loss, color
solve, or count-changing/stop events before the full-target boundary) fail closed. When the
checkpoint control and low-pass arm are requested together, the harness writes
`lowpass_vs_checkpoint.csv` and `lowpass_vs_checkpoint_summary.csv`; these isolate the incremental
curriculum effect, while `default_dominance.csv` necessarily includes both checkpoint and
curriculum changes. This candidate approximates LIG's frequency ordering; it does not claim LIG's
separate residual fields or memory behavior.

FIT-013 adds opt-in geometry-consistency rows (`structsplat_best_gcr015`, `gcr030`, `gcr060`, and
intermittent variants). These apply target-gradient-weighted Sobel supervision on top of the pinned
default. They are experimental candidates: dense 0.015 improves quality/convergence in the current
COCO proxy and Kodak4 slice, but its larger-resolution timing cost blocks default promotion.

FIT-014 adds generation-cohort covariance-filter rows at `alpha={9*pi,18*pi,36*pi}`. They implement
the GaussianImage++ inverse-density variance rule faithfully, but all three lose PSNR, proxy
MS-SSIM, LPIPS, and AUC against the pinned default on the COCO4 640/500 proxy. Keep
`covariance_filter_mode=none`; the artifact is
`results/structsplat_generation_caf_proxy/index.html`.

```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  STRUCTSPLAT_INSTANT_GI=/path/to/Instant-GI/quard_image.py \
  python -m benchmarks.fair_density_control_compare --resume
```

`native_reference_compare.py` is the separate BENCH-005 path for real external repository code.
Each repository runs in an isolated subprocess because several ship incompatible packages named
`gsplat`. The initial adapter instantiates GaussianImage++'s upstream `SimpleTrainer2d` on one
arbitrary image, verifies that the compiled extension belongs to that checkout, records its hash
and commit, exports a float reconstruction, and centrally recomputes shared metrics. The current
`matched_axes` protocol aligns image, resolution, count cap, requested steps, and seed only; native
renderer/loss/optimizer/growth behavior remains native and must not be described as same
hyperparameters. Actual codec bpp stays blank until a native encoded stream is produced.
GaussianImage++ restores its upstream best-training-PSNR checkpoint before export; the native
artifact records the selected iteration and explicitly notes that StructSplat exports its terminal
field. The harness requires clean tracked upstream Python, fingerprints repo/gsplat trees and
Python sources, and keys resume on source/target bytes, decoded target pixels, harness/adapter/
metric sources, extension build, growth/timing/LPIPS settings, and the exact Python/Torch/CUDA/
NumPy/metric environment. Cached manifests are revalidated and central metrics recomputed before
reuse; the journal is compacted to the requested keys. Target, cell, and reconstruction paths use
canonical-path-and-content-qualified source IDs, so same-named inputs cannot overwrite evidence.
Pairing requires identical decoded target hashes and, for this matched-start protocol, identical
initial Gaussian counts.

```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
  python -m benchmarks.native_reference_compare \
    --images tests/test_images/COCO_train2014_000000000009.jpg \
    --gaussianimage-plus-repo /path/to/GaussianImage_plus \
    --max-sides 160 --budgets 640 --iters 500 --seeds 0 --lpips
```

`native_image_gs_compare.py` is the dedicated Image-GS native harness. Its v2 adapter runs the
clean official checkout at commit `03088368d42684fb54225c981cfd94b58cc0393a` in a separate Python
environment, requires `gsplat` to have been built from that checkout's bundled source, and pins
`fused-ssim` to commit `b4fd8324e81c48c9b2b9f62e1b9c6431fece6ab3`. Preflight and every cell
record and cross-check the repository tree/diff, package `direct_url.json` provenance, installed
Python-source hashes, compiled-extension hashes, Python/Torch/CUDA versions, GPU, and optional
`libstdc++` preload. `scripts/setup_native_image_gs_env.sh` now creates and verifies the official
Python 3.11.10, Torch 2.4.1, CUDA 12.4 environment. It constrains `mkl=2023.1.0` to avoid the
`iJIT_NotifyEvent` loader failure and `cuda-version=12.4` to prevent solver drift, then builds the
pinned fused-SSIM and bundled gsplat extensions. Exact environment exports and binary hashes live
under `results/native_envs/image_gs_official/`.

The four profiles are intentionally non-interchangeable:

- `matched_steps_fixed_n`: arbitrary requested horizon, float32, constant LR, and no progressive
  allocation; Image-GS starts at the full final N.
- `siggraph25`: paper-aligned 5k-step, constant-LR, 16-bit analytical-payload profile with native
  progressive allocation, applied at the requested benchmark resolution/count.
- `release_quickstart`: current 10k-step release behavior plus `--quantize`, progressive allocation,
  and the current LR-decay/early-stop schedule.
- `release_default_float`: the current bare-config 10k-step float32 behavior with progressive
  allocation and the current LR-decay/early-stop schedule.

The latter three are algorithm profiles; they are not native-authentic/full-resolution evidence
unless the requested image, resolution, count, and horizon also match the intended protocol. The
harness exports the terminal float reconstruction and centrally computes shared PSNR, SSIM,
small-image proxy MS-SSIM, and optional LPIPS. It retains upstream metrics separately. Native AUC
and target hits use Image-GS's sparse evaluation cadence and are diagnostic across implementations.
Likewise, `analytical_bpp` is only Image-GS's attribute-bit formula and omits a packed stream and
codec metadata; `actual_codec_bytes` and `actual_bpp` therefore remain blank.

Resume keys cover the target/source hashes, requested axes, adapter/metric/source revisions,
external repository and dependency builds, Python/Torch/CUDA/GPU state, timing settings, and LPIPS
state. A cached cell is revalidated against its manifest and reconstruction hash before reuse.
Paired analysis additionally requires identical run-recorded decoded-pixel hashes, preventing
same-name, stale-target, or different-resize rows from being joined. Progressive profiles also
require the recorded native and StructSplat start counts to match; the fixed-N profile deliberately
allows and reports its full-N versus half-N initialization mismatch.

```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
  python -m benchmarks.native_image_gs_compare \
    --images tests/test_images/COCO_train2014_000000000009.jpg \
    --image-gs-repo /path/to/image-gs \
    --image-gs-python results/native_envs/image_gs_official/bin/python \
    --profile matched_steps_fixed_n --max-sides 160 --budgets 640 \
    --iters 500 --seeds 0 1 --lpips --resume
```

The official-environment COCO4 x seeds 0/1, max-side 160, cap 640, 500-step fixed-N artifact is at
`results/native_image_gs_fixedn_500_official_two_seed/index.html`. Relative to the pinned
StructSplat default, Image-GS gains are -3.6639 dB PSNR (95% CI [-4.3839, -2.7583]), -0.01907
proxy MS-SSIM [-0.02937, -0.00812], -0.1773 LPIPS [-0.2592, -0.1099], and diagnostic AUC
-2.7060 [-3.2294, -1.9944], where positive always favors Image-GS. The familywise final-quality
test supports StructSplat on this bounded slice. It remains non-strict implementation evidence:
Image-GS starts at all 640 Gaussians, while StructSplat starts at half N and grows, and timing/AUC
accounting differs.

The official-environment `siggraph25` proxy at the same target pixels/cap and 5,000 requested
steps completed for COCO4 seed 0 at
`results/native_image_gs_siggraph25_official_seed0/index.html`. Against the terminal StructSplat
default, Image-GS gains +0.2201 dB PSNR, +0.01959 proxy MS-SSIM, and -0.0369 LPIPS. Against
`structsplat_best_checkpoint`, Image-GS gains -0.3601 dB PSNR, +0.01038 proxy MS-SSIM, and
-0.0566 LPIPS. Confidence intervals do not support a uniform winner; both comparisons are
tradeoffs. This remains a single-seed, small-image algorithm-profile result—not full-resolution
or rate-distortion evidence.

`native_gaussianimage_compare.py` and `native_runners/gaussianimage.py` execute the base ECCV
GaussianImage repository at commit `d53393bee7c9fbb24e3510614e3ff2c85b8fbbc1` with pinned gsplat
`bcca3ecae966a052e3bf8dd1ff9910cf7b8f851d`. The runner preserves fixed random count, native
Cholesky/RS parameterization, L2, Adan, the 20k-step LR schedule, and terminal selection. The
harness hashes clean source trees, the retained build wheel, loaded extension, adapter/metric
sources, input pixels, environment, and checkpoint; shared metrics come from exported float
pixels. Resume keys include the shared comparison-source revision; cached manifests are revalidated,
central metrics are recomputed, and stale journal rows are compacted away before evidence output.
`scripts/setup_native_gaussianimage_env.sh` provisions the isolated Python 3.10,
Torch 2.0.0+cu118 build and records exact dependency/linkage provenance.

```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
  python -m benchmarks.native_gaussianimage_compare \
    --images tests/test_images/COCO_train2014_000000000009.jpg \
    --gaussianimage-repo results/native_envs/gaussianimage_official/repo \
    --native-python results/native_envs/gaussianimage_official/env/bin/python \
    --profile matched_steps_fixed_n --max-sides 160 --budgets 640 \
    --iters 5000 --seeds 0 --lpips
```

The 500-step/two-seed artifact
`results/native_gaussianimage_matched_500_official_two_seed/` shows why horizons must be explicit:
GaussianImage is about 0.28 s faster than the terminal default but loses 13.75 dB PSNR, 0.2593
MS-SSIM, 0.5037 LPIPS, and 14.66 AUC because its native optimizer is designed for much longer
fits. At 5k/seed0 (`results/native_gaussianimage_matched_5000_official_seed0/`), GaussianImage is
about 6.4 s faster than the StructSplat checkpoint candidate and +0.01298 MS-SSIM, while
StructSplat is +0.1207 dB PSNR, +0.0253 LPIPS gain, and +1.53 AUC. This is a tradeoff, not a
dominance result; the published 50k/full-resolution and QAT/RD tracks remain open.

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
