# StructSplat

**Hierarchical, feature-aware, anisotropic blue-noise 2D Gaussian image representation.**

A single image is encoded as a set of oriented 2D Gaussians and rendered by a sorting-free,
normalized weighted-sum rasterizer. The repository's tested structural-prior candidate is:

- a **structure tensor** `J = G_ρ * (∇I ∇Iᵀ)` is the single operator for *density* (where to put
  Gaussians), *orientation* (how to elongate them), and *classification* (flat / edge / corner);
- Gaussians are placed by **anisotropic, density-adaptive blue noise** (Weighted Sample Elimination
  with a Mahalanobis metric) — packing across edges, spreading along them, no clumping, no grid;
- edges are tensor-aligned and density-aware; flanking remains available as a control, but the
  current evidence favors on-edge/quadtree WSE placement over flanking;
- an optional progressive WSE ordering improves audited uniform-set geometric prefixes without
  changing the terminal set; SSPL1 currently Morton-reorders the field, so this is not yet an
  embedded-codec/LOD claim.

The broad ingredients and several close combinations now exist in prior work, including
structure-guided allocation/orientation/precision and progressive Gaussian coding. The completed
actual-rate development gate found that tensor-metric WSE did not add enough value beyond the
strong gradient control; that compression claim is closed. Current research therefore treats the
repository as an interpretable causal substrate for new representation, ownership, renderer, and
codec hypotheses rather than as an established SOTA codec. `structsplat` is a **placeholder name**
— rename freely (see the `structsplat-docs-sync` skill).

The [September research portfolio](docs/research/2026-09-05-overnight-research-portfolio.md)
organizes the default-off HIER-033–036 gradient, caching and convergence experiments.
The [overnight findings and handoff](docs/research/2026-09-05-overnight-findings.md) record the
independently audited mixed/negative results, numerical-polish and rollback caveats, and an
unrun simple-fallback design (ARA C68–C72). No maintained pipeline or default is promoted.

The follow-up [code-driven portfolio](docs/research/2026-09-05-code-driven-portfolio.md)
selects guarded color refinement (FIT-050) and same-call quality-measurement reuse (PORT-007)
for bounded experiments. These active investigations do not authorize a default change.

> This is a PyTorch **research reference** with an opt-in exact CUDA extension for the same
> normalized/additive equations. The remaining production port is a tiled CUDA/Vulkan/RHI path
> (`tasks/PORT-001-cuda-rasterizer.md`, ADR-0011).

## Install
```bash
pip install -e .                 # torch, numpy, pillow, imageio
pip install -e ".[metrics]"      # optional: lpips, pytorch-msssim
pip install -e ".[gen]"          # optional: diffusers text-to-Gaussian generation
pip install -e ".[dev]"          # pytest, ruff
```

## The best pipeline: `scripts/convert.py` (ADR-0025/0028/0029/0030)

`scripts/convert.py` is the sole supported conversion CLI for the current best pipeline. It accepts
one image or a recursively scanned folder. Pass `--mask` for one alpha-matted/dome image,
`--mask-dir` for a parallel mask tree, or omit both for the full-frame arm. Both arms use
`structsplat.pipeline.RECIPE` and the same phase ordering, budgets, optimizer, proposal auction,
Pareto gate, and polish. The full-frame arm removes boundary initialization, containment,
boundary losses/metrics/proposals, and uses count-matched general proposals in the same closure
slot.

```bash
# one masked image
python scripts/convert.py frame.png runs/frame --mask frame_alpha.png --device cuda:0

# the same recipe plus the experimental final error-only fine-detail stage
python scripts/convert.py frame.png runs/frame_fine \
  --mask frame_alpha.png --device cuda:0 --fine-detail

# sparse deep-detail pursuit; mutually exclusive with --fine-detail
python scripts/convert.py frame.png runs/frame_pursuit \
  --mask frame_alpha.png --device cuda:0 --fine-detail-pursuit

# one full-frame image
python scripts/convert.py photo.png runs/photo --device cuda:0

# a folder and parallel mask tree
python scripts/convert.py ./rgb ./runs/converted_masked \
  --mask-dir ./mask --device cuda:0 --resume
```

Each image gets an editable `field.npz`, target, final reconstruction, fixed-scale error image,
accepted intermediate reconstructions/errors, `config.json`, `history.json`, and `result.json`.
The destination root also gets `manifest.json`, tidy `metrics.json`/`metrics.jsonl`/`metrics.csv`,
and a portable `index.html`. Existing complete cells require `--resume`; use `--overwrite`
explicitly to replace them.

`--fine-detail` is an experimental, default-off terminal capacity stage (FIT-031/ADR-0029). After
the ordinary schedule, it computes foreground per-pixel RGB MAE `e`, estimates effective residual
support as `ceil((sum e)^2 / sum(e^2))`, requests half that many small isotropic error-ranked
Gaussians, and optimizes under the same Pareto gate to a deterministic fixed point or a logged
4,000-step ceiling. The estimate is a reproducible allocation heuristic, not a promise of zero
error, and the resulting field is neither count- nor rate-matched to the default.

The FIT-031 exposed-image screen accepted 4,608 of 7,089 requested rows and improved its own
pre-tail foreground/boundary PSNR by `+0.522/+0.583 dB` while preserving the full gate (C58).
The clean default and tail runs were not count-, rate-, or CUDA-trajectory-matched, so this keeps
the option experimental and does not change the current recipe.

`--fine-detail-pursuit` is the separate masked-only ADR-0030 path. In 128-row waves it selects
5x5-NMS peaks of the current deep sigma-1.5 high-pass residual, appends 0.35-pixel ordinary
Gaussians, jointly solves only all pursuit-row colors under the exact normalized compositor, and
remeasures. Inherited rows remain bit-exact and every wave must pass the full protected gate. It
stops at the first safe `25%` high-pass and `20%` Laplacian reduction, or at a logged rejection,
site exhaustion, or 2,048-row ceiling.

On the exposed full `1200x1038` Janelle frame, it reached the target at 768 rows:
`25.93%/27.32%` high-pass/Laplacian reduction and `10.46%` relative LPIPS reduction. The same-base
FIT-031 control used 2,777 rows and improved global foreground PSNR more, but placed all rows near
the mask boundary and changed the deep-detail metrics by effectively zero (C60). Thus pursuit is
the measured fine-detail option, not a general/global-quality winner. It remains default-off and
is mutually exclusive with `--fine-detail`.

```python
from structsplat.pipeline import PipelineConfig, run_pipeline

result = run_pipeline(image, mask)  # mask=None selects the full-frame arm
field, metrics = result["field"], result["metrics"]
```

For programmatic experiments, changing `PipelineConfig.capacity` alone rescales the schedule:
phase targets derive from it (5/11 initial, 8/11 coverage, 10/11 detail). The CLI intentionally
runs the fixed current profile, whose defaults reproduce the 5,000/500/8,000/10,000-row Janelle
recipe.

**Evidence scope.** `PipelineConfig` defaults are the measured Janelle recipe: progressive WSE
ordering (C25), Pareto-safe checkpoints every 50 steps (C50), event color solve off (C50), no
specialized detail tail (C52/C59), dynamic storage (C51), global refinement, and a `0.75` px mask
margin (C56). All ten resolved FIT-023/024/025 Janelle arms used `0.75`; the `1.5` result belongs
to an older, different CORE-010 fit. No controlled margin comparison exists, so `0.75` is the
evidence-aligned recipe value, not a general quality claim. The evidence is one masked image, one
seed, one GPU. The full-frame arm is a mechanism extension with no independent benchmark screen
yet; BENCH-017 owns that screen.

To change the recipe after a new approach wins, edit `RECIPE` and the `PipelineConfig` defaults in
`src/structsplat/pipeline.py` together, bump the version, and cite the authorizing claim.

## Evaluation workflows (ADR-0027)

The benchmark, ablation, and stage-search scripts delegate to the same `run_pipeline` definition.
They are evaluation tools, not alternate conversion entrypoints.

### Benchmark the current pipeline

```bash
python scripts/benchmark.py ./images ./results/current_benchmark \
  --seeds 0 1 --lpips --device cuda:0 --resume
```

The report contains PSNR, SSIM, MS-SSIM, optional LPIPS, MSE/MAE/error-tail curves, PSNR AUC,
target-hit steps, phase/end-to-end timings, and target/intermediate/final/error images for every
image and seed.

Pinned official GaussianImage and Image-GS runs are optional. Their repositories and isolated
Python environments must already exist:

```bash
python scripts/benchmark.py ./images ./results/current_benchmark_native \
  --baselines gaussianimage image_gs --seeds 0 --lpips --device cuda:0 \
  --gaussianimage-repo ./results/native_envs/gaussianimage_official/repo \
  --gaussianimage-python ./results/native_envs/gaussianimage_official/env/bin/python \
  --image-gs-repo /path/to/image-gs \
  --image-gs-python ./results/native_envs/image_gs_official/bin/python
```

The comparison aligns decoded target pixels, Gaussian count, requested optimizer horizon, and
seed. Native renderer, optimizer, loss, growth, and timing semantics remain implementation-specific
and are labeled in their linked reports.

### Run the fixed ablation

```bash
python scripts/ablation.py ./images ./results/current_ablation \
  --seeds 0 1 --lpips --device cuda:0 --resume

python scripts/ablation.py ./images ./results/current_ablation \
  --arms full no_coverage_growth no_redistribution no_polish
```

The default matrix contains the full recipe plus no-bootstrap, no-coverage, no-detail, no-closure,
no-redistribution, no-polish, no-Pareto-checkpoint, and, for masked inputs,
no-boundary-specialization arms. It uses the benchmark report contract and accepts the same
optional native-baseline arguments.

### Search every variant of one stage

```bash
python scripts/stage_search.py ./images/kodim01.png ./results/search_detail \
  --stage detail --seeds 0 1 --device cuda:0 --resume
```

Stage search requires exactly one image and runs every registered variant of the selected stage
while freezing the rest of the current recipe. Stages are `initialization`, `storage`,
`checkpoint`, `bootstrap`, `coverage`, `detail`, `closure`, `redistribution`, `polish`,
`commit_gate`, and `hole_budget`. Use `--variants ...` for a subset and `--help` for the complete
interface.

`commit_gate` (BENCH-018) varies the transactional block — the unit of discarded work — across
every gated phase; `hole_budget` (FIT-028) varies the ADR-0026 interior coverage trade-off budget.
Both keep `current` as the shipped baseline and change nothing else. Every current-profile run card
reports commit-gate accounting: attempted versus accepted steps per phase, block counts, and the
schedule's own rejection-reason histogram.

Previous task-specific launchers live in `deprecated_scripts/`. They remain available for
historical evidence reproduction but are not supported entrypoints. The remaining top-level files
in `scripts/` are verification/maintenance tools and the four operational workflows above;
one-off new experiment drivers belong in `scripts/experiments/`.

## Quickstart
`structsplat fit` is the knob-level research command: every stage axis is a flag, and its defaults
are the conservative shipped defaults rather than the recipe above.

```bash
# fit one image with the measured high-budget PSNR winner
structsplat fit photo.png --strategy quadtree_wse --num-gaussians 20000 --iters 2000

# progressive (hierarchical) fit
structsplat fit photo.png --pyramid --num-gaussians 20000

# encode a folder in parallel worker processes, round-robin across GPUs; resumable via
# metrics.jsonl + existing .npz outputs (PORT-005 encode throughput)
structsplat batch-fit ./photos --num-gaussians 20000 --iters 2000 \
    --devices cuda:0,cuda:1 --workers 2 --outdir runs/batch
    
# watch the fit live in a browser (needs the optional igsv package; diagnostic, ADR-0018)
structsplat fit photo.png --live --live-every 25   # then open http://127.0.0.1:8890

# the core experiment: init strategy x budget sweep (writes results/summary.md)
structsplat ablation ./images --budgets 2000 5000 10000 20000 --iters 1500 --target-psnr 35

# full stage-combination screening, including factored refinement site/primitive/NMS axes
structsplat stage-search ./images --budgets 1024 2048 --iters 300 --outdir results/stage_search

# per-stage influence: one-factor-at-a-time deltas vs the baseline (writes influence.md with
# ΔPSNR / ΔMS-SSIM / ΔAUC / Δiters-to-target / Δseconds per stage option, ADR-0010)
structsplat stage-search ./images --mode influence --budgets 2048 --seeds 0 1 2 \
    --iters 500 --target-psnr 30 --outdir results/stage_influence

# long stage-search runs are resumable/shardable; influence HTML marks best paired-delta variants
structsplat stage-search ./images --mode influence --resume --max-new-cells 64 \
    --outdir results/stage_influence

# text-to-Gaussian MVP: sample raster -> fit -> latent SDS refine -> save .npz + PNGs
structsplat generate "flat red calendar app icon" --n 5000 --steps 200 --outdir runs/icon

# mask-contained fit for an alpha-masked object: Gaussians stay inside the mask, render is
# exactly zero outside (needs --support-fade). Mask can be a grayscale or RGBA-alpha image.
structsplat fit object.png --mask object_alpha.png --mask-contain --support-fade \
    --loss-weighting mask --num-gaussians 8000 --iters 2000
# soft alternative (no hard clamp): penalize out-of-mask coverage instead
structsplat fit object.png --mask object_alpha.png --mask-coverage-weight 1.0 --loss-weighting mask
```

## Image ↔ Gaussians2D conversion and diagnostics

StructSplat uses two related 2D-Gaussian artifacts in this workspace. A native
`GaussianField` `.npz` is editable full-precision fit state; it does not contain the canvas size or
renderer settings. A realtime-gs `gaussians2d/*.rtgsv` file is a self-contained calibrated view
with camera, fitted window, renderer semantics, source hashes, and optional exact alpha. The two
formats are not interchangeable, and the 168,000-byte cap applies to the complete `.rtgsv` file,
not to a native `.npz`.

CORE-016/ADR-0032 additionally provides a **default-off research packet** (`.sgdp`) that is not a
third supported conversion format. It stores a fully charged conventional appearance payload and a
separate sparse structural Field V2 measure, then exposes them to realtime-gs as a required
structural-field/query-backend pair. The task-local diagnostic is
`scripts/experiments/core016_codec_native_field.py`; the downstream follow-up is
`scripts/experiments/core016_multiview_downstream.py`. The paired adapter can keep realtime-gs
structural metadata on CPU while executing indexed structure and appearance queries on CUDA. An
exposed reduced-resolution 23-view matched-10k run supports downstream development utility at
lower complete teacher-input bytes than RTGSV, but its full lift/training resources are not better
and native review retains halos, blur, and floaters. Neither diagnostic establishes held-out/full-
resolution compression, artifact freedom, end-to-end speed, or a maintained default. See the
[research portfolio](docs/research/2026-08-06-codec-native-dual-plane-portfolio.md),
[ADR-0032](docs/adr/0032-codec-native-dual-plane-observation.md), and
[results audit](docs/research/2026-08-06-codec-native-dual-plane-results-audit.md).

CORE-017 adds a second default-off research composition to that adapter: sparse structure proposes
rays, exact packet alpha chooses the first maximally supported depth, codec appearance supplies
radiance, and an optional realtime-gs surface-cover pass changes only covariance/opacity. Its
exposed fixed-5k `frame_00009` factorial materially improves quality, silhouette localization,
early convergence, and lift work versus interior-consensus placement, but native review still finds
trailing smear/double silhouettes and blur. The scalar gate passes while the mandatory visual gate
fails; this is causal diagnostic evidence, not a supported pipeline or promotion.

CORE-018 tests a materially different default-off geometry composition: packet-derived DINOv2 and
local features define a source-excluded coarse/fine depth posterior with a dustbin, then reciprocal
candidate agreement must hold before a Gaussian is emitted. Its disjoint unmasked
`karate/frame_00060` killing test rejects the route. The no-reciprocal control starts 1.846 dB above
interior consensus but loses that advantage by step 500 and remains a visibly smeared volume; median
posterior entropy is 0.960 and median reciprocal support is zero. The complete arm cannot satisfy
its frozen 75% support floor and fails closed. The method remains a negative control only; lowering
the gate or integrating it into conversion is not authorized. See the
[results audit](docs/research/2026-08-06-core018-ray-posterior-results-audit.md).

CORE-019 tests the required coherent successor without changing the packet or maintained pipeline.
A pinned lazy VGGT predicts overlapping four-view depth fields from packet appearance; group Sim(3)
alignment supplies scale to known camera rays, and projective support, hard feature anchors,
compatibility-aware WSE, bounded contraction, and depth-normal surfels compile an exact 10,000-row
initializer. On exposed `karate/frame_00005`, the full compiler changes the raw-known-ray tradeoff
rather than winning it: v5 gains +0.0102 MS-SSIM, -0.0206 LPIPS, and 904 fewer
final rows, but loses 0.3075 dB PSNR and worsens gradient/p99 error. It starts 0.969 dB below interior
consensus, misses every fixed-prefix quality gate, never reaches the interior terminal PSNR, and
ends 0.928 dB below it. Replays also flip the terminal full-vs-raw PSNR ordering after density
events. Native reporting renders show broad gray sheets, radial streaks, floaters, black holes, and
erased detail. The method is therefore a default-off negative control, not an artifact-free or
supported route; the separate 5.03 GB public checkpoint is CC-BY-NC-4.0 and is not hidden in the
per-scene byte ratio. See the
[results audit](docs/research/2026-08-07-core019-coherent-depth-results-audit.md).

### One image → native 2D Gaussian field → image

`image-to-gaussians2d` is an explicit alias for `fit`. This example writes
`runs/photo/photo_quadtree_wse.npz` (the Gaussian field) and
`runs/photo/photo_quadtree_wse.png` (the terminal reconstruction):

```bash
structsplat image-to-gaussians2d photo.png \
  --strategy quadtree_wse --num-gaussians 20000 --iters 2000 \
  --outdir runs/photo
```

`gaussians2d-to-image` is an alias for `render`. Supplying the original image provides the canvas
size and enables a fixed-scale absolute-error visualization, the raw signed error, numerical
metrics, and a fitted-Gaussian overlay:

```bash
structsplat gaussians2d-to-image runs/photo/photo_quadtree_wse.npz \
  --reference photo.png \
  --out runs/photo/reconstruction.png \
  --error-out runs/photo/absolute_error_x4.png \
  --raw-error-out runs/photo/signed_error.npy \
  --metrics-out runs/photo/reconstruction_metrics.json \
  --gaussians-out runs/photo/fitted_gaussian_ellipses.png
```

The heatmap displays `4 × mean(abs(reconstruction - reference), RGB)` on a fixed `[0,1]` scale;
it is for visual comparison and is not normalized independently per image. `signed_error.npy` is
the float32 `(H,W,3)` array `clamped_reconstruction - reference`. The JSON records display-referred
MSE, MAE, maximum absolute error, PSNR, SSIM, and MS-SSIM before PNG encoding. Without an original,
give the native NPZ canvas explicitly:

```bash
structsplat render runs/photo/photo_quadtree_wse.npz \
  --height 1080 --width 1920 --out runs/photo/reconstruction.png
```

The same render command accepts a self-describing `SSPL1` stream and then uses the dimensions and
renderer settings stored in its header. For a native NPZ, pass the same `--renderer`,
`--aa-dilation`, `--sigma-cutoff`, and `--support-fade` settings used during fitting whenever they
differ from the defaults.

### Byte-budgeted pooled fitting (FIT-021 / ADR-0020, opt-in)

`--triage-every` switches the fitter to the pooled row lifecycle: tensor capacity is fixed for
the whole fit, "pruning" only parks rows off-image as teleport/spawn donors (exactly zero render
contribution via the CORE-003 support clip), and one triage event per cadence runs
responsibility-gated park → envelope merge → split → spawn in place, with no optimizer rebuild.
`--target-file-kb 168` derives the capacity so the encoded SSPL1 — including the in-container
alpha stream for masked inputs — stays within 168,000 decimal bytes (the same convention as the
`.rtgsv` cap), and writes the budgeted `.sspl` next to the NPZ. It replaces
`--split-every`/`--relocate-every`/`--prune-every`/`--adaptive-count`:

```bash
structsplat image-to-gaussians2d photo.png --mask photo_mask.png \
  --strategy quadtree_wse --num-gaussians 5000 --iters 4000 \
  --mask-contain --support-fade --loss-weighting mask \
  --triage-every 100 --target-file-kb 168 \
  --outdir runs/photo_pooled
```

### Deprecated Janelle-specific diagnostic and transport runners

The supported masked and unmasked workflow is `scripts/convert.py` above. The following
Janelle-specific runners are retained under `deprecated_scripts/` only to reproduce the
source-bound FIT-021--025 evidence and the realtime-gs transport bridge.

The calibrated Janelle conversion is handled by the resumable mask-contained bridge. Run it from
this checkout with the CUDA realtime-gs environment:

```bash
cd /home/alex/Documents/structsplat
PY=/home/alex/Documents/realtime-gs/.venv-cuda/bin/python
"$PY" -m pip install -e . -e /home/alex/Documents/realtime-gs
"$PY" deprecated_scripts/fit_janelle_mask_contained.py convert \
  --dataset-root /home/alex/Dropbox/Work/Janelle \
  --realtime-root /home/alex/Documents/realtime-gs \
  --device cuda:0
```

Inspect every parameter, its resolved default, whether it is active in residual or boundary
growth, and the current cleanup/export limitations before starting a run:

```bash
"$PY" deprecated_scripts/fit_janelle_mask_contained.py convert --help
```

The help output is available without importing PyTorch, CUDA, or realtime-gs. Options are grouped
by input, capacity/growth lifecycle, optimization/stopping, initialization/rendering, mask
containment, and diagnostics. It explicitly distinguishes fit-time births from the independent
activity-ranked byte-cap export; this wrapper currently exposes no fit-time pruning, relocation,
adaptive counting, or merge operator.

For each masked frame it writes `frame_*/gaussians2d/Cxxxx.rtgsv`, enforcing at most 168,000
decimal bytes for the complete view. It also writes `fitting/Cxxxx.json`,
`fitting/Cxxxx_history.csv`, `fitting_summary.csv`, and `run_config.json`. Those sidecars contain
the benchmark fitting curve, target-hit iterations, selected-versus-terminal checkpoints, final
quality, serialization backoff, per-stage wall times, and total fitting time. They also record
staged count plateaus when `--start-gaussians` is below `--candidate-gaussians`; with the default
direct full-candidate fit, the lower-count result is explicitly labeled as a post-fit
activity-ranked saturation proxy rather than an independently converged smaller fit. Verified
outputs are skipped on rerun; partial or provenance-mismatched outputs stop instead of being
overwritten silently.

For a single native-resolution `frame_00008/C0001` quality run that exercises the complete
refinement lifecycle, use the dedicated diagnostic runner instead of the capped multi-view bridge:

```bash
/home/alex/miniconda3/bin/python deprecated_scripts/fit_janelle_complete_refinement.py \
  --capture-root /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric \
  --frame frame_00008 --view-id C0001 --device cuda:0 \
  --out runs/janelle_C0001_complete_refinement_20260722
```

It initializes exactly 5,000 rows (4,500 feature-aware quadtree-WSE plus 500 explicit
boundary-residual rows), then stages high-error tensor births, tangent boundary births,
moment-preserving splits, fit-time low-activity pruning, and residual relocation. Between growth
stages, mutually-nearest redundant pairs are rewritten in a batch: one row moves to the exact pair
midpoint and receives a 5%-enlarged covariance envelope covering both translated input covariances,
while the other row is retained and teleported to a spaced high-error site with local feature-aware
covariance. This merge/teleport is count-neutral. A pair is batch-rejected when its mask-containment
cap cannot preserve the covariance envelope. Final low-rate, boundary-weighted per-pixel settle
rounds continue until both foreground and boundary PSNR plateau or the configured round limit is
reached; a candidate must also reduce foreground and boundary MAE before it replaces the incoming
field.

The primary result is the uncapped editable `C0001_full_refined.npz`. The runner also derives a
168,000-byte `gaussians2d/C0001.rtgsv`, but records it separately because activity-ranked export
backoff is not boundary-error-aware. `index.html`, `progress.csv`, `progress.json`, native final
images, fixed-scale error maps, and per-stage NPZ/JSON checkpoints show how quality and error evolve
through the run.

For the production phase ordering with monotone accepted checkpoints, use the transactional
safe-commit runner:

```bash
PYTHONPATH=src /home/alex/miniconda3/bin/python \
  deprecated_scripts/fit_janelle_safe_commit_schedule.py \
  --capture-root /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric \
  --frame frame_00008 --view-id C0001 --device cuda:0 \
  --out runs/janelle_C0001_safe_commit_schedule_20260723
```

It uses the same 4,500 WSE + 500 boundary initialization, then runs fixed-topology bootstrap,
coverage-first growth, detail birth/moment split, boundary closure, count-neutral redistribution,
and low-rate polish. Optimizer blocks and topology events operate on detached trial fields. A
trial is committed only when full-resolution foreground MSE, boundary MSE, CVaR99 error, reachable
interior holes, reachable boundary holes, and outside-mask coverage are all nonworse and at least
one improves. Rejected trials discard both their parameter changes and optimizer moments.
The one-sided raw-coverage floor is evaluated intermittently (production default: every eight
optimizer steps, with compensating weight scaling); exact hole metrics are still checked at every
commit and topology event.

Birth covariance is estimated from both the local target structure tensor and the spatial support
of the current residual. At full capacity, merge→rebirth keeps one enlarged midpoint/envelope
Gaussian and directly reuses its absorbed partner at a high-error site; prune→rebirth and
donor-funded moment splits are the other count-neutral candidates. The candidates compete through
full-render trials instead of consuming one shared free list in a fixed operator order. Rejected
large batches are halved automatically.

The safe schedule can use the same topology policy with fixed-capacity storage:

```bash
PYTHONPATH=src /home/alex/miniconda3/bin/python \
  deprecated_scripts/fit_janelle_safe_commit_schedule.py \
  --storage-policy fixed_capacity --capacity 11000 \
  --pareto-safe-checkpoints --pareto-checkpoint-every 50 \
  --capture-root /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric \
  --frame frame_00008 --view-id C0001 --device cuda:0 \
  --out runs/janelle_C0001_safe_commit_fixed_capacity
```

This establishes capacity-shaped field and Adam state before growth, renders/optimizes only a
contiguous active prefix, writes accepted births into reserved rows without append/pad resizing,
and compacts once for export. Transactional proposals and Pareto checkpoints still clone
capacity-shaped scratch state; this is not a persistent no-allocation arena. It does not enable
FIT-021 triage or change the proposal auction, recovery fits, commit gate, or Pareto rollback.
Adam update kernels also receive the active shape, avoiding capacity-dependent arithmetic and work
on inactive rows. `dynamic` remains the default for historical reproduction.
On the source-bound Janelle check, fixed capacity matched the two-run dynamic quality span to
reported precision (27.063 dB foreground, 11.400 dB boundary; exact foreground was 0.0000031 dB
below the lower dynamic endpoint and every other protected metric was within or favorable to the
span), used 2,130 MiB peak versus 2,142 MiB for both dynamic controls, and was effectively
runtime-neutral after accounting for different safe-auction paths. The CUDA renderer is last-bit
nondeterministic even for repeated renders of one field, so event-sequence identity is not a valid
parity requirement; multi-image evidence is still required before a default change or broad speed
claim.

FIT-025 separates fixed physical capacity from staged activation. `--capacity` allocates the
physical field and Adam tensors, while `--base-active-limit` caps ordinary coverage/boundary/
redistribution growth. An optional suffix can be reserved for a post-color-solve detail tail:

```bash
PYTHONPATH=src /home/alex/miniconda3/bin/python \
  deprecated_scripts/fit_janelle_safe_commit_schedule.py \
  --storage-policy fixed_capacity --capacity 12024 \
  --base-active-limit 11000 \
  --detail-tail-rows 512 --detail-tail-batch 128 \
  --detail-tail-min-gain-per-row 0 \
  --pareto-safe-checkpoints --pareto-checkpoint-every 50 \
  --capture-root /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric \
  --frame frame_00008 --view-id C0001 --device cuda:0 \
  --out runs/janelle_C0001_safe_commit_detail_tail
```

Tail proposals use only already-covered deep-interior persistent high-frequency residuals, auction
detail births against moment-preserving splits, retain the complete Pareto gate, and stop when no
safe proposal clears the optional gain-per-row floor. Defaults preserve the old behavior:
`base_active_limit=None` resolves to physical capacity and the tail is disabled.

The matched fixed-storage Janelle screen favors ordinary activation, not the specialized tail.
With the same 12,024 physical rows, an 11,512 generic active ceiling reaches
`27.2193/11.5825 dB` foreground/boundary in 406.3 s total; the 11,000 baseline reaches
`27.0653/11.4124 dB` in 416.0 s, and the equal-count adaptive tail reaches
`27.1069/11.4325 dB` in 450.8 s. Generic +512 is better on every nontrivial protected metric.
All three arms are pooled, so this timing ranks activation policies rather than fixed versus
dynamic storage. Keep the tail opt-in and do not select a nonzero threshold from this exposed
single image. The audited comparison is under
`runs/janelle_C0001_detail_tail_ablation_20260724/`.

`index.html` contains only the accepted reconstruction/error sequence and lists rejected trials
separately. `schedule_history.json` is the complete transition audit; the editable result is
`C0001_safe_commit_full.npz`. The optional `.rtgsv` remains separately labeled because its legacy
activity-ranked byte-cap backoff is not yet boundary-error-aware; pass `--no-archive` when only the
authoritative full field is wanted.

The global policy remains the reproducible baseline. An experimental late-local policy keeps
bootstrap, coverage, and detail fitting global, then freezes unaffected rows during boundary,
redistribution, and polish. Residual-owning rows and a batched spatial neighbourhood are selected
for standalone local fits; topology recovery has an independent neighbour count. Boundary closure
can continue with count-neutral boundary merge/prune→rebirth candidates after the capacity is
full. The p99 guard is optional because it changes the Pareto gate rather than only the optimizer:

```bash
PYTHONPATH=src /home/alex/miniconda3/bin/python \
  deprecated_scripts/fit_janelle_safe_commit_schedule.py \
  --capture-root /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric \
  --frame frame_00008 --view-id C0001 --device cuda:0 \
  --out runs/janelle_C0001_safe_commit_hybrid_boundary_20260723 \
  --refinement-policy local_neighborhood --local-start-phase boundary \
  --local-seed-count 384 --local-neighbor-count 8 \
  --topology-neighbor-count 0 --boundary-recycle-at-capacity
```

Use `deprecated_scripts/compare_janelle_safe_schedule_variants.py` with repeated
`--run LABEL=PATH` arguments to create one source-backed comparison page. CUDA atomic accumulation
can change close residual rankings and therefore later topology decisions; a single sequential
pair is a mechanism test, not a deterministic or statistically replicated superiority claim.

FIT-023 tested state-matched block checkpoints and post-topology color solves as a source-bound
2×2 factorial on this one Janelle image. The development winner is the global schedule with
Pareto-safe checkpoints every 50 steps:

```bash
PYTHONPATH=src /home/alex/miniconda3/bin/python \
  deprecated_scripts/fit_janelle_safe_commit_schedule.py \
  --capture-root /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric \
  --realtime-root /home/alex/Documents/realtime-gs \
  --frame frame_00008 --view-id C0001 --device cuda:0 \
  --pareto-safe-checkpoints --pareto-checkpoint-every 50 --no-archive \
  --out runs/janelle_C0001_safe_commit_pareto_checkpoint
```

Against the clean global control it improved foreground/boundary PSNR by +0.502/+0.519 dB,
CVaR99/p99 MSE by 11.00%/19.40%, and relative interior/boundary undercoverage by 43.57%/10.10%,
at +9.61% total time. Four earlier checkpoints were actually committed with matching Adam
moments. Event color solve alone was worse on every recorded quality/coverage metric and 7.8%
slower end to end; combined was 29.3% slower than checkpoint-only and traded slightly better
CVaR/interior coverage for worse foreground, boundary, p99, and boundary coverage. Keep event color
solve off in the recommended schedule. This is single-image/single-seed development evidence, so
the library defaults remain unchanged. The best arm still has 1.436% interior and 28.491% boundary
undercoverage—fixed-point convergence did not meet the configured 0.1%/1% coverage targets.
The full comparison, native images, histories, cold-reload audit, and report are under
`runs/janelle_C0001_transactional_candidates_factorial_20260723/`.

To decode all `.rtgsv` fields back to images and compare them with the exact calibrated source
pipeline, run the realtime-gs gallery utility. Point both roots at the live Janelle capture while
the `rgb/` and `mask/` folders are still present:

```bash
cd /home/alex/Documents/realtime-gs
.venv-cuda/bin/python scripts/render_compact_structsplat_gallery.py \
  --compact-root /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric \
  --source-root /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric \
  --structsplat-root /home/alex/Documents/structsplat \
  --out /home/alex/Documents/structsplat/runs/janelle_2d_reconstructions
```

The output includes native-resolution original crops and reconstructions, foreground previews,
fixed-scale absolute-error heatmaps, per-view PSNR/MAE and render timing, sampled archive-query
versus renderer parity, `manifest.json`, and an `index.html` gallery. Use the `karate` directory as
both roots in a second invocation for that capture. Error comparison requires the original RGB;
the compact field itself does not store source pixels.

### Structure tensor and other source/representation features

Yes—the existing deterministic diagnostics visualize the structure tensor directly over the
original image and expose the other features used by initialization and normalized rendering:

```bash
cd /home/alex/Documents/structsplat
python deprecated_scripts/render_paper_figures.py photo.png \
  --outdir runs/photo/features --max-side 0 \
  --strategy quadtree_wse --num-gaussians 384 --seed 0
```

Useful outputs are `tensor_tangents.png` (cyan edge tangents and orange normals over the image),
`tensor_energy.png`, `tensor_coherence.png`, `tensor_labels.png` (flat/edge/corner),
`sampling_density.png`, `sampling_sites.png`, `gaussian_ellipses.png`,
`initial_abs_error.png`, `coverage_denominator.png`, `effective_contributors.png`,
`responsibility_entropy.png`, and `dominant_owner.png`. `diagnostics.npz` preserves the raw maps;
`method_diagnostics_montage.png` and `manifest.json` provide a labeled overview and provenance.

These tensor/density panels are source-derived initialization diagnostics: they are recomputed from
the supplied image and are not transmitted in `.npz`, `SSPL1`, or `.rtgsv`. They therefore cannot
be recovered from Gaussians alone. The fitted one-sigma Gaussian overlay from
`gaussians2d-to-image --gaussians-out` is representation-derived and can be generated from a field.
See `docs/publication_figures.md` for the exact panel semantics and claim limits.

## Live viewer in the sibling-repository workspace

`structsplat fit --live` uses the optional `igsv` server from
`../interactive-gs-viewer` and serves the built WebGPU client at
`http://127.0.0.1:8890/`. Install both editable packages into the Python
environment that has CUDA PyTorch, then build the browser client once:

```bash
# Run from the StructSplat checkout. This workspace uses realtime-gs's CUDA venv.
PY=../realtime-gs/.venv/bin/python
"$PY" -m pip install -e . -e ../interactive-gs-viewer/server
(
  cd ../interactive-gs-viewer/web
  npm ci
  npm run build
)
```

This is the tested quick live fit for the first image (`C0001`) of
`datasets/2025_03_07_stage_with_fabric/frame_00008`. It uses the current
high-budget PSNR initialization winner, `quadtree_wse`, and the soft boundary
penalty. ImageMagick's `convert` makes a small diagnostic input so the portable
reference renderer updates interactively:

```bash
RUN=runs/live_viewer_frame_00008_C0001
mkdir -p "$RUN/input" "$RUN/output"
convert ../datasets/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg \
  -resize '256x256>' "$RUN/input/C0001.jpg"
convert ../datasets/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png \
  -filter point -resize '256x256>' -threshold 50% "$RUN/input/mask_C0001.png"

../realtime-gs/.venv/bin/structsplat fit "$RUN/input/C0001.jpg" \
  --mask "$RUN/input/mask_C0001.png" \
  --strategy quadtree_wse --num-gaussians 640 --iters 300 \
  --renderer normalized --device cuda --chunk 128 \
  --wse-progressive-order --init-scale-mult 0.35 \
  --loss-weighting mask --mask-coverage-weight 1.0 \
  --ssim-weight 0.3 --checkpoint-policy best_psnr_final_count \
  --live --live-every 10 --live-port 8890 --outdir "$RUN/output"
```

Open `http://127.0.0.1:8890/` while the command is running. After fitting, the
process deliberately keeps the final field available; press `Ctrl-C` to stop
the server. The `256`-pixel input, 640-Gaussian budget, and 300 iterations are a
viewer smoke recipe, not a quality setting; the result is intentionally sparse.
For a longer fit, point the command at the original image and mask and raise the
budget/iterations.

On this Ubuntu/NVIDIA workspace, start a hardware-backed Chrome process with:

```bash
google-chrome \
  --user-data-dir=/tmp/igsv-chrome-webgpu \
  --enable-unsafe-webgpu \
  --enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE \
  --use-angle=vulkan \
  http://127.0.0.1:8890/
```

For Firefox, open `about:config`, set `dom.webgpu.enabled` to `true`, and fully
restart the browser. Browser version alone is insufficient on this Linux
installation; Firefox 153 otherwise leaves WebGPU disabled. The blocklist
override is not needed on the tested machine.

`--mask-coverage-weight 1.0` is a soft penalty on Gaussian weight outside the
mask; it does not force the displayed RGB to zero there. Add
`--mask-contain --support-fade` when exact zero outside the mask is required.
The command above uses `--renderer normalized` because it exercises the
canonical equations without building the optional CUDA extension. If a
`cuda`/`cuda_tiled` build fails, that is a renderer-toolchain problem rather
than an `igsv` connection failure; use the reference command to test the viewer
independently.

Strategies: `random`, `grid`, `iso_blue_noise`, `aniso_onedge`, `aniso_flanking`.
Additional quadtree strategies: `quadtree_aggregate`, `quadtree_hybrid`, `quadtree_wse`.
`local_slic_sobel_control` is a benchmark-only, explicitly local transplant for BENCH-007; its
frozen SLIC assumptions are not presented as upstream Structure-Guided Allocation code.
`feedforward` is the experimental FF-001 predictor warm-start (saved-field or tensor-prior fallback).
Samplers: `wse` (blue noise), `floyd_steinberg` (density-map error diffusion),
`dart_throwing` (Poisson disk), `halton`, `cvt`, `farthest_point`, `density_random`,
`jittered_grid`.
Pure-WSE layouts can add `--wse-progressive-order` to permute the identical terminal Gaussian set
into Yuksel-style nested prefixes. It is opt-in because saved row order and GPU reduction order are
part of experimental provenance; the current codec still Morton-sorts the full field. With a
background layer, frozen background rows stay first and only the detail suffix has WSE ordering.
Renderers: `normalized`, `additive`, `cuda`, `cuda_additive`, `cuda_tiled`,
`cuda_tiled_additive`, `gsplat`. `cuda`/`cuda_additive` and their tiled variants are exact
StructSplat semantics; `gsplat` is a GaussianImage++-style alpha/sum comparator.
Scale caps: `none`, `hard`, `feature` (ADR-0012).
Mask-contained fitting (`--mask`, CORE-010/ADR-0017): `--mask-contain` projects means into the mask
and caps effective scales from the signed distance so the sigma_cutoff support stays inside;
`--support-fade` makes the render exactly zero outside; `--mask-coverage-weight` is a soft
out-of-mask penalty; `--loss-weighting mask` drops out-of-mask pixels from the loss.
Boundary coverage (CORE-011/ADR-0019, all opt-in): `--mask-cap-mode anisotropic` certifies longer
along-tangent caps near the boundary with station-ball SDF probes (recertified every
`--mask-cap-refresh-every` iters), so edge Gaussians elongate instead of tiling;
`--mask-undercoverage-weight` is a hinge on uncovered boundary-band pixels (band/tau knobs);
`--mask-boundary-add-every/-count` spawns tangent-aligned Gaussians at boundary residual peaks
(`-band`/`-spacing` knobs). The CLI reports boundary-band (<=2 px) PSNR next to the out-of-mask
energy. The outermost ~`--mask-margin` px are unreachable by construction (containment reach is
`SDF - margin`); shrink the margin toward its ~0.71 px floor when the dead band matters.

## Publication method figures

Generate deterministic structure-tensor, tensor-metric sampling, initialized Gaussian, and
normalized-responsibility panels from a real image:

```bash
python deprecated_scripts/render_paper_figures.py tests/test_images/COCO_train2014_000000000030.jpg \
  --outdir results/paper_method_figure --max-side 256 --num-gaussians 384 --seed 0
```

The bundle includes a vector encoder/decoder overview, individual PNGs, raw NPZ maps, resolved
config, hashes/provenance, and a labeled montage. It is initialization-only explanatory output, not
optimized or comparative evidence. See `docs/publication_figures.md` for panel semantics and the
completed negative BENCH-007 Stage-1 F5--F9 bundle status.

## Agentic workflow

This repository uses one checked workflow across Claude Code, Codex, and other agent harnesses
(ADR-0031, C61). Start with `CLAUDE.md` and the generated `tasks/SESSION-BRIEF.md`; confirm work
against `tasks/INDEX.md`, which remains the sole outcome authority. The full lifecycle, including
Driver/Reviewer turns, handoffs, reviewed revisions, provisional self-review, and report
integrity, is in `docs/agent_workflow.md`.

- **`CLAUDE.md`** — project guide + a skill-aware routing table. **`AGENTS.md`** is a thin
  cross-harness adapter.
- **`.claude/skills/`** — eight canonical project skills: `structsplat-core`, `structsplat-task-workflow`, `structsplat-review`,
  `structsplat-method`, `structsplat-benchmark`, `structsplat-docs-sync`, `structsplat-research-ideation`,
  `structsplat-results-audit`. Every name is `structsplat-`prefixed so it cannot collide with a
  sibling repository's skill when more than one repo is open in an agent session. They're
  auto-discovered inside this repo; run `scripts/install_skills.sh` to symlink them into
  `~/.claude/skills` for global use (it refuses to install an unprefixed skill). The
  repo-prefixed entries under `.agents/skills/` are relative discovery
  symlinks to these same skill trees for Codex/Agent Skills, not duplicates. The ideation skill is
  a first-party, MIT-licensed adaptation by Alexander Dieckmann of
  `transformational-research-skill-kit` v1.0.0; the results-audit skill is its referee-side
  companion for evidence that already exists.
- **`tasks/`** — work items (`AREA-NNN-slug.md`) tracked in `tasks/INDEX.md`, with
  `tasks/TEMPLATE.md`, exact lifecycle schemas in `tasks/README.md`, and a generated
  `tasks/SESSION-BRIEF.md`. Say *"work on INIT-003"* and the `structsplat-task-workflow` skill
  drives the lifecycle.
- **`docs/adr/`** — architecture decisions the code references by number.
- **`docs/additive_field_v2.md`** — proposed, non-default Observation Field V2 architecture and
  evidence-gated task graph; current behavior remains in `docs/architecture.md`.
- **`ara/`** — the claim and evidence ledger. `ara/logic/claims.md` is where a number becomes a
  claim you may repeat; `scripts/check_ara.py` enforces its structure. See the "Evidence and
  claims" section of `CLAUDE.md`.
- **`scripts/`** — durable tooling and the structural gates `verify.sh` runs: `docs_sync.py`,
  `check_ara.py`, `check_task_policy.py`, `check_script_layout.py`, and
  `check_agent_workflow.py`. `generate_session_brief.py` derives startup context, while
  `check_report_bundle.py RESULTS_DIR` validates maintained portable reports before evidence
  handoff. One-off experiment drivers go in `scripts/experiments/`.
- **`docs/prompts/real-research.md`** — reusable evidence-first prompt for prior-art audit,
  preregistration, execution, negative-result handling, and per-axis conclusions.

Typical loop: `structsplat-core` → `structsplat-task-workflow` → `structsplat-method` (if adding a
component) → `structsplat-review` → `structsplat-docs-sync`; results-bearing work inserts
`structsplat-benchmark` → `structsplat-results-audit` before review.
Research discovery starts with `structsplat-core` → `structsplat-research-ideation`; selected candidates then
enter the normal task/method/benchmark loop.

## Layout
```
src/structsplat/   structure_tensor, density, sampling (NumPy) · gaussians, render, metrics,
                   init, fit, pyramid, codec, visualize, cli (torch)
tests/             pytest (NumPy tests run anywhere; torch tests skip without torch)
benchmarks/        ablation.py (ABL-001), stage_search.py (ABL-002), rate_distortion.py
                   (COMP-001), coco_fit_compare.py, cross_repo_matrix_compare.py,
                   optimization_followup.py, quadtree_init_compare.py, fitness hooks
docs/              adr/ · architecture.md · additive_field_v2.md · theory.md
tasks/             INDEX.md + task files
```

## The question this repo was built to answer — and the measured answer
The optimizer discovers anisotropy on its own, so flanking/tensor init mainly buys **convergence
speed** and **low-budget quality**. Hypothesis (ABL-001): `aniso_flanking ≥ aniso_onedge >
iso_blue_noise > grid > random` at low budgets, with the gap shrinking as the budget grows. If
flanking never wins, the honest move is to prefer the simpler strategy — the benchmark is designed
to tell you either way.

**Measured answer (2026-07-04..07): the flanking half of the hypothesis is dead; the
structured-placement half stands.** ABL-006 completed the decision-grade successive-halving
confirmation on Kodak-24 + COCO4 at max-side 768, 1500 iterations, exact CUDA, and 3-seed finalist
confirmation (`ara/evidence/abl006-complete-2026-07-07/`). Final PSNR winners are budget-specific:
`aniso_onedge` has the higher mean at 2000 Gaussians, but its paired PSNR lead over
`quadtree_wse` is not significant; `quadtree_wse` is the clear 5000-Gaussian PSNR winner
(+0.0930 dB, 95% CI [+0.0168, +0.1700]); and `quadtree_wse` has a small non-significant PSNR lead
at 10000 (+0.0357 dB, 95% CI [-0.0041, +0.0778]) while `aniso_onedge` has higher MS-SSIM.

Operational status: prefer `quadtree_wse` for high-budget PSNR work and keep `aniso_onedge` as the
low-budget/MS-SSIM alternative. `aniso_flanking`, `quadtree_hybrid`, `iso_blue_noise`, and
Floyd-Steinberg were eliminated at stage 1 by the frozen CI rule. ADR-0013 updates the shipped init
default to `quadtree_wse`; flanking stays available as an explicit control arm. The cross-repo
caveat stands: these are matched policy analogues inside StructSplat's harness, not native external
pipelines. BENCH-005 now has isolated, provenance-checked native GaussianImage++, Image-GS, and
GaussianImage runners. The official-environment Image-GS fixed-N 500-step slice supports
StructSplat on final PSNR/proxy-MS-SSIM, but differing initialization and timing semantics prevent
a strict implementation-dominance claim. At 5k steps, official Image-GS remains a tradeoff: versus
the full-count-checkpoint StructSplat candidate it has higher proxy MS-SSIM, while StructSplat has
higher PSNR and substantially better LPIPS. Native GaussianImage is much faster: at 500 steps it
has not converged, while at 5k it is roughly PSNR-competitive, higher in proxy MS-SSIM, lower in
AUC, and worse in LPIPS than the checkpoint candidate. Full-resolution, multi-budget/time-envelope,
native codec/RD, and learned Instant-GI tracks remain open.

**Current actual-rate verdict (2026-07-14).** BENCH-007 completed its preregistered eight-image
DIV2K development killing pilot with 288/288 independent fits and 1,152/1,152 validated complete
SSPL1 candidates. Against the strongest direct control (`local_gradient_control`), tensor-WSE
gained `+0.3457 dB` at 0.5 bpp but only `+0.0089 dB` at 1.0 bpp; mean BD-rate was `-4.5417%`
rather than the required `-10%`, fit-plus-search time was `1.4752x`, and texture MSE regressed
`7.2883%` beyond the 5% guard. The frozen gate failed. Stage 2 was not authorized or run, and the
exact tensor-WSE compression claim is closed without post-hoc tuning. See
`ara/evidence/bench007-stage1-killing-pilot-2026-07-14/run.md`.

The actual-rate harness, direct controls, cold-stream validation, conventional context, and F5--F9
figures are reusable research infrastructure, but Stage 1 is not held-out evidence and does not
support a compression-SOTA claim. Further method work needs a materially new question and disjoint
development screen; untouched DIV2K validation must not be consumed as a rescue set. The older
`storage_budget_168k_external_present` lane remains only a high-rate optimizer/policy diagnostic.

**Responsibility-transfer verdict (2026-07-15).** FIT-018 transplanted SAD's known
responsibility error-density split score into the normalized Gaussian renderer. Its frozen COCO4 x
two-seed shared-start guard rejected the donor `alpha=0.7`: `-0.0198 dB` versus support after 20
recovery steps, only 4/8 positive pairs, and `-0.0411 dB` after 100. Counts, finite values, the
post-100 floor, and the `+1.8%` timing overhead passed, but both post-20 quality gates failed. Keep
the implementation as an opt-in causal control; do not tune this exact lineage on the fixtures.

FIT-019 then tested the distinct opacity-gauge question on eight disjoint procedural families.
Exact half-opacity refinements preserve rendering but changed raw alpha-1 physical-group selection
on both seeds for all 8/8 families; aggregate-first group scoring restored both-alpha top-8 actions
in 16/16 checkpoints. That correctness result did not become a better allocator: quotient alpha 1
won only 5/8 families at post-20, was `-0.6007 dB` versus raw gauge-row alpha 1 at post-100, and
missed the post-20 support floor. Keep exact groups as a benchmark oracle; ADR-0014 rules out
production lineage/quotient state from this evidence. That result motivated a quality/convergence
question about why fresh-optimizer recovery reverses across horizons.

FIT-020 resolved that response question with 432 trajectories over six new procedural families.
The signal was ample (`SD(y)=3.2529 dB`; 35/36 held-out cells exceeded `0.10 dB`), but the frozen
response bend was slightly worse than the early baseline (`2.9641` versus `2.9616 dB` RMSE), kept
`-1.0455 dB` bias, improved only 2/6 families, and changed none of the 12 held-out actions. Its
selector regret was `1.1116 dB`, worse than the observed-step-10 comparator's `0.7669 dB`.
ADR-0015 closes this bend/predictor lineage without tuning. Concentrated C5/C6 arms show a large but
family-sensitive descriptive quality signal, not a promoted policy. No production code changed;
all arms have N=40 and no encoded stream, so there is no speed, compression, or expressiveness
claim.

COMP-006 then completed the exact marginal cold-stream test on 18 disjoint procedural development
targets x two seeds. At `matched no-edit + 16 bytes`, the best of 16 standard births lost
`-1.0714 dB` mean paired PSNR to the strongest precision/no-edit plus count-neutral replacement
envelope; the 95% family-bootstrap interval was `[-1.2873, -0.8417] dB`, all 6/6 family means were
negative, and an exact same-source replay matched. The confirmation split remains sealed. Actual
bytes still changed the exact selected row in 22/36 cells relative to nominal raw bits, but broad
action class changed in only 2/36 and birth won only 5/36. ADR-0016 keeps the operational-RD oracle
as benchmark infrastructure and closes this standard-birth formulation. Performance profiling and
a real equal-byte richer-atom codec remain independent next lanes.

FIT-015 adds opt-in `checkpoint_policy=best_psnr_final_count`. It selects only post-transition
states with the terminal Gaussian count and writes a same-trajectory audit. On COCO4 x seeds 0/1,
640 Gaussians, and 5k steps, 7/8 runs selected an earlier full-count state and improved their own
terminal means by +0.7702 dB PSNR, +0.00892 MS-SSIM, and +0.0076 LPIPS gain. At 500 steps it kept
the terminal state in 7/8 runs and was effectively neutral. A 72-trajectory Kodak4 confirmation
across max-side {160,240,320} and N={1280,2560,5120} gained +0.4884 dB pooled PSNR, but the gain
fell from +1.0380 dB at N=1280 to +0.0458 dB at N=5120. Keep the compute-minimal terminal policy
as the universal default; use checkpoint selection for sparse/moderate-density long-horizon
quality runs. FIT-013's Sobel loss and FIT-014's covariance filter remain experimental and
default-off.

## Verification status
Init-time math is validated numerically in this environment: structure-tensor orientation/labels,
density concentration, WSE exact-count + blue-noise spacing + density adaptivity, unit-area
anisotropy metric, and the conic inverse-covariance + render compositing formulas (NumPy mirror).
The PyTorch modules compile and are covered by tests that run once `torch` is installed
(`pytest -q`); run the smoke test locally to confirm the fit loop end-to-end on your hardware.
The completed available-repository fixed-storage benchmark writes per-image byte sizes,
5,376-Gaussian quality/convergence metrics, cold-decode codec metrics, and explicit completeness
into `results/storage_budget_168k_external_present/index.html`; the portable multi-report entry
point is `results/index.html`. Its analytical and actual rates are reported separately and the
report must not be presented as an actual-rate compression comparison.

**Reproducibility caveat.** Every benchmark writes a `config.json` (resolved args + device +
torch/numpy/structsplat versions + repository commit/dirty diff fingerprint) so a run is
source-bound from its own artifacts. Results are
bit-exact from a seed only on **CPU**: the CUDA renderer accumulates with atomics
(`atomicAdd` / `index_add`), so GPU renders vary run to run — the logged renderer/device/versions
bound that variation. See the `structsplat-benchmark` skill for the full experimental-validity rules.

## Selected references
GaussianImage (ECCV 2024) · Image-GS (SIGGRAPH 2025) · GaussianImage++ (AAAI 2026) ·
Structure-Guided Allocation (2025) · SAD, SGI, AIR, CGVQ (2026) · P-GSVC · Contour-Aware 2DGS ·
WIPES (ICCV 2025) · Instant-GI · SteepGS (CVPR 2025) · *Revising Densification in Gaussian
Splatting* (ECCV 2024) · Li & Wei, *Anisotropic Blue Noise Sampling* (SIGGRAPH Asia 2010) ·
Yuksel, *Sample Elimination* (EGSR 2015).

## License
MIT.
