# Benchmarks

All benchmark scripts write machine-readable rows plus enough resolved configuration to make a run
self-describing. See the `benchmark` skill for the full protocol.

## Supported operational workflows

Use the four top-level workflow scripts for routine conversion and current-profile evaluation:

```bash
python scripts/convert.py SOURCE OUTDIR [--mask-dir MASKS]
python scripts/benchmark.py SOURCE OUTDIR --seeds 0 1 [--baselines gaussianimage image_gs]
python scripts/ablation.py SOURCE OUTDIR --seeds 0 1
python scripts/stage_search.py IMAGE OUTDIR --stage STAGE --seeds 0 1
```

`scripts/convert.py` is the sole conversion CLI; the other three scripts are evaluation
workflows. All four delegate to the same `structsplat.pipeline.RECIPE` / `PipelineConfig`
definition and share `structsplat.workflows`' artifact/report contract (ADR-0025/0027/0028).
Every result includes resolved configuration, source/target hashes, raw JSON/JSONL/CSV, failures,
timings, curves, reconstructions, fixed-scale errors, intermediate accepted states, and a
relative-link `index.html`. A parallel mask tree turns on boundary-specific
initialization/containment/loss/proposals; no mask keeps the same count, phase, and budget contract
while the closure slot uses general proposals.

The modules documented below remain focused research harnesses and implementation APIs. Historical
shell/Python launchers moved to `deprecated_scripts/`.

For any substantial benchmark or visual-audit result that produces plots, reconstructions, or
comparison grids, also write a local `index.html` overview in the result directory. It should embed
the key diagrams/images, link the raw CSV/JSON artifacts, and state whether visuals are original
saved renders or matched reruns. This keeps ignored `results/` artifacts inspectable without
requiring the reader to hunt through subfolders.

`common.py` is shared benchmark plumbing (`BENCH-003`): image load/save, trajectory PSNR AUC,
JSON/CSV row writing, seed/config helpers, and shared COCO comparison analogue builders. It is not
a CLI; scripts import it to avoid drifting helper behavior.

## BENCH-019 downstream-objective adapter

`stage1_downstream_objective.py` is the passive cross-repository adapter for the proposed
Observation Field V2 gate. It does not load, convert, or launch a field. Realtime-gs remains the
authority for exact additive or normalized `.rtgsv` queries and downstream execution; this module
hash-binds its task, data, environment, schedule, source-field manifests, source pixels, masks,
cameras, splits, and clean repository identities. The task-local wrapper is
`scripts/experiments/bench019_stage1_downstream_objective.py`.

The lifecycle separates exact design review from outcomes:

```bash
python -m benchmarks.stage1_downstream_objective template --output protocol.draft.json
python -m benchmarks.stage1_downstream_objective prepare-review \
  --draft protocol.draft.json --output protocol.review.json
python -m benchmarks.stage1_downstream_objective review-template \
  --protocol protocol.review.json --output prospective-review.json
# A distinct reviewer edits and approves prospective-review.json without outcome access.
python -m benchmarks.stage1_downstream_objective finalize \
  --reviewed protocol.review.json --review prospective-review.json \
  --output protocol.frozen.json
python -m benchmarks.stage1_downstream_objective plan --protocol protocol.frozen.json
```

After realtime-gs executes the frozen command and exports
`structsplat.bench019.cell.v1` rows, analysis is one command:

```bash
python -m benchmarks.stage1_downstream_objective analyze \
  --protocol protocol.frozen.json --rows downstream_rows.jsonl \
  --outdir results/bench019_downstream_objective
python scripts/check_report_bundle.py results/bench019_downstream_objective
```

The report averages downstream seeds/initializers inside each frame-family unit, ranks families
within frames, bootstraps capture groups rather than views, includes leave-one-frame-out and A/A
semantic/config replay checks, retains missing/error cells, and follows the preregistered
single-predictor priority without constructing a post-hoc blend. A general-surrogate protocol must
retain BENCH-019's minimum of two frames, three independent capture groups, and three downstream
seeds. A smaller Janelle-only run can be labelled workload-specific, but cannot authorize a
general Stage-1 surrogate.

## BENCH-020 field-semantics factorial

`field_semantics_factorial.py` is the default-off controller for selecting Field V2 semantics; the
task-local wrapper is `scripts/experiments/bench020_field_semantics_factorial.py`. It freezes three
outcome-separated phases: a fixed-geometry coefficient/DC screen, a matched development factorial,
and one sealed confirmation after distinct development-results review. The controller plans cells
for fixed-row and equal-canonical-raw-byte lanes and consumes schema-bound rows from a pinned
executor; it does not silently substitute a second fitter.

```bash
python -m benchmarks.field_semantics_factorial template --output protocol.draft.json
python -m benchmarks.field_semantics_factorial prepare-review \
  --draft protocol.draft.json --output protocol.review.json
python -m benchmarks.field_semantics_factorial review-template \
  --protocol protocol.review.json --output protocol-review.json
# A distinct outcome-unseen reviewer approves the exact design digest.
python -m benchmarks.field_semantics_factorial finalize \
  --reviewed protocol.review.json --review protocol-review.json \
  --output protocol.frozen.json
python -m benchmarks.field_semantics_factorial plan \
  --protocol protocol.frozen.json --phase coefficient_screen \
  --output coefficient-plan.json
```

Every successful result row preserves a sealed field payload and binds its format/hash/bytes in the
semantic manifest alongside the authoritative pre-clamp render, metric receipt, and convergence
history. First-hit time and normalized PSNR-time AUC replay under
the frozen wall-time horizon; DC/background, packed alpha, structural mass, factorized opacity,
and metadata bytes have separate ledgers. Later outcome roots must be empty at each decision
boundary, and all row artifacts must remain inside their frozen phase root. Missing/error cells
fail closed. The development gate uses capture-cluster bootstrap comparisons against both the
incumbent additive and normalized-plain matched controls, then advances only one nondominated
candidate without a hidden scalar score.

After development analysis, generate a distinct results-review receipt before locking confirmation;
after confirmation, generate a distinct final audit before any claim-ready report. Both the
task-local checker and the maintained shared checker accept the portable bundle:

```bash
python -m benchmarks.field_semantics_factorial check-report results/bench020
python scripts/check_report_bundle.py results/bench020
```

This is currently experiment substrate, not a semantic verdict. A general protocol needs disjoint
development and confirmation data with at least three independent capture groups in each split,
three seeds, the BENCH-019 downstream response, exact executor contracts, and distinct prospective
and results reviews. The supplied Janelle frame alone can support only a diagnostic,
workload-specific comparison.

The canonical four-image COCO fixture used by the matched comparison and regression-bisect
harnesses lives in `tests/test_images/`. Keep those four files there so benchmark reruns do not
depend on ignored `results/` artifacts.

`actual_rate_phase_diagram.py` owns BENCH-007, the decision benchmark for any compression claim.
It freezes native decoded-pixel/source hashes and one equal count/bit-mix grid before scoring,
then independently fits all six allocation arms, writes complete SSPL1 streams, cold-decodes and
centrally scores them, and journals every fit/candidate for safe resume. `analyze` applies integer
byte caps, explicit missing-point semantics, nondominated envelopes, no-extrapolation BD-rate,
image-cluster bootstrap/Holm summaries, the Stage-1 stop/go gate, F5--F9, and a portable
`index.html`. `conventional` writes separately labeled PNG/JPEG-444/AVIF-444 context; those rows
never enter the gate. Install the optional dependencies with `pip install -e '.[benchmark,metrics]'`.

The frozen 2026-07-14 Stage-1 run is complete and negative: 288/288 fits and 1,152/1,152 latest
validated candidates, but tensor-WSE fails the quality-at-both-rates, time, and texture guards
against the strongest local gradient control. Stage 2 is not authorized. The result and artifact
hashes are in `ara/evidence/bench007-stage1-killing-pilot-2026-07-14/run.md`.

```bash
# Stage 0b: rate calibration only (the command enforces IDs 0002/0268/0534/0800).
PYTHONPATH=src python -m benchmarks.actual_rate_phase_diagram calibrate \
  --data-root results/datasets/DIV2K_train_HR \
  --images results/datasets/DIV2K_train_HR/{0002,0268,0534,0800}.png \
  --outdir results/bench007_stage0b --renderer cuda --device cuda

# Freeze before metric inspection, inspect the exact workload, then run/resume.
PYTHONPATH=src python -m benchmarks.actual_rate_phase_diagram freeze \
  --stage stage1 --data-root results/datasets/DIV2K_train_HR \
  --images results/datasets/DIV2K_train_HR/{0001,0115,0229,0343,0457,0571,0685,0799}.png \
  --bytes-per-gaussian BPG_FROM_STAGE0B --renderer cuda \
  --manifest results/bench007_stage1/manifest.json
PYTHONPATH=src python -m benchmarks.actual_rate_phase_diagram plan \
  --manifest results/bench007_stage1/manifest.json
PYTHONPATH=src python -m benchmarks.actual_rate_phase_diagram run \
  --manifest results/bench007_stage1/manifest.json \
  --data-root results/datasets/DIV2K_train_HR --outdir results/bench007_stage1
PYTHONPATH=src python -m benchmarks.actual_rate_phase_diagram status \
  --manifest results/bench007_stage1/manifest.json --outdir results/bench007_stage1
PYTHONPATH=src python -m benchmarks.actual_rate_phase_diagram analyze \
  --manifest results/bench007_stage1/manifest.json \
  --data-root results/datasets/DIV2K_train_HR --outdir results/bench007_stage1
```

The public SLIC/Sobel description omits SLIC settings and the dynamic allocation constants. The
registered `local_slic_sobel_control` therefore freezes native SLIC at a 1,024-pixel target region
area and the reported sparse 6:2:1 allocation, serializes every assumption, and is never labeled as
upstream paper code. Scientific Stage-1/2 manifests enforce the preregistered image IDs; only
Stage-0a plumbing tests may opt into a subset.

`renderer=cuda` is StructSplat's owned exact implementation of the normalized weighted-sum
equation, not the semantically different alpha/additive comparator. BENCH-007 freezes both the
equation and implementation in its manifest, requires CUDA for a CUDA-frozen run, and retains the
same `1e-6` cold-field parity tolerance. The slower `renderer=normalized` reference remains
available for portability and oracle checks.

Cold parity is evaluated at the decoded-field boundary: the in-memory encoded blob and the bytes
read back from the persisted SSPL1 file must produce identical decoded-state hashes and satisfy the
frozen maximum-absolute tolerance. The cold field is then rendered once for central scoring. Do not
compare two exact-CUDA renders as the parity oracle, because independent atomic accumulation orders
can differ by a few float32 ulps even for identical fields. After a validator change,
`run --revalidate-candidates --retry-failed` re-encodes and revalidates every saved-field candidate
without refitting.

`ablation.py` runs the core experiment (`ABL-001`): `{init strategy} x {budget}` on fixed images,
scored on PSNR / MS-SSIM / LPIPS + iterations-to-target. Caveat: this is the broad init sweep, so
keep image/budget/seed axes explicit in the output config. ABL-004 control labels are available
alongside the core strategies: `floyd_steinberg`, `density_random`, and `random_relocate`.
Long runs write `ablation.jsonl` incrementally; use `--resume` to skip cells already present there.
For the ABL-004 protocol, `deprecated_scripts/run_abl004_full_ablation.sh` prepares Kodak-24 under
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
FIT-017 adds an orthogonal sampled-add score axis:
`--refine-score-modes legacy_abs gaussian_abs signed_gaussian`. `gaussian_abs` is the required
same-width magnitude control for the signed-coherence hypothesis. The deterministic COCO4 x
two-seed 64->80 guard rejected both wider scores after recovery: signed Gaussian gained
+0.5199 dB immediately but lost -0.0318 dB after 20 steps and -0.2301 dB after 100. Keep
`legacy_abs` as the default. Reproduce the shared-start guard with:

```bash
python -m benchmarks.sampled_add_score_compare \
  --outdir results/fit017_sampled_add_score_guard \
  --seeds 0 1 --max-side 64 --start-count 64 --add-count 16 --pre-iters 40 --device cpu
```

FIT-018 adds the opt-in normalized-ownership site `--refine-sites responsibility`, with fixed
`--responsibility-mass-alpha` (default `0.7`). Its shared-start mechanism guard compares center
residual, raw-support residual, responsibility alpha 1, and responsibility alpha 0.7 under the
same moment-preserving 64->80 split and independent 20/100-step recovery replays:

```bash
python -m benchmarks.responsibility_split_compare \
  --outdir results/fit018_responsibility_split_guard --device cpu
```

The frozen guard rejected the alpha-0.7 donor arm against the stronger `support` control:
post-20 `-0.0198 dB`, 4/8 positive pairs, post-100 `-0.0411 dB`, and `+1.8%` total-100 time. It
passed recovery-loss, count, numerical, and timing limits but failed both post-20 quality gates.
Keep `responsibility` opt-in as a mechanism control and do not tune this lineage on the four
fixtures. Alpha 1 was effectively tied at post-20 and is evidence only for a separate
duplication-invariance question.

The benchmark enforces one CPU thread and PyTorch deterministic algorithms; a second source-frozen
replay matched every non-timing aggregate exactly. `config.json` hashes every input image and all
relevant source files, including the otherwise-untracked benchmark module.

FIT-019 is the benchmark-only opacity-gauge audit for the normalized renderer. It replaces even
canonical rows by exact co-located half-opacity copies, compares raw-row and aggregate-first
responsibility allocation, then maps every selected group action back to the same canonical N=32
checkpoint for an equal eight-row moment-preserving increment. Recovery at 20 and 100 steps uses
independent fresh Adam restarts; it is a mechanism guard, not production optimizer-state evidence.

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
  python -m benchmarks.gauge_equivalence_audit \
    --outdir results/fit019_opacity_gauge_guard_v2_fresh
```

The audit-corrected v2 run confirms commutation: equivalent renders differ by at most `8.345e-7`,
both-alpha quotient top-8 actions match 16/16 checkpoints, and raw alpha-1 physical-group
multisets change on both seeds for all 8 target families. The recovery-utility guard fails.
Quotient alpha 1 versus raw gauge-row alpha 1 is `+0.2111 dB` at post-20 but only 5/8 target-family
wins and `-0.6007 dB` at post-100; it is also `-0.0665 dB` versus canonical support at post-20.
Keep grouping benchmark-only. Primary and replay contain verified 24-file source snapshots and
match exactly on every non-timing row/aggregate field. Timing varies from `+1.38%` to `-1.26%`
overhead and is not a speedup claim.

FIT-020 is the benchmark-only ranked deduplication perturb--recover assay. It freezes six
procedural families x six variants, train/within-assay held-out procedural-variant splits (not
confirmation or natural-image evidence), three repeated seeds, four
equal-N birth paths C5--C8, dense 200-step recovery curves, target-grouped ridge CV, one response-
bend feature, and prediction/selection killing gates. It identifies the whole ranked ticket-
replacement path, not pure distinct-site coverage.

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
  python -m benchmarks.perturb_recover_spectroscopy \
    --outdir results/fit020_response_spectroscopy_v1_fresh \
    --size 48 --seeds 0 1 2 --start-count 32 --add-count 8 \
    --pre-iters 40 --recovery-iters 200 --device cpu --render-chunk 512
```

The completed primary contains 432 trajectories and 108 seed-averaged pairs. Integrity and signal
pass (`SD(y)=3.2529 dB`; 35/36 held-out cells have `|y| >= 0.10 dB`), but the response claim fails:
response/early RMSE is `2.9641/2.9616 dB`, sign accuracy is the same `69.44%`, response improves
only 2/6 families, bias is `-1.0455 dB`, and the bend changes no held-out action. Response regret
is `1.1116 dB` versus `0.7669 dB` for observed step 10. The decision is **stop**; do not retune the
bend, horizons, model, or exposed targets.

Primary trajectory computation finished before an output-only paired-CSV header bug surfaced.
Finalization used its immutable rows and frozen source snapshot, proved the aggregate unchanged,
and repaired the writer to use the union schema. The post-fix replay is measurement-equivalent,
not literally untouched source: every compared non-timing measurement field (excluding source
provenance), paired row, normalized aggregate, manifest, and decision matches exactly; only the
writer and its regression test differ in source.
See `docs/research/2026-07-15-perturb-recover-spectroscopy.md` for the full audit. C6's descriptive
late gain is family-sensitive and not a promoted allocator. Fixed N and missing stream bytes also
preclude compression or expressiveness claims; dense CPU instrumentation is not a speed result.

COMP-006 is the benchmark-only marginal cold-stream RD audit. It starts every branch from one
persisted/cold-decoded N=64 SSPL1 parent, compares 16 standard births with 16 same-candidate
birth-for-death replacements and an exhaustive 875-mix precision envelope, and selects by
cold-decoded MSE under integer complete-stream caps. Every candidate includes headers, ranges,
framing, all attribute streams, and zlib/Morton context; its byte delta is not an incremental patch
or additive row price.

```bash
PYTHONPATH=src:. python -m benchmarks.marginal_cold_stream_rd run \
  --outdir results/comp006_marginal_rd_dev_v1_2026-07-15 \
  --split development --shard-index 0 --num-shards 3
# Run shard-index 1 and 2, then seal all three.
PYTHONPATH=src:. python -m benchmarks.marginal_cold_stream_rd finalize-run \
  --outdir results/comp006_marginal_rd_dev_v1_2026-07-15 \
  --split development --num-shards 3
PYTHONPATH=src:. python -m benchmarks.marginal_cold_stream_rd analyze \
  --outdir results/comp006_marginal_rd_dev_v1_2026-07-15 --split development
PYTHONPATH=src:. python -m benchmarks.marginal_cold_stream_rd verify-replay \
  --primary results/comp006_marginal_rd_dev_v1_2026-07-15 \
  --replay results/comp006_marginal_rd_dev_v1_replay_2026-07-15
```

The primary and replay each contain 36 cells and 33,840 validated streams and match exactly on all
scientific fields. At recovery step 20 and the frozen +16-byte cap, birth lost `-1.0714 dB` mean
PSNR to the strongest control, with family-bootstrap 95% interval `[-1.2873, -0.8417] dB` and
0/6 positive family means. The decision is **stop**; confirmation was not run. Exact and nominal-
raw-bit oracles agreed on only 14/36 rows, so exact-rate selection remains useful infrastructure,
but action class agreed in 34/36 and birth won only 5/36. See
`docs/research/2026-07-15-marginal-cold-stream-rd.md`; no production code or default changes.

BENCH-009 is the completed residual tangent-space auction. Stage 0's deterministic float64 CPU
identifiability control passed, then the frozen Stage-1 development and recovery workload produced
`4,608` immediate cells, `816` trajectories, and `2,448` logical recovery checkpoints with exact
bindings and joins.

```bash
PYTHONPATH=src:. python -m benchmarks.residual_tangent_auction \
  --outdir results/bench009_tangent_auction_stage0
```

The final decision is negative/unavailable. Global causal calibration is `0.268549 < 0.8`; every
causal action-by-horizon stratum fails; and the independently truncated base/joint spaces produce
negative “incremental” projector energies. Affine and carrier both lose immediately and at step 20
against their stronger matched control. Carrier's later positive mean is descriptive and does not
survive the frozen two-radius/all-horizon gate. No expressiveness, optimizer, performance,
compression, or production claim is authorized. See
`docs/research/2026-07-16-bench009-results-audit.md`.

BENCH-011 v1 is an audited invalid run: it seeded each randomized base factorization from a
diagnostic-unit hash rather than BENCH-009's exact cross-fit unit hash. Corrected v2 binds all 24
exact BENCH-009 IDs and base ranks, passes algebra and bit-exact replay checks for all `96` rows,
and fails every frozen calibration stratum. Affine/carrier Spearman correlations are `0.400/0.385`
at radius `0.25` and negative at `0.75`; all `48` radius-`0.75` native rows lose. Close the
current-identity local-linear formulation without retuning or disjoint-data expenditure.

```bash
PYTHONPATH=src:. python -m benchmarks.nested_residual_extension_diagnostic controls
PYTHONPATH=src:. python -m benchmarks.nested_residual_extension_diagnostic run \
  --runner-dir results/bench009_tangent_auction_stage1_runner_v3 \
  --science-dir results/bench009_tangent_auction_stage1_science_v3 \
  --outdir results/bench011_nested_extension_diag_v1 \
  --num-shards 1 --shard-index 0
```

For parallel reproduction, write disjoint shard directories and merge them with the module's
`merge` subcommand. V2's canonical merge returns expected exit code `2` (`diagnostic_fail`). Do not
change the frozen identities, rank threshold, damping, radii, eligibility floor, or gates.

BENCH-012 is the closed spatial-connectivity policy-value preflight. Its pure NumPy topology core
freezes 24 rectilinear binary targets, dual 4/8 digital topology, and a target-independent anchored
connectivity partition distance. The core/focused suite passes `17/17` tests. The first
source-bound action cell nevertheless retained only 2 of the required 4 untruncated equal-work
COMP-006 candidates, so the runner failed before rendering a replacement, selecting an action, or
running recovery.

```bash
PYTHONPATH=src:. python -m benchmarks.topology_policy_value controls
PYTHONPATH=src:. python -m benchmarks.topology_policy_value run \
  --outdir results/bench012_topology_policy_preflight_v1 --max-cells 1
```

The second command is expected to fail closed with the recorded 2/4 feasibility error. Do not
relax the support/work filter or retune the exposed targets/action scale. The artifact is an
availability result only; it says nothing about topology, quality, convergence, compression, or
expressiveness. See `docs/research/2026-07-16-bench012-preflight.md`.

COMP-007 is the completed gauge-free covariance codec assay. It exhausts 84 covariance-bit
allocations, two predictors, three covariance charts, and zlib/zstd complete-stream coders on 24
frozen even-Kodak fields. Protocol-v3 v4 passes the full source/data/byte/re-encode audit and
independently replays all 12,096 decoded renders. The `log_spd` chart nevertheless fails seven of
eight gates: median whole-container change is `-0.4053%` with zlib and `+0.3426%` with zstd,
versus the required `+1%`, with only `5/12` and `7/12` image wins. Confirmation remains sealed.

```bash
sha256sum results/comp007_gauge_free_covariance_dev_v4/{config.json,fields.jsonl,\
candidates.jsonl,artifact_audit.json,analysis.json,executed_sources_v3.tar}
jq '{audit_passed:.checks.artifact_audit_passed,status:.analysis.status,\
decision:.analysis.decision,confirmation:.analysis.confirmation_authorized}' \
  results/comp007_gauge_free_covariance_dev_v4/analysis.json
```

The expected decision is `kill` with `artifact_audit_passed=true`. V2 and v3 are immutable
pre-scoring unavailable artifacts, and v4 is immutable: do not rerun analysis into its canonical
directory after the status-bearing task file changes. Do not repair earlier bundles in place,
retune the chart, or open the odd Kodak IDs. See
`docs/research/2026-07-16-comp007-gauge-free-covariance.md`.

BENCH-013 through BENCH-015 are the closed first-order-reproduction lineage. BENCH-013's
per-pixel local-linear compositor proved affine reproduction but failed because compact irregular
support produced signed leverage and ringing. BENCH-014's explicit affine carrier fixed that
quality problem and added two operator columns, but its six transmitted scalars plus residual-color
entropy failed every complete-byte gate and its convergence guard. BENCH-015 removed all added
state by deriving a robust global plane from the ordinary decoded colors:

```text
beta = robust_fit(X_mu, c)
r    = c - X_mu beta
y    = X beta + W r
```

The canonical BENCH-015 artifact is
`results/bench015_decoder_synchronized_lift_stage0_v1_2026-07-16`. It passed all `27` replay
checks and is a valid scientific kill. Static same-stream MSE ratios are `0.670143`, `0.109768`,
and `0.558233` on the three smooth families, with equal DSL78/NW78 complete bytes. The registered
no-harm, convergence, and cold-decode gates fail: continuous-crease target-range excursion reaches
`0.029832`, smooth median final-loss ratio is `1.058438` with worst `1.400686`, and cold
decode+derive+render is `1.624762x`. Prepared rendering itself is `1.009728x` and passes.

```bash
taskset -c 14 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. \
  python -u benchmarks/decoder_synchronized_lift_assay.py \
    results/bench015_decoder_synchronized_lift_stage0_v1_2026-07-16 --root . --replay
```

Do not retune this exposed robust/global confidence mechanism or run its conditional local
successor: that branch required convergence and cost to pass. See
`docs/research/2026-07-16-decoder-synchronized-affine-lift.md` for the exact hashes and claim
boundary.

`exact_backward_profile.py` owns PORT-004's source-bound untiled exact-normalized CUDA backward
profiles. It records CUDA-event medians for forward, renderer backward, complete loss backward,
Adam update, and a representative micro-fit step over a frozen resolution/count/overlap grid.
Support visits and source atomic callsites are proxies, never measured speedups.

The historical actionability artifact and exact executed-source snapshot are preserved at
`results/bench010_exact_backward_profile/` (snapshot-manifest SHA-256
`4b6cf8805132fbb4e1110c046cf6e785dca1b59240b7731807ae0a2244d92884`). The live module now owns
the follow-on block-reduction comparison; do not rerun it into the historical directory.

On the RTX 3050 representative `256x256`, `N=2048`, overlap-16 cell, exact backward measured
`1.120256 ms` versus `3.377120 ms` for the representative device-side step (`33.1719%`), passing
the frozen `>25%` actionability gate. All eight untiled/tiled parity cells passed, while tiled full
steps were `1.084x--1.752x` the untiled time. Nsight Compute counters were permission-blocked. This
authorizes only an opt-in block-reduction implementation experiment; no speedup, end-to-end fit,
quality, convergence, cross-GPU, or compression claim is made.

The follow-on experiment keeps `cuda` unchanged and compares it with the internal experimental
`cuda_block_reduce` selector, which reduces each Gaussian's 8/9 gradient components within its
own block and performs one direct write per component:

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
  python -m benchmarks.exact_backward_profile \
    --outdir results/bench010_exact_backward_block_reduce_rerun
```

The primary representative cell measured `-57.478%` exact-backward and `-25.288%` device-step
time with zero allocator-memory regression and full parity. Do **not** promote the literal primary
artifact pass: PORT-004's governing gate also requires direction retention over the entire frozen
grid, which the executable predicate omitted. Four `N=512` exact-backward ratios were
`1.0290/1.0100/0.9938/1.0231`, and one full-step ratio was `1.0052`. An identical independent run
repeated the large representative reductions but also failed the unchanged repeat-stability limit
(`5.1154%` candidate-backward CV versus `5%`). The source-bound audit and all eight ratios are in
`results/bench010_exact_backward_block_reduce/audit.md`. Keep this path benchmark-only; it is not
exposed in the fit CLI or broad ablation choices, and no default, end-to-end, universal-speed,
quality, convergence, compression, expressiveness, or cross-GPU claim is authorized.

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
decision-grade fitter deltas: `deprecated_scripts/run_abl005_cuda_native_influence.sh` covers the six
CUDA-native knobs and can support quality/convergence/speed claims, while
`deprecated_scripts/run_abl005_affine_quality_influence.sh` isolates `color_basis=affine` as
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
adaptive growth (up to a `1.5x` cap in ordinary lanes) for reducing absolute diff. The experimental
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

`storage_budget_compare.py` is the frozen equal-capacity/convergence lane. It interprets 168 KiB
as exactly 172,032 bytes and counts the common frozen constant-RGB RS payload only: mean x/y,
log-scale x/y, rotation, and RGB, all float32. That is 32 bytes/Gaussian and exactly 5,376
Gaussians. Source file bytes, prepared-target PNG bytes, reconstruction PNG bytes, decoded float32
array bytes, and actual SSPL1 stream bytes are distinct columns; none is silently substituted for
the analytical payload. The intended registry snapshot had 41 methods; the completed
external-present run had 40 because Instant-GI was absent, and finished 320/320 COCO4 x seed
cells at max-side 160 with a 10k ceiling. All scheduled growth finishes before the 6,500-iteration
plateau gate; six consecutive 100-step evaluations without a 0.005 dB gain stop a run. The scored
reconstruction is the convergence endpoint, including checkpoint selection and final color solve,
and early exits hold that endpoint to the nominal AUC horizon. Max-horizon cells are reported as
right-censored. When available, Instant-GI's native under-allocation is filled with deterministic random target
samples while preserving native/fill counts; the adaptive arm stops adaptive additions at the
exact cap, finishes scheduled fill, and continues optimization. This remains a local analogue
comparison, not a native-codec byte match.

Do not use this lane for a compression or SOTA decision. At the four prepared resolutions its
172,032-byte analytical payload is 71.68–81.15 bpp; completed SSPL1 rows are about 22 bpp, versus
17.99 bpp average for the lossless target PNGs. BENCH-007 owns the replacement protocol:
self-contained SSPL1 targets at 0.25/0.5/1/2/4 bpp, equal codec search, direct structure controls,
and BD-rate over measured overlap. Its completed Stage-1 outcome used development images, failed
the frozen gate, and therefore prohibited the planned held-out Stage 2.

```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  STRUCTSPLAT_INSTANT_GI=/path/to/Instant-GI/quard_image.py \
  python -m benchmarks.storage_budget_compare
```

Each run writes per-image byte/metric tables and its own `index.html`, then
`results_index.py` builds a portable `results/index.html` from an explicit report list. The root
dashboard never scans `results/` or silently promotes stale directories.

`structsplat_best_checkpoint` sets `checkpoint_policy=best_psnr_final_count`. Checkpoint scoring is
post-transition (after the optimizer step and any prune/grow/relocate/on-split solve), and only a
state with the terminal Gaussian count may be restored. The fitter retains its pre-step training
history; fair reports replace only the final convergence sample with the scored reconstruction so
AUC/plots agree with the selected output. Iteration count and fit timing remain intact.
`checkpoint_selection.csv` is the causal audit:
it compares selected and terminal states from the same trajectory/count, avoiding false
attribution from nondeterministic CUDA trajectories. On COCO4 x seeds 0/1, N=640, 5k steps, this
policy selected earlier states in 7/8 runs and gained +0.7702 dB PSNR, +0.00892 MS-SSIM, and
+0.0076 LPIPS on average. At 500 steps it selected an earlier state only once, for a negligible
+0.0066 dB mean PSNR gain with small SSIM/LPIPS tradeoffs. It is therefore a long-horizon quality
option, not the pinned general default. A broader 72-trajectory Kodak4 confirmation at max-side
{160,240,320}, N={1280,2560,5120}, and seeds {0,1} gained +0.4884 dB pooled PSNR (95% image-
bootstrap CI [+0.4167,+0.5304]), +0.00433 MS-SSIM, and +0.00736 LPIPS. The gain falls from
+1.0380 dB at N=1280 to +0.0458 dB at N=5120, with saturated strata often retaining the terminal
state. Use the policy for sparse/moderate-density long fits; keep the compute-minimal terminal
policy as the universal default. New runs also write `checkpoint_selection_summary.csv` with
image-clustered pooled and per-budget intervals.

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

The preregistered COCO4 x seeds 0/1, N=640, 500-step guard rejected this exact `2x_f10`
curriculum: direct gain over the checkpoint control was -0.1645 dB selected PSNR (95% image-
bootstrap CI [-0.2856,-0.0677]), -0.00068 MS-SSIM, -0.0716 dB AUC, and -0.0030 LPIPS gain. These
miss the allowed -0.05 dB short-horizon loss, so the planned 5k and difficult-Kodak stages were not
run. Keep the fields for reproducible research, but do not use this arm as a quality default.

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
`libstdc++` preload. `deprecated_scripts/setup_native_image_gs_env.sh` now creates and verifies the official
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
`deprecated_scripts/setup_native_gaussianimage_env.sh` provisions the isolated Python 3.10,
Torch 2.0.0+cu118 build and records exact dependency/linkage provenance.

The current base-GaussianImage adapter is representation-only. Its `release_cholesky` and
`release_rs` names select covariance form but do not yet enforce the released Kodak protocol, so
they must not be described as native-authentic release/RD runs. Upstream Kodak uses each image's
native 768x512 or 512x768 orientation (393,216 pixels), N={800,1000,3000,5000,7000,9000}, one
ordered seed-1 process per count, 50k representation steps, then another 50k QAT steps; final QAT
evaluation selects the best training-PSNR state. `compress_wo_ec` returns an in-memory dictionary
whose decoder metadata remains in the live model/checkpoint, not a self-contained serialized
stream. The corrected fixed-width no-EC
rate is `56*N + 1728` bits (for N=800: 46,528 bits, 5,816 ideal bytes, 0.118326823 bpp on Kodak),
while `actual_codec_bytes`/`actual_bpp` must remain null. A native QAT profile must preserve these
semantics, export representation and QAT trajectories separately, round-trip the in-memory decode,
and report upstream versus corrected analytical rate without inventing a bitstream.

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

`wse_prefix_audit.py` is INIT-009's terminal-set-preserving ordering audit. It compares historical
candidate-index prefixes with Yuksel-style recursive WSE prefixes on identical survivors. The
uniform Euclidean eight-seed M=2048 -> N=256 audit produced 32/32 descriptive joint
spacing+coverage wins across four correlated prefixes, with the ordering subroutine taking 14.2%
of terminal selection time. This does not measure end-to-end anisotropic/quadtree initialization
overhead or establish optimality over other progressive orders. The initialization flag remains
opt-in for artifact compatibility:

```bash
python -m benchmarks.wse_prefix_audit \
  --outdir results/init009_wse_prefix_audit --seeds 0 1 2 3 4 5 6 7 \
  --candidates 2048 --terminal 256 --prefixes 16 32 64 128
structsplat fit image.png --wse-progressive-order
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
