# CORE-019 calibrated coherent-depth results audit

Date: 2026-08-07

Disposition: **rejected; preserve as a default-off negative control**

Scope: one exposed, reduced-resolution, unmasked development scene and seed; no promotion claim

## Executive result

CORE-019 does not produce an artifact-free or competitive realtime-gs initializer. Its complete
coherent-support/WSE arm is different from, but not uniformly better than, the raw-known-ray VGGT
ablation at the terminal checkpoint: SSIM improves by 0.00197, MS-SSIM by 0.01022, and LPIPS by
0.02062 while PSNR loses 0.3075 dB, gradient MAE worsens by 0.000079, and p99 error worsens by
0.05739. It contains 904 fewer final rows and has a better spacing tail. This mixed result verifies
that the compiler is active without establishing a quality win over balanced top-k selection.

That bounded mechanism result is not enough. The complete arm starts 0.9690 dB below ordinary
interior consensus instead of at least 2 dB above it, fails the step-zero LPIPS gate, fails every
fixed-topology step-500 control comparison, never reaches the interior control's terminal PSNR,
and ends 0.9284 dB below it. Native reporting views are disqualifying: initial coherent-depth
geometry still has black unsupported regions and floaters, while the shared optimizer turns it into
broad gray sheets and radial streaks that erase body, fabric, and stage detail. The interior and
posterior controls are also visibly unusable volumes. The mandatory visual gate and scalar gate both
fail, so no threshold rescue or longer run is authorized on this consumed frame.

## Frozen protocol and provenance

- Scene: `/home/alex/Dropbox/Work/Janelle/karate/frame_00005`, with sibling
  `calibration_dome.json`.
- Construction cameras: the 26 available calibrated cameras other than `C0024`, `C0010`, `C1004`,
  and `C0022`. Those four reporting cameras were chosen from calibration before the frame was
  opened and are excluded from packet creation, depth inference, fusion, support, selection, and
  optimization targets.
- Packet inputs: 26 cold-reloaded, full-canvas WebP-quality-80 `.sgdp` packets with exact alpha,
  1,024 structural proposals per construction view, sigma `0.45`, radius `3`, eight Jacobi steps,
  and deterministic view seeds. No source RGB is opened after packet construction.
- Arms: ordinary CompactCarve interior consensus; CORE-018 posterior without reciprocity; raw
  calibrated VGGT depth on known rays with balanced exact-budget selection; and complete
  projective support plus hard feature anchors, WSE, bounded contraction, and compatible cover.
- Common output/training: 10,000 initial rows, 500 fixed-topology attempted steps, unchanged
  density control from step 600 through step 1,500, 30,000-row cap, seed 0, SH degree 3, no masks,
  and no random background.
- VGGT source revision: `a288dd0f14786c93483e45524328726ab7b1b4ce`. Public checkpoint
  revision: `860abec7937da0a4c03c41d3c269c366e82abdf9`; 5,026,367,224 bytes;
  SHA-256 `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`;
  CC-BY-NC-4.0. The aggregator is bfloat16 and camera/depth heads are float32 under bfloat16 CUDA
  autocast. The checkpoint is a separately reported encoder dependency, not per-scene payload.
- Executed v5 StructSplat revision: `722696c893e4a37cabb69ab24dcf5fcd5d9efb30`, with all method,
  driver, checker, and gate logic committed and only unrelated IntelliJ files dirty. Executed
  realtime-gs revision:
  `36630c7fef14c0907134d2f3c532be3da4a0c43e`, also dirty. Exact used source files are copied into
  the report. The manifest is explicitly `claim_ready=false`; dirty provenance cannot be waived by
  structural report validation.
- Hardware/runtime: RTX 3050 8 GB, torch 2.9.0/CUDA 12.8, Python 3.12.9.

The final immutable execution is
`results/core019_coherent_depth_karate_frame00005_2026-08-07_v5/` (162 MiB). It follows four
preserved attempts. V1 discovered that this frame lacks C0000/C0001 before packet construction; v2
applied the already-frozen available-camera rule but exposed insufficient bounded proposal rounds
for the interior and coherent arms. The v3 correction gave the interior control a 4x proposal pool
and coherent generation twelve bounded rounds. V4 was the first committed-source schema-clean
replay and exposed one conservative decision-bookkeeping defect: the full-vs-raw gate omitted four
already-prespecified quality measures. V5 changes only that pure decision projection and its test.
No reporting metric or visual selected a geometry, support, WSE, contraction, cover, or optimizer
threshold.

## Integrity and diagnostic contract

| Artifact | SHA-256 |
|---|---|
| `manifest.json` | `d196b10fc011a436c2b0b0f8b6fec610c7ac9f53c4906a19aa21b52ece0a5af2` |
| `plan.json` | `7cf4d8dd28f7c65723e724596930f681a7659fb02fabc0e412a1e1d940d2af48` |
| `metrics.json` | `542197a04d79a03afbb8f60f471ef37bbaeb22ce6ae0431cdecae652e270daa6` |
| `decision.json` | `a3d705b5815be0544091ffa4028d3cc95b087ae359f9af177fdd1194686a9719` |
| `all_metric_curves.png` | `0357697354b81ea1fe32b0c66486054fc9af00f9a9b28dcfedc8de5051d1e3e0` |

The v5 status is `ok`, all arms cold-reload an identical packet-hash vector, and total wall time is
142.57 seconds. `python scripts/check_report_bundle.py --allow-dirty RESULTS_DIR` passes. The custom
contract validates the frozen split/checkpoint, exact four arms, packet identity, physical artifact
descriptors and hashes, JSON/JSONL/CSV projection, models, curves, native visuals, HTML exposure,
command, and explicit non-claim flag. Checker acceptance is portable diagnostic handoff only; it
does not override dirty provenance or the failed visual/scalar decision.

## Rate and runtime ledger

The 26 original construction JPEGs total 14,557,266 bytes. Complete packet bytes are 1,256,406,
or 11.586x original/packet; decoded packet PSNR averages 38.646 dB (37.867--39.598 dB). The shared
VGGT checkpoint is 5.03 GB and is separately reported for both coherent arms.

| Arm | Pretrain (s) | Native train (s) | Final N | Final NPZ bytes | Original/(packet + model) |
|---|---:|---:|---:|---:|---:|
| Interior consensus | 10.868 | 10.674 | 25,117 | 5,483,620 | 2.1598x |
| Posterior, no reciprocity | 4.652 | 11.999 | 30,000 | 6,566,549 | 1.8608x |
| VGGT raw known-ray | 25.020 | 10.547 | 15,680 | 3,399,881 | 3.1264x |
| VGGT coherent support/WSE | 26.800 | 10.553 | 14,776 | 3,226,027 | 3.2476x |

The coherent field itself takes 21.748 seconds, including checkpoint hash/load/inference/fusion;
the complete arm then spends 2.830 seconds on lift. These are warm-local-file development timings,
not cold download latency or a production performance claim. The scene-file ratios are physical,
but unusable renders cannot establish useful compression, and the encoder amortization is not
hidden in them.

## Quantitative outcome

| Metric | Interior init | Full init | Interior step 500 | Full step 500 | Interior final | Full final |
|---|---:|---:|---:|---:|---:|---:|
| PSNR (dB) | 7.6551 | 6.6860 | 13.5058 | 10.6497 | 12.8465 | 11.9181 |
| MS-SSIM | 0.05089 | 0.05659 | 0.38578 | 0.29143 | 0.30054 | 0.36055 |
| LPIPS | 0.77244 | 0.84448 | 0.73354 | 0.74051 | 0.74087 | 0.75941 |
| Gradient MAE | 0.02532 | 0.02315 | 0.01985 | 0.02010 | 0.02258 | 0.02004 |

The full-vs-raw mechanism comparison at the terminal checkpoint is PSNR 11.9181 versus 12.2256,
MS-SSIM 0.36055 versus 0.35032, LPIPS 0.75941 versus 0.78003, and gradient MAE 0.02004 versus
0.01996. Its compatible-cover spacing p90/median is 1.8726 versus 1.9560. This is mixed causal
evidence: support/WSE improves perceptual/structural averages and row count while worsening signal
fidelity and error tails.

The scalar decision remains false. The full arm passes exact initial count, final cap, identical
packet, physical scene-ratio, terminal Pareto, and full-vs-raw mechanism gates. It fails the required
step-zero +2 dB and LPIPS gates, all four step-500 comparisons, and the convergence-to-control
target. The interior control reaches its own terminal 12.8465 dB at step 400; the full arm never
reaches it. Density events cause large non-monotone curve discontinuities, and v3/v4/v5 terminal
full-vs-raw PSNR deltas are respectively +0.371/-0.081/-0.307 dB despite matching initial geometry.
Terminal single-run rankings are therefore demonstrably fragile, not evidence of smooth convergence.

Full-arm terminal per-view PSNR is 13.544 dB (`rgb_24`), 9.505 dB (`rgb_10`), 11.103 dB
(`rgb_1004`), and 13.520 dB (`rgb_22`). The wide spread reinforces that a mean metric is not a
substitute for inspecting every reporting view.

## Geometry/compiler diagnosis

Twenty-three of 26 calibration groups pass the frozen camera/depth gates. Fused depth has median
3.977 and median relative uncertainty 0.01577; the learned scale is therefore producing finite,
spatially coherent construction state rather than the high-entropy independent-ray posterior of
CORE-018. This does not imply physical depth because no depth ground truth exists.

The complete lift starts with 40,000 proposals. Projective acceptance leaves 29,844 candidates.
The 1,500 hard feature anchors all survive; dynamic WSE removes 19,844 rows to exactly 10,000 and
keeps a 192-row per-view floor. Bounded contraction absorbs only 96 eliminated cross-view proposals,
moves 0.96% of survivors, and caps displacement at 0.00316 scene units. It is not the source of a
hidden count change. The compatible surface cover uses three neighbors for almost every row; only
44 rows (0.44%) fall back to nearest geometry, while the strictest-visible-camera two-pixel cap is
active for 40.27% of rows. Median tangent sigma is 0.01030 and median opacity is 0.7190.

These receipts show that the candidate is neither a pixel/voxel/KD-tree snap nor an uncontrolled
merge. The remaining failure is more fundamental: a coherent per-view depth raster plus local
surfel compilation does not yield the globally consistent, optimization-stable surface topology
required by the current unconstrained 3DGS density schedule.

## Native visual review

The field contact sheet is semantically coherent: high-confidence depth follows the actor and stage,
with no obvious image-grid lattice. The four native arm contact sheets and common metric curves were
then inspected at stored resolution.

- Interior and posterior initializations are broad cloudy volumes/spikes; their final views remain
  soft smeared volumes.
- Raw and full coherent initializations are more actor-surface-like, which is the one qualitative
  improvement over the controls, but contain black unsupported background regions, disconnected
  floaters, and weak/thin structure.
- After the unchanged shared optimizer, both coherent arms form broad gray sheets and radial
  streaks. Actor limbs, fabric folds, and stage edges are erased or duplicated. The complete arm is
  slightly cleaner than raw, but every reporting view still fails the floater/sheet/smear/hole and
  thin-feature criteria.

This visual failure is controlling. Higher full-arm terminal MS-SSIM and lower gradient MAE than
interior do not make the geometry usable, and no average can waive an artifact in one native view.

## Causal interpretation

The evidence supports four bounded conclusions:

1. Packet codec distortion is not the dominant failure; decoded appearance is about 38.6 dB and
   identical across arms.
2. Spatially coherent learned depth is a better construction object than the independent
   high-entropy ray posterior, but coherent raster depth alone is insufficient.
3. Projective support, hard anchors, compatibility-aware WSE, and bounded contraction change the
   raw learned-depth tradeoff at lower final row count, improving SSIM/MS-SSIM/LPIPS but losing
   PSNR/gradient/p99. They do not repair missing global surface topology or optimizer instability.
4. The common realtime-gs density schedule can destroy initially recognizable surface structure;
   the step-600/1,000 curve discontinuities and final sheets make optimizer compatibility a first-
   class requirement for any successor.

The final mechanism statement is an inference from the visual/curve/geometry receipts, not a proof
that VGGT depth is metrically wrong or that every surfel/3DGS optimizer must fail.

## Disposition and successor constraint

Retain `realtime_gs_coherent_depth.py`, its injected-predictor tests, and the four-arm driver only as
default-off reference/negative-control code. Do not integrate it into `pipeline.run_pipeline`,
alter `.sgdp`, change a default, tune thresholds or train longer on this frame, count the public
checkpoint as free, claim commercial usability, or call the path artifact-free compression.

A successor must jointly own surface topology and optimization stability, not only per-view depth.
A fair next kill-test would compare an explicit cross-view surface/mesh or globally regularized
point-map representation that is rendered without unconstrained early densification, then allow
topology growth only after held-out residual and silhouette checks. It must use a new disjoint scene,
retain raw learned-depth and interior controls, and fail on any native sheet, streak, floater, hole,
or thin-feature deletion.

## Limitations

This is one exposed frame, one requested seed, reduced resolution, no masks, no physical depth, and
dirty external-source provenance. Repeated replays expose nondeterministic CUDA/density trajectories,
so the nominal seed does not provide terminal bit reproducibility. Reporting cameras are held out
from construction but not a sealed benchmark. LPIPS is evaluated at reduced stored resolution,
host peak memory and network/checkpoint download latency are not measured, and no distinct
prospective or results reviewer exists. The result can reject this exact composition on this scene;
it cannot rank all learned MVS, mesh, point-map, or surface-optimization alternatives.
