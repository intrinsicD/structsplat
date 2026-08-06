# CORE-018 ray-posterior results audit

Date: 2026-08-06

Disposition: **rejected; preserve as default-off negative evidence**

Scope: one disjoint, reduced-resolution, unmasked development scene; no promotion claim

## Executive result

CORE-018 does not produce a usable geometry pipeline. The complete reciprocal candidate fails
closed before optimization because fewer than the prospectively required 75% of its selected rows
have reciprocal support. The two arms that do construct a field both fail native visual review:
their held-out renders are translucent volumetric smears with unstable depth support, not coherent
surfaces. Lowering the reciprocal threshold would admit precisely the unsupported candidates the
gate was designed to reject and is not an authorized rescue.

The posterior without reciprocity is still informative. Relative to ordinary CompactCarve
interior consensus, it starts 1.8463 dB higher and lifts faster, but its advantage disappears during
the fixed-topology prefix. At step 500 it is 0.8459 dB worse, and at the terminal checkpoint it is
only 0.0928 dB better while gradient MAE is worse. Its candidate posterior is highly ambiguous:
median normalized entropy is 0.9596, mean selected confidence is 0.04149, and median reciprocal
support is zero. Independent source-ray scoring is therefore rejected as the geometry unit on this
scene.

## Frozen protocol and provenance

- Scene: `/home/alex/Dropbox/Work/Janelle/karate/frame_00060`, sibling
  `calibration_dome.json`.
- Construction cameras: 28 cameras, all calibrated views except `C0004`, `C0025`, `C1004`, and
  `C1005`.
- Reporting cameras: the four excluded cameras above (`rgb_4`, `rgb_25`, `rgb_1004`, `rgb_1005`).
- Packet inputs: one full-canvas WebP-quality-80 `.sgdp` packet per construction camera, exact full
  alpha, 1,024 structural proposals, sigma `0.45`, radius `3`, eight Jacobi steps, and a distinct
  deterministic view seed.
- Geometry arms: ordinary interior consensus plus surface cover; packet-feature posterior plus
  surface cover; and the same posterior with reciprocal consistency.
- Common output/training: 10,000 initial rows, 500-step fixed-topology checkpoint boundary,
  1,500 total attempted steps, classic density events from step 600 through 1,400, 30,000-row cap,
  seed 0, SH degree 3, no masks, no random background.
- Packet features: DINOv2-S/14 plus a packet-derived local chroma/gradient descriptor. The cached
  checkpoint is 88,283,115 bytes with SHA-256
  `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`; its cached repository
  declares Apache-2.0. Features were derived from cold packet appearance and alpha, and the receipt
  records that source RGB was not opened.
- Executed StructSplat revision: `998e48e942705431f4685213d4da341df1cef55f`, dirty with the
  task implementation and unrelated IntelliJ files. Executed realtime-gs revision:
  `36630c7fef14c0907134d2f3c532be3da4a0c43e`, also dirty. Exact used source files are copied into
  the result.
- Hardware/runtime: RTX 3050 8 GB, torch 2.9.0/CUDA 12.8, Python 3.12.9.

Executed command:

```bash
PYTHONPATH=/home/alex/Documents/structsplat/src:/home/alex/Documents/realtime-gs/src \
  /home/alex/Documents/realtime-gs/.venv/bin/python \
  scripts/experiments/core018_ray_posterior_downstream.py \
  --out results/core018_ray_posterior_karate_frame00060_2026-08-06_v1
```

No algorithm parameter was changed after this outcome, and the result directory was not
overwritten.

## Integrity and rate ledger

The immutable local result is
`results/core018_ray_posterior_karate_frame00060_2026-08-06_v1/` (73 MiB). Its main hashes are:

| Artifact | SHA-256 |
|---|---|
| `manifest.json` | `e11c4a73e8a94afbf149e52b9a1acc889bf22a7beb2ec4bd89c27ac36f8d0610` |
| `plan.json` | `e7bc7e132908b18bf30a0e72fe8ebdb5ae8f5883e4bb2fbe2d99401eb0a82429` |
| `metrics.json` | `d5abc9b408d5fb2d639ba4401de1a7a0de2d84c64179e2cbb75a78331b75b087` |
| `decision.json` | `3f6119220e773f12e55a2e8b1d82ec7c7d44578868f7ebc39876d99f839a7d1f` |
| `all_metric_curves.png` | `e623901c5ef94d44a8a77e070892d116518cc5257f7f6894799499043e5767b9` |

The run status is `partial`, its total wall clock is 80.86 seconds, and the decision is
`scalar_pass=false`, `advance=false`. All 28 packets total 1,360,834 bytes versus 15,741,328 bytes
for their original JPEG files, or 11.567x original/packet. Mean decoded packet PSNR is 38.581 dB
(37.719--39.422 dB). The shared DINO weights are an 88,283,115-byte amortized dependency and are
not silently counted as per-scene packet bytes.

| Arm | Status | Final N | Final NPZ bytes | Original/(packet + model) |
|---|---|---:|---:|---:|
| Interior consensus | ok | 29,422 | 6,440,830 | 2.0177x |
| Posterior, no reciprocity | ok | 30,000 | 6,567,692 | 1.9854x |
| Posterior + reciprocity | error | — | — | — |

The rate ratios are physical file ratios, but they do not establish a useful compression result:
the renders are unusable, source/model repositories were dirty, and the shared feature weights and
their amortization break-even are reported separately.

The driver emits a task-local report rather than a maintained StructSplat workflow bundle.
`python scripts/check_report_bundle.py RESULTS_DIR` therefore reports 42 schema-contract problems;
with the error cell and dirty revisions, the result is diagnostic even under a compatible schema.
The internally referenced packet/model/visual files and hashes remain available for audit.

## Quantitative outcome

Held-out aggregates at stored downscale-8 pixels:

| Metric | Interior init | Posterior init | Interior final | Posterior final | Final delta |
|---|---:|---:|---:|---:|---:|
| PSNR (dB) | 8.7276 | 10.5739 | 14.0120 | 14.1048 | +0.0928 |
| SSIM | 0.13064 | 0.25426 | 0.50698 | 0.51995 | +0.01297 |
| MS-SSIM | 0.07892 | 0.14669 | 0.39757 | 0.43199 | +0.03442 |
| LPIPS | 0.75222 | 0.83119 | 0.69835 | 0.66933 | -0.02903 |
| Gradient MAE | 0.02337 | 0.02615 | 0.02184 | 0.02247 | +0.00063 |
| P99 absolute error | 0.99381 | 0.86183 | 0.63444 | 0.64339 | +0.00894 |

The initial PSNR/MS-SSIM gain does not mean the initialization is optimization-compatible. By step
500, before topology growth, interior consensus reaches 14.3380 dB and 0.44724 MS-SSIM while the
posterior reaches 13.4921 dB and 0.41678. The posterior reaches the interior arm's terminal 14.0120
dB only at step 1,200; the interior arm already exceeds that value at step 500. Native training
time is 16.31 versus 14.72 seconds.

Pretraining does favor the cheaper reference posterior implementation: shared features plus lift
and cold adapter total 6.09 seconds versus 9.24 for the interior arm, with posterior placement
itself taking 0.810 seconds versus 6.235. This speed result cannot promote a method whose geometry
and visual result fail.

Density events are not monotone refinement. Interior drops 5.316 dB from step 500 to 600, while the
posterior drops 2.319 dB. The step-1,000 opacity reset drops 4.062 dB and 6.332 dB respectively
relative to step 900. Both later recover, but these discontinuities and the terminal-count mismatch
make the last-checkpoint delta weak convergence evidence.

## Posterior and reciprocal diagnosis

The no-reciprocal arm proposes 20,000 candidates. Of these, 16,029 have the required two real
source-excluded observations; balanced selection fills all 10,000 rows without fallback. However:

- normalized entropy is 0.94195 on average and 0.95960 at the median;
- likelihood margin is 0.01333 on average and 0.00866 at the median;
- selected confidence is 0.04149 on average and 0.03036 at the median;
- reciprocal support has mean 0.5347 and median 0; and
- selected depths span 1.061--4.098 with median 2.403 despite no ground-truth depth evidence.

The complete arm requires at least one reciprocal neighbor and at least 75% primary selected rows.
It cannot meet that requirement and raises before radiance/covariance construction. Because the v1
exception text did not retain the exact count, this audit states only the proven bound: its selected
primary fraction is below 0.75. The implementation now includes the count/fraction in future error
messages without changing selection behavior.

## Native visual review

The interior contact sheet has SHA-256
`5ea97ffdd97140690771b72b9831fbf0cc62cabbba21b384530eddcdf2690347`; the no-reciprocal
posterior sheet has SHA-256
`5ffc408ce391ae71d8ad8672d50781bbda31b7cb4155d9c044e1b64f0d5c3fed`.

All four reporting rows show the same disqualifying pattern. Initial fields are broad, translucent
point clouds or volumes. Final RGB is a soft purple/gray smear with duplicated directional texture;
the actor and dome structure do not form coherent surfaces. Alpha is nearly opaque across large
unsupported regions, and the depth panels contain broad radial bands and disconnected modes. The
posterior changes the location and granularity of these artifacts but does not remove floaters,
trails, or blur. The complete reciprocal arm has no visual because it correctly refused to emit a
field.

This is a direct visual failure, not an aesthetic preference that can be outweighed by aggregate
PSNR. The method is not usable by realtime-gs for the requested high-quality rendering.

## Causal interpretation

The evidence supports three bounded conclusions:

1. Packet appearance is sufficiently faithful for this diagnostic (about 38.6 dB decoded PSNR),
   so input codec distortion is not the dominant cause of the catastrophic geometry.
2. Source-excluded matching can improve a raw step-zero scalar and eliminate expensive structural
   index queries, but independently solved rays remain too ambiguous and spatially incoherent.
3. Reciprocal filtering detects that incoherence rather than fixing it. Relaxing the detector would
   relabel unsupported geometry as accepted geometry.

The likely mechanism is that low-resolution semantic/detail matching has no spatial surface model:
repeated dome structure, textureless regions, view-dependent appearance, and occlusion can each
produce shallow local modes. This is an inference from the entropy/support diagnostics, not a
ground-truth depth finding.

## Disposition and next constraint

Retain `realtime_gs_ray_posterior.py` only as a default-off reference and negative control. Do not
integrate it into `pipeline.run_pipeline`, alter `.sgdp`, change a default, run longer on this
scene, lower the reciprocal threshold, or claim general compression/speed/quality.

A successor must change the geometry unit from independent rays to a spatially coherent depth or
surface explanation. Mature dense-MVS/learned point-map geometry, or a global edge-aware cost-volume
method, should be tested as the direct control before inventing another local score. It must keep
reporting cameras excluded, compare against both interior and this no-reciprocal posterior, expose
uncertainty/consistency before emission, and fail on any native floater/trail. This consumed scene
may remain a regression only; method selection and tuning require a new disjoint development scene.

## Limitations

This is one frame, one seed, reduced resolution, no masks, no physical depth, and a dirty-source
diagnostic. The task-local schema is not accepted by the maintained report checker, host peak memory
was not measured, the reciprocal failure did not retain its exact support count, and no independent
prospective or results review exists. The result can reject this configuration on this scene; it
cannot rank every plane-sweep, learned feature, or spatial-consistency method.
