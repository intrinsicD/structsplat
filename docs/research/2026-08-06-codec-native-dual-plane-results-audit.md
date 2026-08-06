# CORE-016 codec-native dual-plane results audit

Audit date: 2026-08-06. Disposition: diagnostic plumbing confirmed; scientific and production
claims narrowed. No claim-ledger promotion or default change is authorized.

## Audited artifact

- Development input: exposed Janelle `frame_00008/rgb/C0001.jpg` plus `mask_C0001.png`.
- Selected ignored bundle:
  `results/core016_prefiltered_sigma045_steps8_selected_janelle_native_2026-08-06/`.
- Packet schema: `structsplat.codec_native_dual_plane.v2` (`.sgdp`).
- Selected configuration: lossless WebP effort 75, Gaussian lattice sigma `0.45` px, radius `3`,
  eight prefilter iterations, 512 structural rows, seed 0.
- Hardware/software: NVIDIA RTX 3050 for CUDA compatibility timing; PyTorch 2.9.0+cu128. The packet
  producer itself is NumPy/Pillow and CPU-bound.

The selected setting is post-hoc development data. The artifact is not a sealed confirmation run.

## Independent accounting and replay checks

The packet is 3,896,344 physical bytes. Recomputed ZIP compressed-member accounting is:

| component | bytes |
|---|---:|
| appearance payload | 3,871,384 |
| structural payload | 24,034 |
| compressed manifest | 594 |
| ZIP framing | 332 |
| complete packet | 3,896,344 |

The four components sum exactly. All 25 diagnostic-manifest file sizes and SHA-256 values were
recomputed successfully. A cold load followed by a second save is byte-for-byte identical and has
the same ledger. Focused tests cover duplicate/unknown members, payload corruption, dimension and
codec mismatch, exact transforms, alpha, deterministic structure, and optional dependency imports.

`scripts/check_report_bundle.py` does **not** accept this task-local custom diagnostic bundle. It
reports the workflow schema, executed-command, repository-identity, and metrics-row shape as
unrecognized. That is a packaging limitation, not a passed maintained-report gate. The bundle has
an internally verified custom manifest; it must not be called a portable maintained report.

## Recomputed result row

| axis | value | audit interpretation |
|---|---:|---|
| source JPEG bytes / packet | `3.66195x` | Narrow: source is a 5328x4608 full frame while packet stores a 3417x903 crop |
| canonical crop PNG / packet | `1.13896x` | Fairer crop-local context; still PNG-specific |
| bits / active mask pixel | `18.0431` | Complete packet bytes, 1,727,574 active pixels |
| bits / crop pixel | `10.1022` | Complete packet bytes, 3,085,551 crop pixels |
| masked pixel-center PSNR | capped `120 dB` | Reconstruction is below float/display relevance at sampled centers |
| raw foreground max absolute error | `4.77e-7` | Confirmed only for selected decoded crop and implementation |
| displayed worst pixel / 7x7 patch | `0 / 0` | Confirms pixel-center display, not off-grid artifact freedom |
| MS-SSIM | `1.0` | Same narrow pixel-center domain; LPIPS not run at native resolution |
| component-summed encode estimate | `3.286 s` | Not a separately timed monolithic production pipeline |
| cold decode | `1.316 s` | Includes derived coefficient construction in reference path |
| NumPy full crop render | `6.835 s` | Reference evaluator, not realtime renderer FPS |
| NumPy appearance query median | about `364k points/s` | 20k-point repeated reference query |
| RTGS CUDA query median | `0.821 ms / 256`, about `312k points/s` | Small-batch adapter/index measurement; not full rendering |
| NumPy/RTGS color parity max | `1.19e-7` | Interface plumbing confirmed |
| NumPy/RTGS structural-weight parity max | `1.43e-6` | Interface plumbing confirmed |
| full-canvas float32 coordinate roundtrip max | `1.81e-4 px` | Reported separately from field parity |
| sampled structural coverage | `1.0` | Sampling diagnostic only; nearest-center q99 is `79.18 px` |

The historical `.rtgsv` is 167,765 bytes and the CORE-016 packet is 23.23 times larger. That
`.rtgsv` uses calibrated/undistorted/different crop processing and much lower image fidelity, so it
is contextual only. The HIER-009 8,192-row control uses an uncoded/lossless NPZ proxy and a resized
image; it is likewise not a rate-matched codec comparator.

## Continuous-query guard

Deep-alpha off-grid samples are compared to bilinear interpolation of the decoded raster:

| metric | selected value |
|---|---:|
| bilinear-control PSNR | `49.3754 dB` |
| q99 per-sample RGB RMSE | `0.01261` |
| maximum absolute channel difference | `0.11711` |
| finite-difference gradient RMS ratio | `1.16359` |
| channels outside local 2x2 sample envelope | `3.784%` |
| samples outside global `[0,1]` | `0.0244%` |
| sampled coefficient-field range | `-0.00510..1.00836` |

This guard exposes rather than resolves cardinal ringing. Bilinear interpolation is not continuous
scene ground truth, and the sample is not a proof over the domain. Therefore “no visual artifacts”
is valid only for the decoded pixel-center display inspected in this artifact.

## Claim disposition

| Candidate statement | Disposition | Reason |
|---|---|---|
| Strict packet grammar, deterministic bytes and exact byte ledger work | Confirmed as implementation plumbing | Unit tests plus independent replay/accounting |
| Selected C0001 pixel-center display is artifact-free | Confirmed narrowly | Full/crop/error images inspected; display error is zero |
| Packet compresses the original by `3.66x` | Narrowed | Full-frame source versus crop packet; crop-PNG ratio is `1.139x` |
| Method converges faster | Narrowed | Zero optimizer steps and 3.29-second component sum versus incompatible 315-second context |
| Packet is usable by realtime-gs | Confirmed at interface/smoke level | Query parity and deterministic synthetic two-view CompactCarve lift |
| Continuous appearance is high quality | Inconclusive | No continuous truth; measured ringing and envelope escape remain |
| 512 structural rows are sufficient | Unsupported | Coverage sampling is not downstream multiview fidelity |
| Method is generally high quality/compressed/SOTA | Unsupported | One exposed image, post-hoc choice, no held-out or matched baselines |
| Maintained report gate passes | False | Custom schema is not accepted by `check_report_bundle.py` |

## Integrity and leakage audit

- The source and mask are exposed development data; no confirmation split was accessed.
- Sigma and prefilter steps were selected after inspecting development sweeps. The selected bundle
  is therefore evidence of feasibility, not an unbiased estimate.
- Failed and superseded result directories were retained rather than overwritten.
- No realtime-gs source file was edited; the existing checkout was used read-only through its
  public backend and initializer interfaces.
- Appearance bytes, structure bytes, manifest bytes and container framing are all charged. Derived
  coefficient memory/compute is not a hidden byte payload, but must be included in cold cost.
- No public claim ledger entry was added.

## Reproduction

Focused unit and compatibility checks:

```bash
PYTHONPATH=/home/alex/Documents/structsplat/src:/home/alex/Documents/realtime-gs/src \
  pytest -q tests/test_codec_native_field.py
```

The exact diagnostic command and resolved configuration are preserved in the bundle's
`config.json`; source snapshots and generic numeric curves are referenced by its `manifest.json`.
Before any result is promoted, freeze a held-out full-frame/multiview protocol and use a maintained
report schema or add an explicit checker for this diagnostic schema.
