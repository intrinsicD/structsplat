# CORE-016 codec-native dual-plane results audit

Audit date: 2026-08-06. Disposition: diagnostic plumbing and exposed multiview development utility
confirmed; scientific and production claims narrowed. No claim-ledger promotion or default change
is authorized.

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

## Exposed multiview downstream extension

Five immutable follow-ups propagate the required structural-field/appearance-backend pair through
the current realtime-gs CompactCarve and 3DGS trainer. V1 uses eight training and three reporting-
only cameras. V2 changes only density control. V3--V6 use all 23 non-reporting cameras for packet
construction and training, retaining `C0004`, `C0025`, and `C1004` as reporting-only. Packet inputs
are calibrated/undistorted at downscale 4; common source targets and reporting renders are
calibrated/undistorted at downscale 8. The reporting cameras were inspected after every run and
then reused for post-hoc development, so none is confirmation evidence.

### Progression and retained point

| run | decisive outcome |
|---|---|
| v1 fixed 835 | Candidate quality exceeds control but all arms are visibly unacceptable; the 2,048-row safety arm is only `2.304x` smaller, failing its 3x rate gate. |
| v2 8-view density | 512-row candidate reaches 22.403 dB, but misses 22.5 dB / 0.90 alpha floors and finishes with 7,688 versus 5,631 Gaussians. |
| v3 23-view density | Candidate reaches 25.203 dB and 0.9483 alpha at `4.0266x` lower input bytes, but 11,689 versus 10,022 Gaussians fails the 1.10x count gate. |
| **v4 matched 10k** | **Both finish at exactly 10,000; every scalar gate passes. This is the retained development Pareto point.** |
| v5 strong mask | Alpha rises to 0.9604, but candidate loses 0.446 dB and worsens gradient MAE versus v4; frozen cross-run preference fails. |
| v6 late mask polish | Alpha rises to 0.9548, but candidate loses 0.316 dB and worsens gradient MAE versus v4; frozen cross-run preference fails. |

The executed v5/v6 custom reports say `scalar_pass: true` because their original driver encoded
only within-run control gates. Independent audit replay applies the separately frozen cross-run v4
retention gates and returns **false** for both PSNR and gradient-MAE guards. The driver is corrected
for future execution; immutable bundles were not rewritten.

V4 terminal rows are:

| metric | RTGSV control | dual-plane q92 / 512 | candidate minus control |
|---|---:|---:|---:|
| complete 23-view input bytes | 3,850,647 | 956,301 | `4.0266x` smaller |
| final 3D Gaussians | 10,000 | 10,000 | 0 |
| reporting foreground PSNR | 24.0119 dB | 25.1880 dB | +1.1761 dB |
| reporting crop PSNR | 30.0384 dB | 31.5712 dB | +1.5329 dB |
| reporting MS-SSIM | 0.960839 | 0.967342 | +0.006504 |
| reporting LPIPS | 0.099172 | 0.079733 | -0.019439 |
| reporting gradient MAE | 0.014792 | 0.012926 | -0.001865 |
| reporting alpha IoU | 0.928680 | 0.949653 | +0.020972 |
| lift time | 4.894 s | 9.467 s | +4.573 s |
| native full training time | 18.521 s | 20.740 s | +2.219 s |
| first time at control-final PSNR | step 1,400 / 13.266 s | step 500 / 5.148 s | candidate reaches the lower target earlier |
| peak VRAM | 0.310 GiB | 0.366 GiB | +0.055 GiB |

The time-to-target row supports only faster convergence to the control's lower terminal reporting
quality within this run. Candidate lift, full native training, available pipeline time, and VRAM
are worse. Candidate available-pipeline time includes packet construction, whereas production time
for the pre-existing RTGSV containers is unavailable; no end-to-end speed ratio is valid.

The candidate packet ledger is 592,784 appearance bytes, 342,484 structure bytes, 13,397 compressed
manifest bytes, and 7,636 ZIP-framing bytes. The 956,301-byte sum is `1.302` bits per downscale-4
crop pixel and `3.440` bits per active crop pixel. The contextual canonical crop-PNG sum is
6,964,513 bytes (`7.283x` larger); the 332,472,461-byte original-JPEG sum is scope-mismatched because
the packets store cropped, downscaled, calibrated tensors. Neither ratio is a full-resolution image
codec claim. Final candidate/control NPZ and PLY models are essentially the same size at equal count,
so the input-teacher rate win is not a final-3D storage-compression result.

### Native-pixel visual disposition

V1 fixed topology has severe blur, long rays/streaks, bright silhouette halos, and missing thin
anatomy. V2 density control recovers the subject but leaves conspicuous streaks/floaters. Full
angular support in V3/V4 materially improves anatomy and removes any obvious periodic grid or
KD-tree imprint. V4 still shows soft silhouette halos, fine-detail blur, and sparse floaters on the
three reporting views. V5/V6 tighten some silhouettes but do not remove the residual artifacts and
fail their cross-run fidelity guards. The mandatory manual gate therefore fails for every run.

### Independent integrity replay

The audit recursively rehashed and resized every `{path, bytes, sha256}` receipt in each v2--v6
plan, record, and manifest: 108 unique receipts for v2 and 213 for each full-capture run. It reloaded
and byte-identically resaved all 100 candidate packets across those five runs, recomputed every
packet component sum and complete input sum, verified that packet view IDs equal training IDs and
exclude all reporting IDs, independently recreated every evaluation image/mask tensor hash, checked
21 checkpoint rows per arm for v2--v5 and 26 for v6, loaded every final NPZ, and found all arrays
finite with the declared cardinalities. Every reported input ratio and candidate/control delta
recomputes to `1e-12` absolute tolerance.

CUDA density trajectories are not bitwise deterministic despite declared seeds: the nominally
identical v4/v6 base recipes end at nearby but not identical counts/metrics (for example v6 control
has 9,989 rows before fixed-topology polish versus v4's 10,000). This does not invalidate the
single-run rows, but it prevents treating small cross-run differences as deterministic effects.
V5 timing is additionally contaminated by concurrent CPU compilation; its quality rows remain
valid, but its wall time is excluded from speed interpretation.

`scripts/check_report_bundle.py` rejects every multiview bundle with the same four expected custom-
schema errors: wrong workflow schema, no recognized executed-command field, unrecognized repository
identity shape, and unrecognized metrics-row envelope. These are internally audited diagnostic
bundles, not maintained portable reports. V4's immutable `rate_quality.png` also incorrectly labels
the x-axis as an eight-view input; its numeric bytes are correct and later driver output labels the
actual profile view count. The artifact is preserved rather than silently repaired.

## CORE-017 visibility-ordered shell follow-up

CORE-017 tests whether CORE-016's residual halo/floaters originate primarily from CompactCarve's
arbitrary interior-depth selection and broad localization covariance. It uses a new exposed
`frame_00009`, the same 23 training IDs and three reporting-only IDs, one shared 970,310-byte set of
quality-92 WebP/512-structure packets, exactly 5,000 fixed-topology Gaussians per arm, seed 0, and
1,000 common gsplat steps. The 2x2 factorial is `{ordinary interior consensus, first-maximum alpha
shell}` x `{inherited localization covariance, local surface-cover covariance}`. No reporting view
enters packet construction or training, but the split is exposed development data rather than
sealed confirmation.

The placement-only backend analytically reconstructs 0.95 soft coverage inside exact packet alpha,
adds zero packet/index bytes, preserves codec-native colors, and never calls the sparse structural
index. Sparse Field V2 mass still proposes the source rays. Across 960,000 sampled world points,
ordinary placement evaluates 212,517,051 sparse index pairs; shell placement evaluates zero pairs
and performs 22,080,000 alpha/appearance query points. The selected first-maximum depth index has
min/p10/median/p90/max `14/20/22/25/38` among 48 ordered samples. Cover reconciliation preserves
means, SH coefficients, count, and packet bytes exactly.

| metric | interior / inherited | interior / cover | shell / inherited | shell / cover |
|---|---:|---:|---:|---:|
| initial reporting PSNR | 14.0186 | 13.6915 | 18.6018 | 17.6160 |
| final reporting PSNR | 20.7469 | 20.0434 | **22.3342** | 22.1278 |
| final MS-SSIM | 0.901271 | 0.891446 | **0.941373** | 0.938626 |
| final LPIPS | 0.163428 | 0.150969 | 0.131568 | **0.104433** |
| final gradient MAE | 0.016095 | 0.015670 | 0.015708 | **0.014749** |
| final alpha IoU | 0.790721 | 0.774645 | 0.921256 | **0.931089** |
| final alpha outside | 0.021650 | 0.027524 | 0.006991 | **0.005634** |
| lift seconds | 9.420 | 8.951 | **5.810** | 6.246 |
| native training seconds | 9.831 | **9.270** | 9.752 | 10.159 |
| available packet-build + lift + train seconds | 28.020 | 26.990 | **24.331** | 25.174 |
| first step at baseline-terminal PSNR | 1,000 | never | **200** | 500 |
| final Gaussians | 5,000 | 5,000 | 5,000 | 5,000 |

The combined shell/cover arm passes every frozen numerical guard versus interior/inherited:
+1.3809 dB PSNR, +0.14037 alpha IoU, -0.01602 outside alpha, -0.001347 gradient MAE, exact count,
and identical packet bytes. The stronger scalar quality arm is shell/inherited at +1.5873 dB.
Cover alone loses 0.7035 dB and 0.0161 alpha IoU, confirming that covariance repair cannot rescue
volumetric centers by itself.

Mandatory visual review nevertheless fails. Shell placement removes most broad silhouette glow and
scattered floaters, but all three reporting views retain conspicuous directional trailing smear or
double-silhouette structure, especially around the feet/body in `C0004` and `C1004`; fine texture
is still soft. Surface cover sharpens some local detail and improves perceptual/gradient/alpha
metrics, but does not eliminate those errors. The route is therefore not advanced or retuned on
this frame. It establishes a useful causal placement result, not artifact freedom, physical surface
recovery, general convergence/speed, full-resolution compression, or a production pipeline.

The immutable bundle is
`results/core017_visibility_surface_janelle_frame00009_2026-08-06_v1/`, with manifest SHA-256
`bf14b8e8d08609bdf89dd3c4474422a7ea8c0281c45cb81de8ba50e17252be2e`.
Independent replay validates 222 unique path/byte/SHA receipts, the identical per-view packet hashes
in every arm, exact counts, finite metrics, and decision arithmetic. The maintained report checker
returns the same four expected schema-envelope errors as CORE-016; this remains an internally
audited task-local diagnostic.

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
| Candidate is useful in real multiview lifting/training | Confirmed narrowly on exposed reduced-resolution v4 | Better reporting metrics and 4.027x lower complete teacher-input bytes at equal 10k output count |
| Candidate converges faster | Narrowed | Reaches control-final PSNR earlier, but full training/lift/VRAM are worse and only one CUDA run exists |
| Candidate is artifact-free | False | Native review retains halos, fine-detail blur, and sparse floaters |
| Strong alpha weighting fixes artifacts | Rejected for the tested settings | V5/V6 improve alpha but fail frozen v4 PSNR/gradient retention |
| First-maximum alpha-shell placement improves this exposed fixed-5k assay | Confirmed narrowly | +1.587 dB, +0.131 alpha IoU, earlier baseline-target hit, and zero sparse-index depth pairs versus interior/inherited |
| Surface cover alone repairs interior geometry | Rejected for the tested setting | -0.703 dB and -0.016 alpha IoU versus inherited interior covariance |
| Alpha shell plus surface cover is artifact-free | False | Scalar gate passes, but native trailing smear/double silhouettes and blur remain |

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
