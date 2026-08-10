# CORE-018 occlusion-aware ray-posterior disjoint diagnostic

## Scope

This note binds the ignored immutable CORE-018 bundle to its tracked task and to the CORE-016/017
results audits. It is one scene (`karate/frame_00060`), one seed, packet inputs at
calibrated/undistorted downscale 4, and common training/reporting targets at downscale 8. The frame
is disjoint from the `frame_00008`/`frame_00009` stage captures consumed by CORE-016/017, but it is
still exposed development data in a single capture group: it is not sealed confirmation and cannot
satisfy BENCH-019. No realtime-gs source, maintained StructSplat path, packet grammar, renderer,
optimizer equation, or default changed.

The note is written after the fact, during the CORE-018/019 ledger reconciliation of 2026-08-08. It
records an existing negative result; it does not re-run, re-tune, or rescue it.

## Method

Twenty-eight construction cameras build one byte-identical shared packet set
(`core018.shared_packets.v1`, 1,360,834 complete packet bytes over 28 views, 1.853 s encode);
`C0004`, `C0025`, `C1004`, and `C1005` are reporting-only and excluded from packet construction and
training. All arms cold-reload the same packets and run the unchanged shared realtime-gs
optimizer/density schedule: 10,000 initial rows, 500 fixed-topology steps, then density through step
1,500 under a 30,000-row cap, seed 0, RTX 3050, torch 2.9.0/CUDA 12.8.

The candidate replaces CORE-017's alpha-support placement with an independently scored ray depth
posterior over source-excluded DINOv2 (`dinov2_vits14`, checkpoint SHA-256 `b938bf1b…`, 88,283,115
bytes) plus local coarse/fine sampling. Three arms were planned:

1. `interior` — ordinary CompactCarve interior consensus (control).
2. `posterior_no_reciprocal` — ray posterior with `min_primary_fraction = 1.0`, reciprocal check off.
3. `posterior_reciprocal` — the complete candidate, frozen floor `min_primary_fraction = 0.75`,
   `min_reciprocal_views = 1`.

## Result

**The complete arm never emitted a Gaussian.** `posterior_reciprocal` is a persisted error cell:
`ValueError: ray-posterior selected primary fraction falls below min_primary_fraction`. Its frozen
0.75 support floor was not met, and the floor was not lowered. The lift diagnostics explain why:
candidate posterior entropy is near-maximal (median 0.95960, mean 0.94195, p10 0.88193, p90 0.99122
on a 0–1 scale) and reciprocal support is sparse (median 0.0, mean 0.53472, p90 2.0 of a maximum 4).
Independent per-ray scoring does not concentrate depth on this scene.

Reporting aggregates for the two arms that ran (four held-out cameras, downscale 8):

| step | arm | PSNR (dB) | MS-SSIM | LPIPS | rows |
|---|---|---|---|---|---|
| 0 | interior | 8.7276 | 0.0789 | 0.7522 | 10,000 |
| 0 | posterior_no_reciprocal | 10.5739 | 0.1467 | 0.8312 | 10,000 |
| 500 | interior | 14.3380 | 0.4472 | 0.6918 | 10,000 |
| 500 | posterior_no_reciprocal | 13.4921 | 0.4168 | 0.7332 | 10,000 |
| 1500 | interior | 14.0120 | 0.3976 | 0.6984 | 29,422 |
| 1500 | posterior_no_reciprocal | 14.1048 | 0.4320 | 0.6693 | 30,000 |

Recomputed deltas (posterior minus interior): step 0 `+1.8463 dB` but LPIPS `+0.0790` worse; step 500
`-0.8459 dB`, `-0.0304` MS-SSIM, `+0.0414` LPIPS — the posterior **loses the fixed-topology prefix
comparison outright**; terminal `+0.0928 dB` only while saturating the 30,000-row cap against
29,422 control rows, at 16.312 s versus 14.719 s native training. Terminal gradient MAE is worse
(0.02247 versus 0.02184) and terminal p99 absolute error is worse (0.64339 versus 0.63444).
Conservative `original/(packets + final model)` is 1.9854 for the posterior arm and 2.0177 for the
control against 15,741,328 source bytes.

## Visual and decision disposition

The bundle's own decision record is `advance = false`, `scalar_pass = false`, reason
`"one or more arms failed"`, with `manual_visual_review_required = true`. Native review of the
no-reciprocal arm finds a translucent smeared volume rather than a surface: the geometry is not
occlusion-resolved, and its small terminal PSNR lead is bought with the row cap and extra time after
losing the fixed-topology prefix. The route is rejected without threshold rescue. No support floor,
temperature, dustbin cost, feature weight, or reporting view was retuned on `frame_00060` after
outcomes were visible.

## Binding

The immutable bundle is `results/core018_ray_posterior_karate_frame00060_2026-08-06_v1/`
(git-ignored); `manifest.json` SHA-256 is
`e11c4a73e8a94afbf149e52b9a1acc889bf22a7beb2ec4bd89c27ac36f8d0610`, recomputed on 2026-08-08. Its
manifest records `status: partial`, `scope:
single_disjoint_scene_single_seed_reduced_resolution_diagnostic`. Calibration is
`karate/calibration_dome.json`, 49,655 bytes, SHA-256 `dfac34e5581cca9395b6858e129199a85c03a5f2e63f3dc949e5b59e5c9b6e3b`.
Every number above was recomputed from `metrics.json`, `partial_records.json`, `plan.json`,
`shared_packets.json`, and `decision.json` rather than copied from task prose. The bundle uses a
task-local schema and is not a maintained portable report; `scripts/check_report_bundle.py` is not
the gate for it.

## Limitations

One scene, one capture group, one seed, reduced resolution, dirty external source with snapshotted
executed sources, no physical geometry truth, non-bit-reproducible CUDA density, and provisional
self-review only. No general quality, convergence, speed, geometry, compression, artifact-freedom,
or downstream-surrogate claim follows.

## Reproduction

```bash
PYTHONPATH=/home/alex/Documents/structsplat/src:/home/alex/Documents/realtime-gs/src \
  /home/alex/Documents/realtime-gs/.venv/bin/python \
  scripts/experiments/core018_ray_posterior_downstream.py \
  --out results/core018_ray_posterior_karate_frame00060_2026-08-06_v1
```

The driver pins `karate/frame_00060` internally; there is no `--frame` flag. Executed StructSplat
revision `998e48e942705431f4685213d4da341df1cef55f` and realtime-gs revision
`36630c7fef14c0907134d2f3c532be3da4a0c43e`, both dirty, with exact used sources copied into the
bundle. See `docs/research/2026-08-06-core018-ray-posterior-results-audit.md` for the full integrity
and rate ledger.
