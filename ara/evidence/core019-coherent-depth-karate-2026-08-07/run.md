# CORE-019 calibrated coherent-depth exposed diagnostic

## Scope

This note binds the ignored immutable CORE-019 v5 bundle to its tracked task and results audit. It
is one scene (`karate/frame_00005`), one seed, packet inputs at calibrated/undistorted downscale 4,
and common training/reporting targets at downscale 8. `frame_00005` is a development frame in the
same single capture group as every other consumed Janelle frame: it is exposed, not sealed
confirmation, and cannot satisfy BENCH-019. No realtime-gs source, maintained StructSplat path,
packet grammar, renderer, optimizer equation, or default changed.

The note is written after the fact, during the CORE-018/019 ledger reconciliation of 2026-08-08. It
records an existing negative result; it does not re-run, re-tune, or rescue it.

## Method

Twenty-six construction cameras build one byte-identical shared packet set
(`core019.shared_packets.v1`, 1,256,406 complete packet bytes over 26 views, 1.586 s encode);
`C0024`, `C0010`, `C1004`, and `C0022` were selected from calibration geometry before the frame was
opened and are reporting-only. All four arms cold-reload the same packet-hash vector and run the
unchanged shared realtime-gs optimizer/density schedule: 10,000 initial rows, SH degree 3, 500
fixed-topology steps, density from step 600 through step 1,500 under a 30,000-row cap, seed 0,
RTX 3050.

The candidate replaces CORE-018's independent ray scoring with a pinned VGGT
(`facebook/VGGT-1B`, checkpoint 5,026,367,224 bytes, SHA-256 `f164acf6…`, CC-BY-NC-4.0) predicting
coherent depth over deterministic four-view calibration-only groups. One Sim(3) per group supplies
**scale only**; known calibrated rays own back-projection. Overlapping views fuse by
confidence-weighted robust median with MAD uncertainty. Acceptance separates projective support,
compatible occlusion, free-space contradiction, and invalid evidence; selection adds hard feature
anchors, dynamic WSE to exactly 10,000, and bounded post-selection contraction.

Four arms: `interior` (CompactCarve consensus control), `posterior_no_reciprocal` (CORE-018's
completed negative control), `vggt_raw_known_ray` (calibrated depths, no support/contraction/WSE),
and `vggt_coherent_wse` (the complete candidate). The raw/full pair is the mechanism ablation.

## Result

Reporting aggregates on the four held-out cameras (downscale 8), recomputed from `partial_records.json`:

| step | interior | posterior_no_recip | vggt_raw | vggt_coherent |
|---|---|---|---|---|
| 0 PSNR | 7.6551 | 9.4768 | 6.6814 | **6.6860** |
| 500 PSNR | 13.5058 | 11.6146 | 11.2960 | **10.6497** |
| 1500 PSNR | 12.8465 | 12.7199 | 12.2256 | **11.9181** |
| 1500 MS-SSIM | 0.3005 | 0.3101 | 0.3503 | **0.3605** |
| 1500 LPIPS | 0.7409 | 0.7295 | 0.7800 | **0.7594** |
| 1500 rows | 25,117 | 30,000 | 15,680 | **14,776** |

**Every frozen advancement gate that involves a control fails.** At step 0 the candidate is
`-0.9691 dB` below interior against a required `+2 dB`, and its LPIPS is worse. At step 500 it is
`-2.8561 dB` below the strongest control (interior) against a `0.1 dB` tolerance, and fails the
MS-SSIM, LPIPS, and gradient-MAE comparisons. It never reaches interior's terminal PSNR — the
bundle's `convergence_to_control_terminal` records `candidate_step: null` against the control's
step 400 / 2.680 s — and ends `-0.9284 dB` below it.

The full-versus-raw mechanism ablation is genuinely mixed, and the v5 correction was to *count* the
prespecified metrics it had previously ignored, not to move a threshold: full gains `+0.010225`
MS-SSIM, `+0.001968` SSIM, `-0.020617` LPIPS, and a better spacing tail
(p90/median 1.8726 versus 1.9560) at **904 fewer final rows**, while losing `-0.30749 dB` PSNR and
worsening gradient MAE (`+0.0000793`), p99 absolute error (`+0.05739`), and training time. The
mechanism changes the tradeoff; it does not produce a uniform win.

Byte accounting against 14,557,266 source JPEG bytes: conservative
`original/(packets + final model)` is 3.2476 for the candidate, 3.1264 for raw, and 2.1598 for
interior — physically greater than 1, and explicitly **not** a compression claim. The
5,026,367,224-byte CC-BY-NC-4.0 encoder is recorded separately as amortized dependency state
(`encoder_checkpoint_bytes_separate`) and is never counted as per-scene payload.

## Visual and decision disposition

The bundle's decision record is `advance = false`, `scalar_pass = false`,
`manual_visual_review_required = true`, `strongest_step500_control` and `strongest_terminal_control`
both `interior`. Native inspection of every coherent reporting view finds black holes, floaters,
broad gray sheets, radial streaks, and erased thin detail. The visual gate is mandatory and controls
even though `terminal_pareto_nondominated` and the byte-ratio gate pass. No threshold was rescued on
this scene, and the composition is retired rather than escalated.

## Binding

The immutable bundle is `results/core019_coherent_depth_karate_frame00005_2026-08-07_v5/`
(git-ignored); `manifest.json` SHA-256 is
`d196b10fc011a436c2b0b0f8b6fec610c7ac9f53c4906a19aa21b52ece0a5af2`, recomputed on 2026-08-08. The
manifest records `status: ok`, `claim_ready: false`, `scope:
single_exposed_development_scene_single_seed_reduced_resolution_diagnostic`, and 142.567 total wall
seconds. `python scripts/check_report_bundle.py --allow-dirty RESULTS_DIR` passes; `--allow-dirty`
is a disclosure, not a waiver. Every number above was recomputed from `metrics.json`,
`partial_records.json`, `plan.json`, `shared_packets.json`, `manifest.json`, and `decision.json`
rather than copied from task prose.

Executed v5 StructSplat revision `722696c893e4a37cabb69ab24dcf5fcd5d9efb30` (method, driver,
checker, and gate logic committed; only unrelated IntelliJ files dirty); realtime-gs revision
`36630c7fef14c0907134d2f3c532be3da4a0c43e`, dirty; VGGT source revision
`a288dd0f14786c93483e45524328726ab7b1b4ce`, checkpoint revision
`860abec7937da0a4c03c41d3c269c366e82abdf9`. See
`docs/research/2026-08-07-core019-coherent-depth-results-audit.md` for the full ledger.

## Limitations

One exposed scene, one capture group, one seed, reduced resolution, dirty external source, no
physical depth truth, and provisional self-review only. The v3/v4/v5 terminal full-versus-raw PSNR
deltas are `+0.371 / -0.081 / -0.307 dB` across matching construction: the shared CUDA/density path
is **not bit reproducible**, so single-run terminal deltas are fragile and only the step-zero,
step-500, convergence, and visual failures — which are stable across all three replays — carry the
conclusion. No general quality, convergence, speed, geometry, compression, artifact-freedom, or
downstream-surrogate claim follows.

## Reproduction

```bash
PYTHONPATH=/home/alex/Documents/vggt:/home/alex/Documents/realtime-gs/src:src \
  /home/alex/Documents/realtime-gs/.venv/bin/python \
  scripts/experiments/core019_coherent_depth_downstream.py \
  --frame /home/alex/Dropbox/Work/Janelle/karate/frame_00005 \
  --weights /home/alex/.cache/huggingface/hub/models--facebook--VGGT-1B/blobs/f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e.repairing-20260807 \
  --out results/core019_coherent_depth_karate_frame00005_2026-08-07_v5
```
