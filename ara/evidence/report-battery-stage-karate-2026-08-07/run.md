# Exploratory current-profile report battery: stage_with_fabric (masked) + karate (full-frame)

Status: **exploratory development evidence, no claim.** These are maintained-workflow report
bundles on development-exposed image classes (the stage capture is the Janelle/C0001 source
family; both karate frames were consumed by CORE-018/019 diagnostics). No task froze a protocol
and no prospective reviewer approved one, so nothing here may be promoted, used to change a
default, or cited as a BENCH-017 outcome. The runs characterize the shipped profile on the two
requested datasets and stress the unscreened full-frame arm.

## Execution identity

- Source commit: `282b468e81bfe2f15f88d6069c62af43c296928c` (clean tree; recorded per-bundle in
  each `*.manifest.json`).
- Machine: NVIDIA GeForce RTX 4090 (24 GB), torch 2.7.0+cu126, renderer `cuda`
  (exact owned extension), LPIPS alex enabled.
- Profile: `safe-commit-schedule_2026-07-27.1` via `scripts/benchmark.py` / `scripts/ablation.py`
  (ADR-0025/0027/0028). CUDA atomic accumulation is not bit-reproducible; identity is bound to
  source, seed, device, and versions, not bit-exact replay.
- The installed editable `structsplat` package on this machine points at a *different checkout*
  (`~/Documents/Deeplearning3/external/structsplat`); every command below was executed with
  `PYTHONPATH=/home/alex/Documents/structsplat/src` so this repository's code ran. Three
  pre-existing portable-gate test failures exist in this environment in closed benchmark modules
  (`test_ssp2e_actual_run`, `test_ssp2v_decode_worker`, `test_affine_carrier_core`; torch 2.7
  lacks `_CudaDeviceProperties.pci_bus_id`); the conversion-path tests pass.
- Battery wall-clock: 2026-08-07T14:14 → 2026-08-08T15:21 (+02:00), sequential on one GPU. All
  four bundles pass `scripts/check_report_bundle.py` with no diagnostic allowances.

## Cohort

Deterministic rule (recorded with source SHA-256s in `selection.md`): first 3 rgb files per
frame in `LC_ALL=C` sorted order.

- `2025_03_07_stage_with_fabric` frames 00008/00009, views C0001/C0004/C0005 (5328×4608 JPG)
  with canonical PNG soft masks (threshold >127) → **masked arm**.
- `karate` frames 00005/00060, first 3 `rgb_*` views (2664×2304 JPEG), no masks →
  **full-frame arm** (unscreened per BENCH-017).

Both dataset classes resolve to the same 1200×1038 working resolution at `--max-side 1200`.

## Commands

```
scripts/benchmark.py results/datasets/stage_karate_selection_2026-08-07/rgb_stage \
  runs/report_stage_masked_2026-08-07 --mask-dir .../mask_stage --device cuda:0 \
  --max-side 1200 --seeds 0 1 --lpips --quiet
scripts/benchmark.py .../rgb_karate runs/report_karate_fullframe_2026-08-07 \
  --device cuda:0 --max-side 1200 --seeds 0 1 --lpips --quiet
scripts/ablation.py .../rgb_stage/frame_00008/C0001.jpg runs/report_abl_stage_C0001_2026-08-07 \
  --mask-dir .../mask_stage/frame_00008 --device cuda:0 --max-side 768 --seeds 0 --lpips --quiet
scripts/ablation.py .../rgb_karate/frame_00005/rgb_1000.jpeg \
  runs/report_abl_karate_rgb1000_2026-08-07 --device cuda:0 --max-side 768 --seeds 0 --lpips --quiet
```

Full image-bearing bundles (index.html with all metric curves over attempted steps,
target/reconstruction/absolute-error PNGs, snapshots, per-cell `config.json`/`history.json`/
`field.npz`) remain local under `runs/` (gitignored); this directory preserves the raw metric
tables and manifests.

## Results — stage masked benchmark (12/12 ok, seeds {0,1})

| view | seed-mean PSNR | MS-SSIM | LPIPS | N | total s |
|---|---:|---:|---:|---:|---:|
| 00008/C0001 | 26.33 | 0.9973 | 0.0109 | 10,944–10,990 | 156–169 |
| 00008/C0004 | 28.58 | 0.9988 | 0.0055 | 9,800 | 148–149 |
| 00008/C0005 | 29.39 | 0.9986 | 0.0070 | 9,160–9,728 | 156–193 |
| 00009/C0001 | 25.95 | 0.9966 | 0.0137 | 10,696–11,000 | 178–203 |
| 00009/C0004 | 28.71 | 0.9980 | 0.0079 | 9,336–10,088 | 171–219 |
| 00009/C0005 | 28.85 | 0.9976 | 0.0101 | 9,576–11,000 | 176–182 |

Mean 27.97 dB, mean 175 s/cell. Seed spread ≤0.37 dB per view. Acceptance 6.0–10.3% of
attempted steps (canonical-run shape reproduced on 6 new views × 2 seeds).

## Results — karate full-frame benchmark (12/12 ok, seeds {0,1})

| view | seed-mean PSNR | MS-SSIM | LPIPS | N | total s |
|---|---:|---:|---:|---:|---:|
| 00005/rgb_10 | 33.37 | 0.9702 | 0.2019 | 6,024 | 3,116–3,611 |
| 00005/rgb_1000 | 34.78 | 0.9725 | 0.2341 | 6,792–7,824 | 4,015–4,866 |
| 00005/rgb_1001 | 34.14 | 0.9697 | 0.2091 | 7,560–7,816 | 8,598–10,256 |
| 00060/rgb_0 | 35.80 | 0.9766 | 0.2108 | 6,792–7,944 | 3,343–3,407 |
| 00060/rgb_1 | 33.80 | 0.9677 | 0.2081 | 8,072 | 11,953–19,033 |
| 00060/rgb_10 | 33.44 | 0.9702 | 0.1973 | 6,024 | 3,297–3,709 |

Mean 34.22 dB but LPIPS 0.196–0.237 and MS-SSIM 0.967–0.977; the schedule self-terminates at
6,024–8,072 of 11,000 rows on every cell (the O89 pattern). **Mean 6,600 s/cell vs 175 s/cell
masked at identical working resolution (≈38×; worst cell 19,033 s ≈ 5.3 h)**, with similar
attempted-step counts — the per-attempted-step cost, not the step count, explodes. Raw PSNR is
not comparable across arms (full-frame scores the whole frame; masked scores a masked
composite).

## Results — stage ablation (frame_00008/C0001 masked, 768 px, seed 0, 9/9 ok)

ΔPSNR vs `full` (24.688 dB, 127 fit-s): `no_polish` +0.359 · `no_bootstrap` +0.239 ·
`no_closure` +0.197 · `no_pareto_checkpoints` +0.192 · `no_redistribution` +0.089 ·
`no_detail_growth` −0.031 · `no_coverage_growth` −0.115. Every removal arm is also faster
(77–127 fit-s). `no_boundary_specialization` reads +7.947 dB PSNR but its MS-SSIM collapses to
0.4376 and LPIPS to 0.5837 at N=5,168 — a metric-semantics artifact of dropping boundary work
against the masked composite, not a win.

## Results — karate ablation (frame_00005/rgb_1000 full-frame, 768 px, seed 0, 8/8 ok)

ΔPSNR vs `full` (36.482 dB, 1,216 fit-s): `no_closure` +0.101 · `no_detail_growth` −0.176 ·
`no_redistribution` −0.265 · `no_coverage_growth` −0.276 · `no_polish` −0.291 ·
`no_pareto_checkpoints` −0.327 · `no_bootstrap` −0.368. All arms within ±0.37 dB; even at
768 px full-frame cells cost 704–1,428 fit-s (≈10× the masked 768 px cells).

## Reading (exploratory only)

1. The masked profile's canonical single-image behavior (quality range, ~7–10% acceptance,
   ~3 min/cell) generalizes across 6 new views × 2 seeds of the same capture family.
2. The unscreened full-frame arm is operationally prohibitive under the safe schedule at
   1200 px (38× mean, up to 109× worst-case cell cost vs masked at equal resolution) while
   leaving high LPIPS and self-terminating well below capacity. This is directly relevant
   context for BENCH-017 and for the direct-lane proposal in
   `docs/research/2026-08-07-fast-convergent-compressive-pipeline-design.md`, but it is not a
   BENCH-017 outcome: no protocol was frozen and these views are one capture family per dataset.
3. On one masked image/seed at 768 px, no single schedule stage except boundary specialization
   changes PSNR by more than 0.36 dB, and several removals are simultaneously slightly better
   and cheaper; on one full-frame karate cell the ordering reverses (`no_bootstrap` worst).
   Single-image, single-seed screening signals only — they motivate FIT-028/FIT-029/BENCH-018,
   not any recipe change.

## Forbidden follow-ups

Do not cite these numbers as a BENCH-017 verdict, a full-frame default decision, a schedule
retune justification on these images, or any cross-arm PSNR comparison. Any promotion needs a
task with a frozen protocol, disjoint data, and independent review.
