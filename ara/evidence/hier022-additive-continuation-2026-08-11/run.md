# HIER-022 learned-mass additive continuation diagnostic

## Scope and authority

This note audits the immutable development bundle
`results/hier022_coco4_s160_n640_i500_s01_diagnostic_2026-08-11`. It is a frozen, source-bound
COCO4 x two-seed x four-arm diagnostic executed on one RTX 4090 from a dirty implementation tree.
It is not formal confirmation, a default/semantic/codec decision, or evidence that every pure
additive Gaussian representation is inferior. The source snapshots inside the bundle are the
executed-source authority.

Manifest SHA-256:
`a334ed4eb21eb2bd635627a8eaeb8f0968905d929397175ace9e6cf6e47ff9fc`.
Metrics SHA-256: `2b9f900428aa8d148735c649c76a0a7daa502c3ca6cd74f0c410175d9fd32a28`.
Decision SHA-256: `1bb0920313b79885f01332fb056b2862dfcdecd0c6f743a97a46233d3cf95c18`.
`python scripts/check_report_bundle.py RESULTS --allow-dirty` passes.

## Frozen protocol

Five programmatic 48x48 fixtures at `N=128`, seed 0, and 160 steps selected coverage weight from
`{0.01, 0.05, 0.2}` before natural images; `0.05` won the frozen bounded terminal-MSE rule. Natural
cells use the four SHA-bound repository COCO images at max-side 160, `N=640`, seeds 0/1, 500
attempted steps, exact owned CUDA rendering, L1 + 0.3 SSIM loss, and required LPIPS. Arms are
ordinary normalized, ordinary additive, continuation without coverage loss, and continuation with
the selected loss. The continuation holds/anneals/ends at 35/50/15% and selects checkpoints only
from the exact additive tail.

## Aggregate results

| arm | PSNR dB | MS-SSIM | LPIPS | pixel max | 7x7 max | PSNR AUC | fit s |
|---|---:|---:|---:|---:|---:|---:|---:|
| normalized plain | 26.8399 | 0.96757 | 0.11518 | 0.43092 | 0.15383 | 25.8368 | 0.799 |
| additive plain | 26.2912 | 0.96654 | 0.16838 | 0.40206 | 0.15887 | 24.2236 | 0.816 |
| continuation, no coverage | 26.0448 | 0.96569 | 0.16989 | 0.42820 | 0.16614 | 25.1782 | 1.906 |
| continuation, coverage | 25.8371 | 0.96355 | 0.18038 | 0.43302 | 0.17606 | 25.0308 | 1.839 |

Coverage loss changes from `0.51250384` without the auxiliary to `0.01404593` with it, a 97.259%
reduction. Nevertheless the selected candidate trails plain additive by `0.45414 dB`, worsens
MS-SSIM by `0.002989`, LPIPS by `0.011995`, pixel maximum by `0.030962`, and 7x7 maximum by
`0.017187`; four of eight LPIPS cells exceed the allowed `+0.01`. Its fit-time ratio is `2.2546x`.
The no-coverage continuation has better AUC than additive (`+0.9546 dB-step normalized`) but still
ends `0.24639 dB` lower, so warm-path speed does not rescue terminal quality.

## Integrity and visual audit

All eight coverage candidates finish at exact `lambda=0`, exact `N=640`, finite, and mass-free.
Maximum coefficient magnitude is `2.83049`, maximum cold parity is `4.768e-7`, and no opacity,
mass, scale-optimizer metadata, or auxiliary RGB payload is serialized. Producer review of the
full-frame montage and representative worst crops finds no catastrophic lattice/checker pattern;
the coverage arm is visibly softer in fine foliage and zebra stripes, consistent with the numeric
local/perceptual regressions. Numeric failure is sufficient regardless of the report's immutable
`visual_review=pending` field.

## Causal disposition

The mechanism is rejected. Enforcing partition-like coverage succeeds numerically and harms the
image objective, so normalization is not replaceable here by a training-only learned-mass coverage
constraint. Telemetry also shows that the independent numerator/mass gauge does not follow the
ordinary normalized optimizer path during the nominal `lambda=1` hold, and the exact additive tail
gets only 75 steps. A permissible successor must use a new task/output/data selection, unit masses
so the start equation is exactly ordinary normalized, no coverage loss, a longer exact-additive
tail, and an explicit Adam-state-reset ablation. No HIER-022 cell may be retuned.

## Limitations

The result is one small consumed bank, two seeds, one device, dirty executed sources, and producer
review. Iteration/count matching is not equal renderer work. It does not establish asymptotic
approximation efficiency, actual rate, larger-budget behavior, downstream utility, or novelty.
