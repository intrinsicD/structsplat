# BENCH-007 Stage-0b rate calibration

**Status:** complete rate-calibration evidence only; no method-quality conclusion is permitted.

## Question and frozen scope

Stage 0b estimates complete SSPL1 bytes per fitted Gaussian for the resolution-normalized count
ladder used by the preregistered Stage-1 killing pilot. It uses only the frozen DIV2K training IDs
`0002`, `0268`, `0534`, and `0800`; these images are excluded from method comparison. Each image
was measured at 2,048 and 8,192 Gaussians per megapixel with `quadtree_wse`, seed 0, 50 fit
iterations, codec bit mix `16/8/8/8`, and the owned exact-CUDA implementation of the normalized
weighted-sum renderer.

The resolved calibration identity is
`00c1495da6940162b57e745f114ad5c0d4bb975c08076e21996379871eea8358`. The run was
source-bound to clean commit `31837269aa892694c697b9d45c55d8bd78aa2374` on
`bench/007-actual-rate-phase-diagram`; the configuration records the four source-file and decoded
RGB hashes, dimensions, Python/package versions, renderer semantics, and clean repository state.

## Command

```bash
PYTHONPATH=src python -m benchmarks.actual_rate_phase_diagram calibrate \
  --data-root results/datasets/DIV2K_train_HR \
  --images results/datasets/DIV2K_train_HR/{0002,0268,0534,0800}.png \
  --outdir results/bench007_stage0b_20260714 \
  --renderer cuda --device cuda
```

## Result

All 8/8 calibration cells completed. Complete-stream bytes per Gaussian ranged from
`6.831850853548967` to `9.30172076652327`; the preregistered median estimator selected
`8.614970513660953 B/G`. This value, not the diagnostic calibration PSNR, was frozen into the
Stage-1 manifest before inspecting method metrics.

| Image | Native size | 2,048 G/Mpix | 8,192 G/Mpix |
|---|---:|---:|---:|
| 0002 | 2040 x 1848 | 8.916721 B/G | 8.526082 B/G |
| 0268 | 2040 x 1356 | 9.235658 B/G | 8.551476 B/G |
| 0534 | 2040 x 1224 | 9.301721 B/G | 8.678465 B/G |
| 0800 | 2040 x 1332 | 7.433962 B/G | 6.831851 B/G |

The large local streams and machine-readable rows remain under ignored
`results/bench007_stage0b_20260714/`. Their committed audit hashes are:

| Artifact | SHA-256 |
|---|---|
| `calibration_config.json` | `26a881f0e7e41227848592ea39ec57093b11efdd28b806fbc2cf73957ef8dacf` |
| `calibration_summary.json` | `cc64b98ab8d89d002aa9950c55438250c9dfdffedd5b18b5ef8cccf23b1b20a8` |
| `calibration.csv` | `03c15fbc41c00bf506807801570eb9d187592ccde5e121682f3f4725e5300579` |
| `calibration.jsonl` | `4dcae784b10f67cc3db4604540746e14ba770c45e5bd1e6a47d09665176b35dd` |

## Claim boundary

Stage 0b establishes only a measured complete-stream rate conversion for candidate planning. The
image-dependent range is retained rather than hidden, every Stage-1 arm receives the same frozen
count ladder, and no calibration PSNR is used to rank or tune allocation methods.
