# Top-K responsibility oracle ceiling

This is an **oracle work ceiling, not a measured speedup**. The implementation first
evaluates every current tile candidate and only then selects the per-pixel top-K weights.
A deployable kernel would need a cheaper exact/approximate winner-search mechanism.

The oracle matches the normalized renderer's rounded, clipped support rectangles. It
does not add a q cutoff. `positive_weight_contributions` excludes only weights that are
numerically zero (including float32 exponential underflow or support fade).

Input field: `results/coco4_current_20k_500/COCO_train2014_000000000009_aniso_flanking_20k_500.npz` (20,000 Gaussians).
Input image: `tests/test_images/COCO_train2014_000000000009.jpg` (640x480).
Full oracle quality: 34.62571 dB PSNR, 0.994055 MS-SSIM.
Full-oracle / `cuda` parity: max abs 1.34e-05, mean abs 6.47e-08 (gate 2e-05, passed=True).

| K | Target PSNR | Delta vs full | MS-SSIM | PSNR to full | Mean abs to full | Mean mass | p05 mass | Ideal positive-work reduction | Ideal rectangle-work reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.61907 | -20.00664 | 0.760053 | 14.6833 | 0.109407 | 0.370045 | 0.180439 | 96.74% | 97.51% |
| 2 | 21.10620 | -13.51950 | 0.897496 | 21.3436 | 0.048897 | 0.586001 | 0.334210 | 93.47% | 95.02% |
| 4 | 28.33474 | -6.29097 | 0.973430 | 29.5287 | 0.017221 | 0.807563 | 0.567700 | 86.95% | 90.03% |
| 8 | 33.43626 | -1.18945 | 0.992198 | 39.6745 | 0.004067 | 0.951555 | 0.826495 | 73.94% | 80.10% |
| 16 | 34.57605 | -0.04966 | 0.994010 | 54.0074 | 0.000432 | 0.996064 | 0.977087 | 49.59% | 61.50% |

Rectangle contributions: 12,328,179; numerically positive contributions: 9,413,898; tile candidate-pixel pairs actually formed by this audit: 28,881,152.

The two reduction columns count retained color/normalization contributions relative to
positive weights and current clipped-rectangle evaluations. They assume an oracle reveals
the winners for free and therefore must not be quoted as throughput, latency, or FLOP
measurements. `audit_seconds` in `result.json` is provenance only.

## Reproduce

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src TORCH_EXTENSIONS_DIR=/tmp/structsplat_torch_extensions python -m benchmarks.topk_responsibility_oracle --image tests/test_images/COCO_train2014_000000000009.jpg --field results/coco4_current_20k_500/COCO_train2014_000000000009_aniso_flanking_20k_500.npz --outdir results/topk_responsibility_oracle_2026-07-15 --device cuda --validation-renderer cuda --ks 1 2 4 8 16 --tile-size 16 --sigma-cutoff 3.0 --aa-dilation 0.0 --parity-max-abs 2e-05
```

Relevant-source combined SHA-256: `f4f69da4ede522f1eb61654d524da7f9f897c0b77c9d257b776253a24f70e6e5`.
Field SHA-256: `3da2d93ea9ce18b1bb1683679e7631eb355d19541fb394ee47f4c143be940031`.
Image SHA-256: `35cdfe8259aca40d564baf33ee749d82ce852446bd9574f0c47551d8bfffda99`.

Machine-readable outputs: `rows.csv`, `rows.json`, `result.json`, and `config.json`.
