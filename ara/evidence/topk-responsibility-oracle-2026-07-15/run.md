# Top-K responsibility oracle ceiling

**Verdict:** top-16 retains near-full quality on one saved 20,000-Gaussian field and exposes a
large ideal contribution-work ceiling, while top-8 and below are rejected for near-lossless use.
This audit computes every tile candidate before selection, so it is not a measured speedup and
does not establish deployable top-K performance.

## Source-bound field audit

- Field: `results/coco4_current_20k_500/COCO_train2014_000000000009_aniso_flanking_20k_500.npz`
  (`N=20,000`, SHA-256 `3da2d93ea9ce18b1bb1683679e7631eb355d19541fb394ee47f4c143be940031`).
- Image: `tests/test_images/COCO_train2014_000000000009.jpg` (`640x480`, SHA-256
  `35cdfe8259aca40d564baf33ee749d82ce852446bd9574f0c47551d8bfffda99`).
- Full oracle: `34.62571 dB` PSNR and `0.994055` MS-SSIM.
- Full-oracle versus owned exact-CUDA maximum absolute error: `1.344e-5`, below the `2e-5` gate.

At `K=16`, target PSNR is `34.57605 dB` (`-0.04966 dB`), responsibility mass is `0.996064`
on average and `0.977087` at p05, and an oracle would remove 49.59% of numerically positive color
contributions or 61.50% of clipped-rectangle visits. At `K=8`, quality falls by `1.18945 dB`;
smaller K values lose still more. The reduction percentages assume that winner discovery is free.

The exact rows for `K={1,2,4,8,16}`, parity measurements, contribution counts, environment, and
reproduction command are in `rows.csv`, `rows.json`, `result.json`, and `config.json`. Relevant
source combined SHA-256:
`f4f69da4ede522f1eb61654d524da7f9f897c0b77c9d257b776253a24f70e6e5`.

Before any performance claim, a future implementation must find winners without first evaluating
all candidates, repeat the phase diagram on multiple fields, and report end-to-end latency,
memory traffic, quality, and fallback behavior.
