# FIT-018 responsibility-density recovery guard

**Verdict:** the transferred SAD-style `responsibility_alpha0.7` site score fails the frozen
shared-start recovery guard and remains an opt-in research control. This is a bounded negative
result on the existing COCO4 proxy fixture, not evidence against responsibility-based allocation
in general.

## Frozen comparison

Every arm starts from the same fitted 64-Gaussian field, chooses 16 parents, applies the same
moment-preserving split to reach 80 Gaussians, and independently replays 20- and 100-step recovery
horizons. The four images and seeds `{0,1}` produce eight paired cells. Before inspecting the donor
arm, `support` was selected as the stronger of the repository's `residual` and `support` controls
by mean post-20 PSNR.

The run explicitly pins `torch_num_threads=1` and enables deterministic algorithms. The first
parallel-CPU draft exposed small numerical trajectory drift despite a CPU label; it was discarded
before the final source-bound run. A second source-frozen replay preserved every non-timing
aggregate exactly.

## Result

| Site score | Immediate PSNR | Post-20 PSNR | Post-100 PSNR | Score s | Total-100 s |
|---|---:|---:|---:|---:|---:|
| `residual` | 20.3105 | 21.5835 | 23.0925 | 0.000096 | 0.996246 |
| `support` | 20.3093 | 21.6255 | 23.2022 | 0.003097 | 0.993924 |
| `responsibility_alpha1` | 20.3210 | 21.6274 | 23.2444 | 0.005861 | 0.981542 |
| `responsibility_alpha0.7` | 20.2470 | 21.6057 | 23.1611 | 0.005718 | 1.012152 |

Against the frozen `support` comparator, `responsibility_alpha0.7` changes immediate/post-20/
post-100 PSNR by `-0.0623/-0.0198/-0.0411 dB`, wins 4/8 post-20 pairs, and adds 1.8% to total
100-step time. It fails the required `+0.10 dB` post-20 gain and 6/8 sign agreement, so the guard
returns `survives=false`. Gaussian count, finiteness, post-100 non-regression, and time guard pass.

The `alpha=1` diagnostic is nearly invariant to a balanced opacity split, whereas each child's
`alpha=0.7` score is analytically scaled by `2^(0.7-1) ~= 0.812` under the exact half-opacity
gauge transform. This motivates a new gauge-equivalence question; it does not rescue or retune
FIT-018 on the same eight cells.

## Reproduction and provenance

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
  python -m benchmarks.responsibility_split_compare \
  --images tests/test_images/COCO_train2014_000000000009.jpg \
           tests/test_images/COCO_train2014_000000000025.jpg \
           tests/test_images/COCO_train2014_000000000030.jpg \
           tests/test_images/COCO_train2014_000000000034.jpg \
  --outdir results/fit018_responsibility_split_guard \
  --seeds 0 1 --max-side 64 --start-count 64 --split-count 16 \
  --pre-iters 40 --device cpu --render-chunk 512
```

Relevant-source combined SHA-256:
`32035c6988e66c3ec8a0c9a088433ab4a0833a66c2d5adc6a95dcd66b67d992b`.
The primary and replay configs contain the image hashes, source-file hashes, environment, dirty
repository snapshot, and exact command. `aggregate.json`/`rows.csv` contain the decision run;
`rerun_aggregate.json`/`rerun_rows.csv` contain the deterministic replay.

No parameters or fixtures were retuned after the gate result.
