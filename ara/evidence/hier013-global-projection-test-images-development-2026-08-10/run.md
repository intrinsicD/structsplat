# HIER-013 global projection on repository test images — development diagnostic

## Evidence status

Negative, non-claim development evidence. The run uses all 16 requested repository images (four
COCO and twelve DIV2K training images), three CUDA replicates, and four exact-7k arms. All inputs
were exposed development data, the source tree was dirty, and no distinct prospective reviewer
approved the protocol. The result can reject direct promotion of the frozen HIER-012 recipe; it
cannot establish held-out/general performance, select Field V2 semantics, complete FIT-046, make a
rate/speed claim, or change a default.

Portable report:
`results/hier013_global_projection_test_images_development_2026-08-10/index.html`.
Manifest SHA-256:
`e7315213558050b88b72470211a54779e7bb0df6aee3ce06d0d0144cdfe616a3`.
Frozen protocol-record digest:
`b9d38479fb61e3853bc8302fd25d60488ede40718a96addec40e2bccf6ffcd87`.

The report is not bundle-gate clean: `check_report_bundle.py --allow-dirty` reports 141
maintained-render parity failures above the frozen `2e-6` threshold. That integrity failure is
retained; the bundle was not repaired or regenerated after outcome access.

## Protocol

- Inputs: exact SHA-256-bound contents of `tests/test_images` and
  `tests/test_images/DIV2K_train_HR` (the supplied directory is `DIV2K`, not `FIV2K`).
- Raster: deterministic Pillow LANCZOS resize to maximum side 512; full-frame all-true mask.
- Replicates: seeds/labels 0, 1, and 2. The topology has no random sampler; repeats measure the
  CUDA recovery/render trajectory.
- Arms: HIER-005 control, touched-only projection, global projection, and guarded exchange plus
  global projection.
- Every arm: exact N=7,000, additive CUDA renderer, chunk 256, lossless Observation Field NPZ.
- Global projection: all 7,000 RGB rows, ridge `1e-8`, tolerance `1e-6`, 48-iteration cap,
  coefficient absolute limit 16, and exact step-zero fallback.
- Primary gate: global versus HIER-005 paired MSE/PSNR, requiring geometric-mean MSE ratio
  `<=0.80`, image-bootstrap upper bound `<1`, both families improving, no cell/local regression,
  perceptual noninferiority, complete integrity, and median projection overhead `<=25%`.
- Statistics: average paired log-MSE ratios within image, then over 16 images; 20,000 image-cluster
  bootstrap resamples with seed 13013.

Exact command:

```bash
PYTHONPATH=src python scripts/experiments/hier013_global_projection_development.py \
  --images tests/test_images \
  --out results/hier013_global_projection_test_images_development_2026-08-10 \
  --seeds 0 1 2 --target-gaussians 7000 --max-side 512 \
  --projection-ridge 1e-8 --projection-tolerance 1e-6 \
  --projection-max-iterations 48 --projection-coefficient-limit 16 \
  --max-exchanges 128 --site-count 96 --site-nms-radius 1 \
  --donor-count 64 --proposal-frontier 24 --coefficient-limit 16 \
  --device cuda --renderer cuda_additive --render-chunk 256 --lpips
```

Environment: repository revision `ceccd0e5a689e58eb2a0d7bced3e00e21de405e7` on dirty
`main`; Python 3.11.15, NumPy 2.2.4, torch 2.7.0+cu126, CUDA 12.6, NVIDIA RTX 4090. Total run
elapsed time was 5,546.30 seconds.

## Aggregate outcome

All 192 expected cells completed; there were no execution, missing, LPIPS-unavailable, or
non-finite cells.

| arm | geometric-mean MSE ratio | MSE reduction | mean PSNR delta | mean MS-SSIM delta | mean LPIPS delta | mean pixel-max delta | mean 7x7-max delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| HIER-005 control | 1.000000 | 0.000% | +0.0000 dB | 0 | 0 | 0 | 0 |
| touched projection | 0.997310 | 0.269% | +0.0117 dB | -0.00007767 | -0.00037612 | -0.0000509 | +0.0003690 |
| **global projection** | **0.997310** | **0.269%** | **+0.0117 dB** | **-0.00007767** | **-0.00037629** | **-0.0000509** | **+0.0003690** |
| exchange + global | 0.983446 | 1.655% | +0.0725 dB | +0.00192247 | -0.00699206 | -0.0471671 | -0.0148084 |

For global projection, the image-bootstrap 95% MSE-ratio interval is
`[0.9932034, 1.0000000008]`; the corresponding mean-PSNR interval is approximately
`[-3.3e-9, +0.02962] dB`. COCO's geometric-mean ratio is `0.9999999991` (numerical equality),
and DIV2K's is `0.9964146`. The primary 20% reduction clause fails by a wide margin, and the
interval includes no effect.

Only two images show material nonzero direct solves:

| image | global geometric-mean MSE ratio | mean PSNR delta | mean MS-SSIM delta | mean LPIPS delta | mean 7x7-max delta |
|---|---:|---:|---:|---:|---:|
| DIV2K/0268 | 0.984301 | +0.06872 dB | -0.00004735 | -0.00412625 | +0.000678 |
| DIV2K/0534 | 0.973089 | +0.11847 dB | -0.00119543 | -0.00189209 | +0.005227 |

The other 14 images are numerically unchanged by direct projection. Two of 48 global cells have
tiny MSE regressions beyond the frozen `1e-8` ratio tolerance because separate CUDA renders of the
unchanged step-zero field do not reproduce exactly. Four active cells worsen the displayed 7x7
maximum. Aggregate MS-SSIM also regresses. Thus perceptual/local controls cannot rescue the failed
primary gate.

Exchange plus global is the aggregate winner among the measured arms, including mean perceptual
and local metrics, but its +0.0725 dB / 1.66% MSE effect is still small, heterogeneous, and spends
extra topology work. It is a control result, not an exceptional successor or promotion decision.

## Mechanism audit

The main failure is an explicit precondition boundary:

- Only 6/48 global cells run past step zero: all three replicates of DIV2K/0268 and DIV2K/0534.
- The remaining 42 cells enter with coefficient maxima above the frozen absolute limit 16, so the
  solver correctly records checkpoint zero as non-selectable and performs no PCG iteration.
- Across global cells, incoming/selected coefficient maxima range from 2.985 to 2010.808, with
  median 91.797. The six admissible cells remain between 2.985 and 6.327.
- All six active solves select iteration 48 (the cap), with relative normal residuals about
  `1.0e-5` to `3.1e-5`; none reaches the declared `1e-6` tolerance.
- Touched-only and global projection activate on exactly the same six cells and have effectively
  identical aggregate results. The advertised benefit of releasing all rows therefore does not
  transfer under these full-frame HIER-005 coefficient trajectories.
- Exchange commits at least one pivot in 24/48 cells and can improve fields even when the global
  solve is blocked. Its gains vary materially across CUDA repeats on some images, and some searches
  reach the 128-pivot cap.

Raising the coefficient limit after observing these images would be a post-hoc rescue and is not
authorized. A future formulation must stabilize or constrain the incoming coefficient domain
prospectively, then use disjoint development data.

## Work and storage

| arm | median algorithm seconds | mean algorithm seconds | median projection seconds | median exchange seconds |
|---|---:|---:|---:|---:|
| HIER-005 | 93.305 | 98.843 | 0 | 0 |
| touched projection | 93.795 | 99.003 | 0.046 | 0 |
| global projection | 93.790 | 99.002 | 0.047 | 0 |
| exchange + global | 102.238 | 109.792 | 0.045 | 1.210 |

The global median overhead ratio is only 0.0475% because 42/48 cells stop before iteration one.
The six active global solves take 0.904--0.983 seconds. Exchange time ranges from 0.625 to 38.005
seconds. Median/max recorded peak CUDA allocation is 97,184,768/104,390,144 bytes. Allocation was
sampled once for each image/seed base workload and repeated on its four arm rows, so it is a
conservative per-seed ceiling rather than an arm-differential memory comparison.

Every exact-7k lossless Observation Field NPZ is 226,692 bytes, and every canonical raw array
payload has the same exact layout. These are self-contained reference-stream bytes at the resized
raster, not a selected codec or actual compression-rate claim.

## Integrity, replay, and bundle gate

Independent ledger replay verifies:

- 192/192 unique expected row keys;
- 192/192 field file hashes and canonical hashes;
- 192/192 exact N=7,000 counts and row-ledger files;
- 192/192 source hashes;
- 96/96 direct touched/global-versus-HIER-005 non-RGB array comparisons.

The CUDA numerical integrity gate fails:

| check | maximum | rows above 2e-6 |
|---|---:|---:|
| cold versus in-memory maintained render | 0.00113738 | 141/192 |
| repeated cold render | 0.000137687 | 131/192 |
| projection internal versus maintained render | 0.000136495 | 95/192 |
| projection adjoint relative error | 0.0000469685 | 48/192 |

Large ill-conditioned coefficients amplify atomic accumulation-order drift. Consequently,
`python scripts/check_report_bundle.py ... --allow-dirty` fails with 141 maintained-render parity
problems. `--allow-dirty` correctly does not waive numerical integrity. The raw bundle remains
useful diagnostic evidence because its exact executed source, fields, rows, hashes, and failures
are preserved, but it is not structurally claim-ready.

An independent cold render and metric replay of all 192 fields found maximum PSNR drift
`2.3031e-7 dB`, MSE drift `4.2803e-10`, MS-SSIM drift `1.7881e-7`, and zero drift in both displayed
pixel and patch maxima. This supports the stored aggregate interpretation, but it does not waive
the stricter frozen image-space parity predicate or make the bundle gate pass.

## Repository verification

All 59 focused HIER-010/011/013 and pixel-contraction tests pass. The required repository wrapper
passes whole-tree Ruff and reaches 1,739 portable tests passed, 25 skipped, with three disclosed
failures in untouched affine-diagnostics, SSP2E environment-capture, and SSP2V descriptor-race
tests. Its five structural checks pass when run directly after pytest stops the wrapper. The
HIER-010/011/012 bundles validate; HIER-013 alone retains the 141 parity findings above.

## Visual review

Full frames and native worst crops show obvious square/lattice artifacts on all inspected
full-frame fields. DIV2K/0268 exposes checker-like defects across face and hair; DIV2K/0534 shows
grid seams in smooth cloud texture; the COCO fields show coarse blur, ringing, and bright tiled
defects. The small active-projection metric gains are not a perceptible artifact repair. Exchange
changes some localized errors but does not make the 7k full-frame representation production-safe.

## Disposition

HIER-013 rejects promotion of the frozen HIER-012 pipeline on this requested development bank.
The exposed Janelle +2 dB result depended on a well-behaved incoming coefficient domain that is
absent from 14/16 new images. Global projection is therefore a conditional solver component, not
an exceptional general pipeline.

Recommended next action: do not retune these 16 images or simply raise the coefficient cap. First
design a new, prospectively bounded coefficient-domain/conditioning mechanism at HIER-005 recovery
or under the semantics selected by BENCH-020. Then test that formulation on a new development bank
before feeding any variable-projection schedule into FIT-046/BENCH-021.
