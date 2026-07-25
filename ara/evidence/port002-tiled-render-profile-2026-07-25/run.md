# PORT-002/PORT-003 tiled renderer profile — frozen gate passed

The GPU-native tiled renderer (`cuda_tiled`: in-extension CUB binning, shared-memory staged
kernels, PORT-003's warp-reduced backward atomics, opt-in exact ellipse culling) passes its
preregistered profile against the untiled exact `cuda` renderer on an RTX 3050. The shipped GPU
default is unchanged and remains `cuda`.

## Environment

NVIDIA GeForce RTX 3050, PyTorch 2.9.0+cu128, 5 warmup and 15 timed repeats per cell, CUDA-event
timing on the current stream with synchronized per-sample medians. Grid: resolutions
`{256, 512}` x counts `{2048, 8192}` x requested support overlap `{4, 16}` x axis ratio `{1, 6}`.

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
  python -m benchmarks.tiled_render_profile \
    --outdir results/port002_tiled_render_profile_rtx3050_head_pass
```

`LD_PRELOAD` is required on this machine because the conda `libstdc++` lacks `CXXABI_1.3.15`
needed by the built extension; without it the extension fails to load. That is an environment
property, not a repository defect.

## Parity precondition (ADR-0024)

The 2026-07-24 run of this profile failed its parity precondition while passing every performance
sub-check. The failure was diagnosed, not excused. All four recorded failures were one cell,
512²/N=8192/overlap-4/ratio-1, counted once per arm, and every arm reported the identical
`max_abs = 8.877516e-4` — including the unchanged shipped-default `cuda` baseline and the legacy
`cuda_tiled_torch_index` builder. Exactly `1` of `786,432` values exceeded tolerance; median gap
`0`, q99 `1.19e-7`, q99.99 `5.36e-7`; candidate-vs-baseline agreement was `1.788139e-7`.

The offending pixel `(row=235, col=348)` has normalized denominator `1.822550e-7` from a single
Gaussian at its support-fade cutoff. Under `support_fade`, `w = exp(-q/2) - exp(-c^2/2)` is
approximately proportional to `(c^2 - q)`, and float32 spacing near `q = 9` is `ulp ~ 9.54e-7`, so
`w` carries relative error `~ulp/(c^2 - q)`; measured `1.11e-2` at this pixel. ADR-0003's
normalize-by-denominator amplifies that by `eps/(D + eps)^2`, predicting `~7e-4` against the
observed `8.88e-4`.

An algebraic fix was tested and **rejected**: the identity
`exp(-q/2) - exp(-c^2/2) == exp(-c^2/2) * expm1((c^2 - q)/2)` improves the formula's own error at
that point from `3.76e-4` to `5.51e-8` relative, but leaves total error unchanged (`1.11e-2` naive
versus `1.15e-2` expm1) because the dominant term is the float32 representation of `q` itself. The
renderer's fade formulation is therefore unchanged.

ADR-0024 scopes governing parity to candidate-versus-baseline at the **same** frozen
`PARITY_ATOL`/`PARITY_RTOL` (`5e-4`), with reference agreement reported per arm and excused only
when the unmodified baseline mismatches there too. No tolerance, threshold, cell, or grid entry was
retuned and no new numeric constant was introduced. The amendment was authored **after** the
2026-07-24 timings were seen; that ordering is recorded in the gate object
(`parity_precondition_revision.authored_after_seeing_timings`), in the ADR, and in PORT-002's
notes.

## Result

`primary/` is the HEAD run and the authoritative artifact. `pre_ssim/` is an earlier passing run
taken before the separable-SSIM loss change, retained so the loss and renderer effects are
separable.

| check | limit | `pre_ssim` | `primary` |
|---|---|---|---|
| governing parity failures | 0 | 0 | 0 |
| representative step ratio | `<= 1.00` | `0.7227` | `0.6308` |
| high-N grid ratios | all `<= 1.00` | 8/8 | 8/8 |
| GPU index share of tiled step | `<= 15%` | `1.062%` | `1.358%` |
| representative CV | `<= 5%` | pass | pass |

Representative cell 512²/N=8192/overlap-16/ratio-6, milliseconds:

| arm | index | forward | backward | step |
|---|---|---|---|---|
| exact `cuda` (`pre_ssim`) | — | `2.366` | `5.960` | `12.673` |
| tiled (`pre_ssim`) | `0.097` | `0.703` | `2.110` | `9.159` |
| exact `cuda` (`primary`) | — | `2.330` | `5.918` | `11.215` |
| tiled (`primary`) | `0.096` | `0.734` | `2.145` | `7.075` |

`primary` grid ratios: `0.6276, 0.6533, 0.4654, 0.5390, 0.6512, 0.6602, 0.6374, 0.6308`. The ratio
differs between the two runs only because the separable-SSIM change shrinks the shared
non-renderer term, enlarging the renderer's share of the step. No renderer code differs between
them.

Baseline-attributable reference mismatches are reported rather than dropped: 4 records, one cell,
`1/786,432` values per arm, `max 8.877516e-4` versus reference and `<= 1.788139e-7` versus
baseline. Their numerical identity across the two runs also confirms that extracting
`classify_arm_parity` was behaviour-preserving.

## Run-order disclosure

The profile was executed six times on this machine. Run 1 was the original gate (parity fail).
Run 2 was the first ADR-0024 build and failed the frozen `5%` CV limit with every arm noisy,
including the untouched baseline at `8.58%`. Run 3 added a hygiene fix — the four parity render
buffers are freed before the timed section, restoring the pre-amendment allocation footprint — and
passed; the CV recovery may be attributable to that fix or to desktop contention easing, and the
two are not separated by this evidence. Runs 4 and 5 followed the separable-SSIM change and failed
CV (`4.74--8.23%`) despite medians agreeing to `0.2%`. Run 6 is `primary/` and passed.

This GPU drives a desktop, so `nvidia-smi` reports non-zero utilization with no compute apps. The
frozen CV guard rejected three of six runs and is doing real work. Two of the three passes were not
preceded by any renderer change, and medians are stable across all six runs, so the variance is in
the machine rather than the kernels.

## Scope

Passing authorizes the fair-protocol end-to-end fit benchmark and nothing else. It does **not**
authorize a default flip, a cross-GPU claim, a quality/convergence claim, or a compression claim.
This is a single consumer-GPU microprofile of a synthetic Gaussian field, not an image-fitting
result. PORT-003's deterministic-accumulation item remains open: per-warp reduction order is fixed
but cross-warp `atomicAdd` ordering is not.

## SSIM loss term (`ssim_microbench.py`, `ssim_microbench.json`)

The passing profile showed the renderer is no longer the dominant fit-step cost. At 512² the
non-renderer remainder is `~6.1--6.3 ms` and flat in `N`, overlap, anisotropy, and renderer arm.
It is almost entirely the SSIM half of the default `0.7 L1 + 0.3 SSIM` objective: at 512²,
SSIM forward+backward measures `6.5--9.4 ms` against L1's `0.21--0.32 ms`, with Adam at
`~0.19--0.34 ms` and independent of `N`.

`metrics._gaussian_window` now caches a separable Gaussian pair and `_ssim_builtin_bchw` runs two
1D passes instead of one 11x11 pass — the same operator up to float associativity. Parity is
enforced by
`tests/test_metrics.py::test_ssim_separable_window_matches_dense_outer_product_reference`, which
reconstructs the dense outer-product form and checks value and gradient; measured agreement is
`~1e-8` absolute on the value and `~2e-6` relative on the gradient.

Speedup versus the dense form, stable across repeated runs at and above 256²:

| size | separable | + cached target (FIT-027 prototype) |
|---|---|---|
| 256² | `1.36--1.47x` | `1.54--1.87x` |
| 384² | `1.42--1.44x` | `1.56x` |
| 512² | `1.39--1.48x` | `1.68--1.76x` |
| 1024² | `1.39--1.42x` | `1.65--1.72x` |

**No claim is made below 256².** At 128² and 192² a single forward+backward is under a
millisecond and both arms sit pinned at the same `~1.1--1.2 ms` launch-overhead floor — the
absolute time barely scales with pixel count there. Repeated measurement of that regime on this
machine produced `0.64x`, `0.92x`, `0.94x`, `1.04x`, `1.45x` and a nonsensical `3.08x`, with
per-sample spreads of `144--570%` of the median, and a dense-versus-dense control measured `1.45x`
apart from itself. An earlier reading of `0.64x` as a genuine small-image regression was noise;
a size-gated dispatch was prototyped in response, then removed once the control run showed the
crossover it defended against could not be resolved on this hardware. The shipped code is the
unconditional separable form. If a small-image regression matters for the ablation's `max-side
160` cells, it needs a quieter machine or a dedicated harness, not this artifact.

The `cached target` column is an unimplemented FIT-027 prototype living only in
`ssim_microbench.py`: it caches `mu_t`/`mu_t2`/`sig_t`, which are recomputed every iteration
although the target is fixed within a fit. Gradient agreement with the dense reference is
`~2e-6` relative, the same as the separable form.

Reproduce with:

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
  python ara/evidence/port002-tiled-render-profile-2026-07-25/ssim_microbench.py
```

Absolute times in `ssim_microbench.json` were captured during a noisier period than the profile
runs (512² dense read `6.6 ms` early and `9.4 ms` late). Ratios were stable throughout; treat the
absolute milliseconds as indicative and the ratios as the result.

## Artifact hashes

SHA-256: `primary/raw.json` =
`f87a799056aac65ebcee48f37d28989d38bbb3720ffdac5a35ada250632b9e4f`, `primary/summary.md` =
`395dc29a85e6984a739b4d0440c78b16150372395890818f9de09eba4ac39186`, `pre_ssim/raw.json` =
`cd0b25acbb007fdfdedc2bda3d4a7230cd05d27398666189f819b77381db217e`, `pre_ssim/summary.md` =
`c671560eb35b33ebfc1823dad6a124f1ecaa9f145075a3e55e7e86f3734a467c`.

Executed sources at the `primary` run: `benchmarks/tiled_render_profile.py` =
`b1a1924f1c26e0de892c63ca58e583dad5752a7b93cfe2c8f78d5af7fe4641d8`,
`src/structsplat/metrics.py` =
`50da8cc358816ddb47c4e542b422595405b14aa564167d3c3589f8c2c0ef10a2`.
