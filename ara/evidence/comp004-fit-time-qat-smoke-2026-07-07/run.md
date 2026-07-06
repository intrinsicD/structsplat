# COMP-004 Fit-Time QAT Smoke

Date: 2026-07-07

Purpose: validate the first compression-aware fitting slice. This checks that `fit()` can optimize
through a quantized/noisy render view, apply a differentiable rate proxy, and return the frozen
codec config needed for final encoding. It is not an RD improvement claim.

## Implemented

- Added `FitConfig.qat_mode in {"off", "ste", "noise"}`.
- Added `FitConfig.lambda_rate`.
- Added QAT bit-depth knobs for means, scales, rotation, colors, and opacity.
- Added `differentiable_rate_bpp(...)` covering means, log-scales, rotations, colors, and opacity.
- `fit()` now logs `qat_rate_bpp` / `rate_loss` and returns `qat_codec_config` when QAT is enabled.
- `structsplat fit` exposes `--qat-mode`, `--lambda-rate`, and QAT bit-depth flags.

## Focused Tests

Command:

```bash
PYTHONPATH=src:. pytest \
  tests/test_fit_dynamics.py::test_fit_time_qat_ste_returns_codec_config_and_rate_history \
  tests/test_fit_dynamics.py::test_fit_time_qat_noise_runs_and_affine_fails_closed \
  tests/test_fit_dynamics.py::test_differentiable_rate_proxy_has_gradients \
  tests/test_cli.py::test_fit_cli_accepts_qat_rate_flags -q
```

Result: 4 passed in 1.47 s.

## Smoke Command

```bash
PYTHONPATH=src:. python - <<'PY'
from benchmarks.common import load_image, target_tensor
from structsplat import init as _init, codec
from structsplat.config import FitConfig, InitConfig
from structsplat.fit import fit

img = load_image('tests/test_images/COCO_train2014_000000000034.jpg', max_side=32)
target = target_tensor(img, 'cpu')
field = _init.build_field(img, InitConfig(strategy='random', num_gaussians=16, seed=0), device='cpu')
cfg = FitConfig(
    iters=4,
    log_every=1,
    render_chunk=64,
    ssim_weight=0.0,
    qat_mode='ste',
    lambda_rate=0.01,
    qat_bits_means=10,
    qat_bits_scales=5,
    qat_bits_rot=5,
    qat_bits_colors=5,
)
out = fit(field, target, cfg, verbose=False)
rd = codec.rd_point(out['field'], target, cfg, out['qat_codec_config'])
print({
    'psnr': round(float(out['psnr']), 4),
    'qat_rate_bpp': round(float(out['qat_rate_bpp']), 6),
    'rd_psnr': round(float(rd['psnr']), 4),
    'rd_bpp': round(float(rd['bpp']), 4),
    'raw_bpp': round(float(rd['raw_bpp']), 4),
    'bits': rd['bits'],
})
PY
```

Result:

```text
{'psnr': 18.1677, 'qat_rate_bpp': 0.580206, 'rd_psnr': 18.101, 'rd_bpp': 10.0952, 'raw_bpp': 1.1905, 'bits': [10, 5, 5, 5]}
```

Verdict: fit-time QAT/rate plumbing works and can pass the optimized quantizer assumptions to final
encoding. COMP-004 remains partial because no lambda sweep or low-bit RD improvement has been
shown yet.
