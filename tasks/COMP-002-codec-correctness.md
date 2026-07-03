# COMP-002: Codec / metrics / CLI correctness and protocol fixes

**Status: todo.** Confirmed defects from the 2026-07-03 repo review (all hand-verified against
the code).

## Context
1. **QAT and RD evaluation hardcode the normalized renderer.** `qat_finetune` and `rd_point`
   call `render()` directly, ignoring `fcfg.renderer` — fields fitted with additive/CUDA/gsplat
   modes (ADR-0006) are fine-tuned and RD-scored through the wrong compositing model.
   (`src/structsplat/codec.py:233-235,256-258`)
2. **QAT drops config:** parameter groups omit `fcfg.lr_opacity` (opacities fine-tune at the
   hardcoded 1e-2), and the loss hardcodes L1 ignoring `fcfg.pixel_loss`/`charbonnier_eps`.
   (`src/structsplat/codec.py:229-230,236-237`)
3. **Means quantization domain doesn't cover the fitted domain.** `fit` never clamps means;
   off-image Gaussians snap to the border on encode with error unbounded by the lattice step.
   (`src/structsplat/codec.py:113`)
4. **`ms_ssim` crashes on batched `(B,3,H,W)` input** despite the documented contract
   (`squeeze(0)` no-op → 4-D permute fails), and has no minimum-size guard — at 48×48 the
   coarse scales are 3×3 vs an 11×11 window, dominated by zero padding, silently entering
   benchmark tables. (`src/structsplat/metrics.py:53-62`)
5. **Unclamped-render metrics.** `rd_point` (and fit's final eval) compute PSNR/MS-SSIM on the
   unclamped render, which fitted fields overshoot — biased vs display-referred baselines
   (JPEG/GaussianImage; COMP-001 criterion 5). (`src/structsplat/codec.py:255-268`)
6. **CLI:** `save_image` truncates instead of rounding (every saved PNG biased up to −1/255,
   `src/structsplat/cli.py:17-20`); `--lpips` loads AlexNet and computes a value that is never
   printed or saved (`src/structsplat/cli.py:87`).
7. **Reproducibility/self-containedness:** `rate_distortion.py` rows omit strategy/iters/
   qat_iters/lr/zlib-level (invariant 5, `benchmarks/rate_distortion.py:49`); the QAT-vs-no-QAT
   comparison has no equal-budget control (QAT rows get 150 extra optimization iterations, so
   the reported gain conflates lattice-settling with extra training,
   `benchmarks/rate_distortion.py:57`); the bitstream header records no render semantics
   (renderer mode, aa_dilation, sigma_cutoff) or scale_max, so a blob is not decodable without
   an out-of-band FitConfig that must happen to match (`src/structsplat/codec.py:129-136`).

## Goal
The codec round-trips the field *and* its rendering semantics; metrics obey their documented
contracts; RD numbers are unbiased and reproducible from their own artifacts.

## Acceptance criteria
- [ ] `qat_finetune`/`rd_point` render via `render_field(..., mode=fcfg.renderer, ...)`;
      QAT passes `lr_opacity` and reuses `fit._pixel_loss` (hoisted to a shared location);
      test: additive-fit field round-trips through QAT+RD with additive rendering.
- [ ] Means either clamped to the extent at the end of `fit()` or quantized over a padded
      extent stored in the header; round-trip test with off-image means bounds the error by
      the lattice step.
- [ ] `ms_ssim` accepts BCHW (call `ssim(p, t)` directly) and drops scales when
      `min(H,W) < win` at that scale (renormalizing weights) or raises; tests for both.
- [ ] Renders clamped to [0,1] before PSNR/MS-SSIM in `rd_point` (and fit's final eval, or the
      unclamped convention is documented in metrics.py's protocol notes — one convention,
      stated).
- [ ] `save_image` rounds (`np.rint`); `--lpips` prints/saves the value; tests.
- [ ] `rate_distortion.py` writes a full config record per run and adds an equal-budget
      no-STE fine-tune row per bit mix (isolating lattice-settling from extra compute).
- [ ] Header carries renderer mode, aa_dilation, sigma_cutoff (+ scale_max or a documented
      statement that decoded fields lose it); decode-and-render works with no out-of-band
      FitConfig; round-trip test.

## Interfaces touched
`src/structsplat/codec.py`, `src/structsplat/metrics.py`, `src/structsplat/cli.py`,
`src/structsplat/fit.py` (shared `_pixel_loss`, optional mean clamp),
`benchmarks/rate_distortion.py`, `tests/test_codec.py`. Header change bumps the blob format →
note in ADR-0007.

## Depends on
COMP-001, FIT-001.
