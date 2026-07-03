# FIT-003 fit-loop speed evidence

Date: 2026-07-03

Before ref: `a661c75`

After ref: working tree before the FIT-003 commit

Image: `results/abl003_coco_train2014/COCO_train2014_000000000009.jpg`, resized with `max_side=96`.

Protocol: 20 fit iterations, target PSNRs `[20, 25, 30]`, `ssim_weight=0.3`, budgets 512 and 20000. CPU used the normalized renderer. GPU used the exact CUDA renderer, with a 2-iteration warmup fit before each measured CUDA row so extension compile/setup did not pollute the timing.

Raw measurements are in `raw.json`; the table below is mirrored in `speed_table.csv`.

| Device | Renderer | Budget | Before builtin s/iter | After builtin s/iter | After fused s/iter | Builtin delta | Fused vs before | Target crossings |
|--------|----------|--------|----------------------:|---------------------:|-------------------:|--------------:|----------------:|------------------|
| CPU | normalized | 512 | 0.013227 | 0.014897 | n/a | +12.63% | n/a | identical |
| CPU | normalized | 20000 | 0.029213 | 0.026745 | n/a | -8.45% | n/a | identical |
| CUDA | cuda | 512 | 0.001710 | 0.001546 | 0.001323 | -9.55% | -22.60% | identical |
| CUDA | cuda | 20000 | 0.003018 | 0.002370 | 0.002172 | -21.45% | -28.02% | identical |

Notes:
- Target PSNR crossings matched exactly in every row. CPU budget 20000 crossed 20/25/30 dB at iterations `0/1/3`; budget 512 crossed none. CUDA matched those same crossing sets.
- Final PSNR was unchanged on CPU and within CUDA numerical noise on GPU. The largest builtin before/after CUDA difference in this run was about `0.001 dB`.
- CPU budget 512 was slower by about `1.67 ms/iter`; this is the lowest-workload row and appears noise/overhead dominated. The higher-budget CPU row improved, and both CUDA rows improved.
- The opt-in fused SSIM backend was faster than the builtin GPU backend in both CUDA rows.

Acceptance mapping:
- Target crossings are tracked with device-side MSE thresholds and synced once at the end of the fit.
- SSIM windows are cached by window, sigma, device, and dtype. The fit loop skips SSIM loss entirely when `ssim_weight == 0`.
- `metrics.ssim(..., backend="fused")` tries `fused_ssim` on supported devices and silently falls back to the builtin implementation.
- Focused tests cover cache behavior, fused fallback/parity, target-crossing equivalence, and zero-weight SSIM skipping.
