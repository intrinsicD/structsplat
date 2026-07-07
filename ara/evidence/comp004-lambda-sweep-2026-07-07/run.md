# COMP-004 lambda sweep

Date: 2026-07-07

Goal: finish the COMP-004 decision slice by comparing fit-time QAT plus `lambda_rate` against the
existing codec-only and post-fit QAT controls under matched initialization, bit depths, images, and
extra optimization budget.

## Protocol

- Images: `kodim01`, `kodim07`, `kodim13`, `kodim19`.
- Budget: 512 Gaussians.
- Init: `strategy=quadtree_wse`, seed 0.
- Fit: 600 base iterations, `renderer=cuda`, `max_side=384`, `render_chunk=512`.
- Codec ladders: `16/8/8/8`, `12/8/6/8`, `12/6/6/6`, `10/5/5/5`.
- Controls per fitted base field:
  - `none`: encode the base fitted field directly.
  - `qat`: existing post-fit STE QAT finetune for 100 iterations.
  - `refine_noste`: 100 extra plain fit iterations with no STE.
  - `fit_qat`: 100 fitter iterations with `qat_mode=ste`, matching each codec ladder's bit depths.
- Lambda sweep: `lambda_rate in {0.0, 0.001, 0.01}` for `fit_qat`.

Command:

```bash
python -m benchmarks.rate_distortion \
  results/datasets/abl004/kodak24/kodim01.png \
  results/datasets/abl004/kodak24/kodim07.png \
  results/datasets/abl004/kodak24/kodim13.png \
  results/datasets/abl004/kodak24/kodim19.png \
  --budgets 512 \
  --strategy quadtree_wse \
  --seeds 0 \
  --iters 600 \
  --qat-iters 100 \
  --fit-qat-modes ste \
  --lambda-rates 0.0 0.001 0.01 \
  --max-side 384 \
  --renderer cuda \
  --render-chunk 512 \
  --outdir results/comp004_lambda_sweep_2026_07_07 \
  --device cuda
```

## Aggregate result

Paired deltas are versus the matching `mode=none` encoded row. `RD wins` means higher PSNR at no
larger encoded bpp.

| mode | qat mode | lambda | pairs | mean PSNR | mean bpp | dPSNR | dBPP | RD wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `none` | off | 0 | 16 | 21.8484 | 0.3830 | 0.0000 | 0.0000 | - |
| `qat` | off | 0 | 16 | 22.5064 | 0.3832 | +0.6580 | +0.0001 | 6/16 |
| `fit_qat` | ste | 0 | 16 | 22.4707 | 0.3834 | +0.6222 | +0.0003 | 7/16 |
| `fit_qat` | ste | 0.001 | 16 | 22.4718 | 0.3833 | +0.6234 | +0.0002 | 7/16 |
| `fit_qat` | ste | 0.01 | 16 | 22.5023 | 0.3831 | +0.6539 | +0.0000 | 6/16 |
| `refine_noste` | off | 0 | 16 | 21.7204 | 0.3820 | -0.1280 | -0.0010 | 7/16 |

By bit ladder:

| bits | best fit-time QAT dPSNR | best fit-time dBPP | post-fit QAT dPSNR | post-fit dBPP | interpretation |
|---|---:|---:|---:|---:|---|
| `10/5/5/5` | +1.3965 | +0.0038 | +1.5036 | +0.0041 | QAT helps strongly, but costs a little bpp. |
| `12/6/6/6` | +0.6995 | +0.0008 | +0.6368 | +0.0006 | Fit-time QAT slightly beats post-fit PSNR. |
| `12/8/6/8` | +0.7794 | -0.0025 | +0.6636 | -0.0025 | Fit-time QAT with `lambda=0.001` is best. |
| `16/8/8/8` | -0.1928 | -0.0013 | -0.1719 | -0.0017 | QAT slightly hurts high-bit final PSNR. |

## Verdict

Fit-time STE QAT is implemented and meaningful: it improves encoded low/mid-bit PSNR over direct
post-hoc quantization, and `benchmarks/rate_distortion.py` now writes lambda-sweep RD rows with
the same quantizer assumptions used for final encoding.

It is not a new default. The existing post-fit QAT control remains marginally stronger on this
slice overall (+0.6580 dB versus +0.6539 dB for the best fit-time lambda), while `lambda_rate`
barely changes actual bpp at this scale. Keep `qat_mode` and `lambda_rate` searchable for future
codec work, but present COMP-004 as plumbing plus a local low/mid-bit option rather than evidence
that entropy-aware fitting beats the current QAT baseline.

## Artifacts

- `rate_distortion.csv`, `rate_distortion.json`, `rate_distortion.md`: raw 96-row sweep.
- `rate_distortion_config.json`: resolved run config and environment versions.
- `paired_deltas_vs_none.csv`: row-level deltas versus direct encode.
- `aggregate_by_mode.csv`: overall grouped comparison.
- `aggregate_by_bits.csv`: grouped comparison by codec ladder.
