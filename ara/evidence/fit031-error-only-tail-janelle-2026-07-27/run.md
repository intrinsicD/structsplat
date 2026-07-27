# FIT-031 error-only tail — Janelle C0001 development screen

## Outcome

The optional terminal stage is functional, bounded, and Pareto-safe on the exposed masked Janelle
C0001 development image. After the ordinary schedule reached 11,000 rows, foreground-MAE
effective support estimated 14,177 residual sites and requested half, 7,089 rows. Nine 512-row
batches committed (4,608 rows total); the tenth wave failed the full gate at every geometric
bisection from 512 through the configured eight-row minimum. The first low-learning-rate
fixed-topology block also failed after deterministic backtracking, so the stage stopped at a fixed
point with no global convergence step committed.

Within this run's own pre-tail/post-tail pair, foreground and boundary PSNR improved by
`+0.522239/+0.582752 dB`, CVaR99 and p99 MSE fell by `12.09%/14.82%`, and boundary holes fell by
`0.4821` percentage points. Interior holes and both outside-mask metrics stayed exactly zero. The
saved 15,608-row field cold-rescored within `5.59e-9` of every protected stored metric.

This is an exposed one-image, one-seed, one-GPU development result. The clean existing default run
ended at 10,824 rows while this run reached 11,000 before the tail; CUDA atomic accumulation also
makes their trajectories tolerance-reproducible rather than bit-exact. The final fields are
neither count- nor rate-matched. The `+0.490408 dB` display-PSNR difference between those terminal
files is context, not an authorized direct-method comparison. No default, generality, efficiency,
or codec-rate claim follows.

## Claim disposition

| claim | kind and scope | evidence | disposition |
|---|---|---|---|
| `--fine-detail` is default-off and logs the estimator, allocation, convergence, and protected metrics | implementation plumbing | focused tests, run config/history/report, executed-source patch | confirm |
| The logged estimate and request implement the specified formula | mechanism on one executed run | logged sums, independent arithmetic replay, quantized snapshot replay | confirm |
| Tail births are residual-ranked, isotropic, at most 1.25 px, and bounded to 8–512 rows | mechanism on one executed run | nine winner records plus rejected tenth-wave bisections | confirm |
| The accepted tail improves its own pre-tail protected state | exposed single-image development result | source-bound within-run before/after pair and gate replay | confirm |
| The tail is better than the existing default at equal count or rate | comparative method claim | terminal fields are unmatched in count, work, and CUDA trajectory | not tested; unauthorized |
| The tail is efficient or should become the recipe default | efficiency/default claim | one execution, 4,608 extra rows, unmatched wall time | not tested; unauthorized |
| The requested rows completely remove the residual | zero-error claim | only 4,608/7,089 rows committed and nonzero residual remains | refute as an outcome; never claimed by the estimator contract |

## Protocol

- Source: `C0001_crop.png`, RGB SHA-256
  `9e933e93797fd806adc4255f443361d4c2304956049a31ee1cddfeab72b8e1c1`; mask SHA-256
  `58940a83c1858b8c48ff818593b98c6029e7f78224dd067195ed3146a173540c`; decoded
  target-pixel SHA-256
  `c6893d12b3265bf68219d6af4f084ccaaef32e9d2d4c8b0a19e9cacfa5148fb4`.
- Fit size: `1200x437` from native `3964x1444`; seed `0`; mask margin `0.75`.
- Device: NVIDIA GeForce RTX 4090; PyTorch `2.7.0+cu126`; CUDA runtime `12.6`; NumPy `2.2.4`.
- Execution base: commit `797eac6b0471a8f655e485e47b9af610e35b2fc0`, dirty source preserved by
  `executed-source.patch` (SHA-256
  `f309e59d3081c42174cf8c2f55039883820cfe8caf608ed8b132074e266f52cc`).
- Tail: fraction `0.5`, maximum batch `512`, minimum batch `8`, isotropic maximum scale `1.25`,
  80 recovery steps per attempted batch, full strict Pareto gate, and a 4,000-step
  fixed-topology ceiling in 250-step blocks.
- Existing clean baseline:
  `runs/janelle_C0001_current_pipeline_20260727`.
- Audited tail run:
  `runs/janelle_C0001_error_tail50_min8_20260727`.

The first diagnostic execution exposed a hidden `event_min_count=1` override after seven full
batches. It completed with five single-row commits, was excluded from scientific evidence, and is
preserved locally at `runs/janelle_C0001_error_tail50_20260727`. The corrected run above preserves
the configured eight-row minimum and contains no sub-minimum commit.

Exact run command:

```bash
PYTHONPATH=src python scripts/convert.py \
  runs/janelle_C0001_current_pipeline_20260727_inputs/C0001_crop.png \
  runs/janelle_C0001_error_tail50_min8_20260727 \
  --mask runs/janelle_C0001_current_pipeline_20260727_inputs/mask_C0001_crop.png \
  --device cuda:0 --max-side 1200 --seed 0 --lpips --fine-detail
```

## Results

The protected values below are the raw normalized-renderer metrics used by the gate.

| state | N | FG PSNR | boundary PSNR | CVaR99 MSE | p99 MSE | interior holes | boundary holes |
|---|---:|---:|---:|---:|---:|---:|---:|
| same-run pre-tail | 11,000 | 26.789684 | 15.003354 | .1845860 | .0107341 | 0.0000% | 11.9772% |
| same-run post-tail | 15,608 | **27.311923** | **15.586106** | **.1622617** | **.0091428** | 0.0000% | **11.4951%** |

The report's clamped display metrics are:

| result | N | PSNR | SSIM | MS-SSIM | LPIPS | total |
|---|---:|---:|---:|---:|---:|---:|
| existing clean default context | 10,824 | 26.821604 | .965199 | .988934 | .047659 | 270.272 s |
| corrected fine-detail run | 15,608 | 27.312012 | .966735 | .989417 | .046202 | 570.917 s |

The table is intentionally not an equal-count, equal-rate, or work-normalized comparison. The
tail run attempted/accepted `21,624/2,740` optimizer steps overall. Its nine accepted tail
transactions contributed 690 accepted recovery steps; the rejected tenth wave attempted 560
steps across counts `512,256,128,64,32,16,8`, and fixed-topology backtracking attempted 468 steps
without a commit.

## Scientist pass

- Source, mask, decoded target, seed, environment, initialization config, and fit config match the
  existing clean default bundle.
- Recomputing `ceil(sum(e)^2 / sum(e^2))` from the logged float64 sums gives exactly 14,177, and
  `ceil(0.5 * 14,177)` gives exactly 7,089. Recomputing from the persisted 8-bit pre-tail PNG gives
  14,055 (`0.861%` lower), consistent with the documented quantization limitation.
- All nine accepted waves contain exactly 512 rows. Winner metadata reports residual-only score,
  isotropic axis ratio `1.0`, and maximum base scale at or below `1.25`. The accepted-row sum
  equals the recorded 4,608 activated rows.
- Independent history replay found zero continuity failures, matched top-level attempted and
  accepted step sums, passed 36 accepted gate records, and reproduced 68 rejected attempt
  decisions without mismatch.
- Cold loading the terminal NPZ on the RTX 4090 reproduced protected metrics within `5.59e-9` and
  display metrics within `1.01e-7`; the field hash is
  `48a71fb5878a3600b42c03f6b94224152d30a4820b34e925318020a6d636e1c4`.
- Visual inspection of the pre-tail and final reconstruction/error images showed a modest
  residual-darkening consistent with the metrics and no obvious new containment artifact. This is
  descriptive QA, not a perceptual preference claim.
- The portable verification invocation completed with 1,434 passed, 32 skipped, and three
  failures already present in the crash-recovered `lastfailed` cache: an affine rank-condition
  expectation, a `9.54e-7` no-mask parity tolerance assertion, and a descriptor-mutation
  environment test. None touches the FIT-031 diff, and every changed-surface test executed by that
  suite passed. Post-correction focused subsets passed 4/4 error-tail tests and 7/7
  workflow/default-parity tests; Ruff and all four structural gates pass.
- The fine-run `metrics.json`, `manifest.json`, and `index.html` hash to
  `2a91b2fed8b2833294d9574676aa80c923704b4754824017435617a5f9e30a38`,
  `c321b63c4174fe2b122505dd5d0474f38bf208833a1a3c56db14cb7b4327b502`, and
  `f415c9edac81f17f76608b5bb010e5a2e7f57e8598a3866ff6b9fd68832eed83`.

The machine-readable audit is `audit.json`. Reproduce it with:

```bash
PYTHONPATH=src python scripts/experiments/audit_fit031_error_tail.py \
  runs/janelle_C0001_current_pipeline_20260727 \
  runs/janelle_C0001_error_tail50_min8_20260727 \
  --device cuda:0 \
  --source-patch \
    ara/evidence/fit031-error-only-tail-janelle-2026-07-27/executed-source.patch \
  --output ara/evidence/fit031-error-only-tail-janelle-2026-07-27/audit.json
```

## Disposition

Keep the mechanism available behind `scripts/convert.py --fine-detail` and keep the default off.
The screen confirms a finite, safely improving terminal capacity experiment on the exposed
Janelle image, but it does not answer whether 4,608 extra rows are rate-efficient or whether the
result generalizes. Any promotion needs a preregistered equal-rate or equal-count comparison on
new images with replicated seeds.
