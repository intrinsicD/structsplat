# HIER-023 unit-gauge continuation diagnostic

## Scope and receipts

This is consumed development evidence on four mechanically selected, historically used DIV2K
images, two seeds, N=640, max-side 160, 500 attempted steps, and one RTX 4090. It is dirty-source
and producer-reviewed, not held-out confirmation or a default/semantic/codec/novelty result.

The first output without suffix is an immutable invalid harness run: continuation cells completed,
but ordinary controls failed row emission because their ledger has no literal step-175 record. It
is excluded from method evidence. The repaired bundle is
`results/hier023_div2k4_s160_n640_i500_s01_diagnostic_rerun1_2026-08-11`.

- manifest: `2d8fd2a12c4f02bd70439d300d7193dd06a1d64b42ba77ecd146875b06f6c13c`
- metrics: `11aff6f0d6800bd4d010f2456cf2f2dbd20f2259c7062270c57e391f7bf17caa`
- decision: `3593dd6201a82b392c5f21ba8695f515ea99f3086397d4e69def09d3fbdc376c`
- report checker: passes with `--allow-dirty`

## Frozen method

The 35/15/50 path performs 175 direct maintained-normalized steps, 75 unit-mass quotient steps
whose final lambda remains positive, and 250 direct maintained-additive steps. Only endpoint
checkpoints compete. The reset arm rebuilds Adam once before step 251; the no-reset arm retains
moments. Ordinary normalized and additive controls use the same initialization, loss, count, and
500-step horizon. The predeclared selector chooses higher mean endpoint PSNR, preferring no-reset
within 0.02 dB.

## Aggregate results

| arm | PSNR | MS-SSIM | LPIPS | pixel max | 7x7 max | PSNR AUC | fit s | calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normalized plain | 29.7498 | 0.981250 | 0.065890 | 0.280133 | 0.095270 | 28.5880 | 0.866 | 522.8 |
| additive plain | 29.0850 | 0.979378 | 0.100209 | 0.300410 | 0.101631 | 26.7802 | 0.805 | 522.1 |
| gauge no reset | 29.0524 | 0.979550 | 0.096634 | 0.296345 | 0.094604 | 27.9852 | 1.387 | 620.0 |
| gauge endpoint reset | 28.9824 | 0.979066 | 0.099216 | 0.305436 | 0.097344 | 27.7665 | 1.414 | 620.0 |

No-reset is selected. Relative to additive it is `-0.03259 dB`, `+0.0001723` MS-SSIM,
`-0.003574` LPIPS, `-0.004065` pixel maximum, `-0.007027` 7x7 maximum, and `+1.20498` PSNR-AUC.
It uses 18.75% more recorded fit renderer calls and 1.724x fit time. One `0343` seed raises LPIPS
`0.012264`, failing the per-cell guard. The positive normalized/additive gap is `0.66477 dB`; the
selected arm is below additive and therefore closes none of it.

## Integrity and visual review

All continuation fields are finite exact lambda-zero N=640 endpoints with no opacity, mass,
denominator, optimizer state, or auxiliary RGB payload. Selected-arm maximum coefficient is
`2.76302`; maximum cold/internal parity is `4.172e-7`. Maximum hold PSNR deviation across both
continuation arms is `0.03442 dB`, and reset telemetry is exact. Native full-frame and worst-crop
review finds no lattice, checker, ringing, black hole, or new wash. All arms show the expected
N=640 fine-detail blur; no-reset is visually similar to additive.

## Disposition

The mechanism gate fails. Unit-gauge normalized pretraining is an efficient additive warm start,
but it does not preserve normalized rendering's fixed-count quality advantage. Resetting moments
is rejected. The bank must not be retuned. The next admissible causal test applies the same
safeguarded fixed-geometry additive RGB solve to plain-additive and gauge geometries on a new data
selection; basis/topology changes follow only if coefficient optimization is not the cause.

## Limits

Historically consumed images, one device, dirty sources, producer review, small resolution/count,
and unequal renderer work prevent confirmation, generality, speed, asymptotic, rate, downstream,
or publication claims.
