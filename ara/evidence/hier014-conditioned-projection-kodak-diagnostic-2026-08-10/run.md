# HIER-014 conditioned projection on Kodak — development diagnostic

## Evidence status

Negative, dirty-source development evidence.  Four SHA-bound Kodak images were selected before
execution and evaluated at max-side 512, exact N=7,000, seed 0, and `cuda_additive`.  No distinct
prospective reviewer participated.  This can reject the fixed-geometry numerical hypothesis; it
cannot establish general performance, select Field V2 semantics, complete FIT-046, or change a
default.

Portable report:
`results/hier014_kodak_conditioned_projection_2026-08-10/index.html`.
Manifest SHA-256:
`c547e987a6fd72e5ebf65ae1401fbf2a74cd0750a25615a3946b2ce81927592b`.

## Protocol and command

The exact source/config bindings are frozen in
`tasks/HIER-014-conditioned-minimum-norm-projection.md` and the report's `config.json`.  Arms are
HIER-005, legacy input-centered/subtractive projection, origin/zero-centered projection with the
subtractive base, and the same origin solve with an explicit frozen base.  Every solve uses ridge
`1e-8`, relative normal tolerance `1e-6`, at most 96 PCG iterations, and coefficient limit 16.

```bash
PYTHONPATH=src python scripts/experiments/hier014_conditioned_projection.py \
  --phase kodak --images /home/alex/Documents/datasets/kodak24 \
  --out results/hier014_kodak_conditioned_projection_2026-08-10
```

Environment: repository revision `91ba376b4048ada69661e20af5b13024368456d2` on dirty `main`;
Python 3.12.9, NumPy 2.1.3, torch 2.9.0+cu128, CUDA 12.8, NVIDIA RTX 3050.  Total elapsed time was
563.25 seconds.

## Outcome

| arm | geometric-mean MSE ratio | mean PSNR delta | mean MS-SSIM delta | mean LPIPS delta | nonzero solves |
|---|---:|---:|---:|---:|---:|
| legacy input/subtractive | 0.9978106440 | +0.009519 dB | +0.00072995 | +0.00092378 | 1/4 |
| origin/subtractive | 0.9927876659 | +0.031436 dB | +0.00155303 | +0.00103495 | 3/4 |
| origin/explicit | 0.9927876691 | +0.031436 dB | +0.00155304 | +0.00103469 | 3/4 |

Origin restart reconditions `kodim07` and `kodim19` from unsafe incoming coefficients to maxima
about `5.09` and `7.81`; `kodim01` was already bounded.  `kodim13` has bounded lower-SSE iterates,
but all violate the frozen displayed-local transaction, so the exact stage-zero field with maximum
`84.70` is correctly retained.  `kodim19` improves PSNR `0.0464 dB` while worsening LPIPS and the
7x7 maximum.  Explicit and subtractive bases have indistinguishable aggregates, ruling out
subtractive cancellation as the material cause at this precision.

The frozen gate fails coefficient, nonzero-solve, parity, 7x7-local, 10%-MSE, and LPIPS clauses.
Phase C on the consumed HIER-013 repository bank was therefore not run.  The interpretation is
narrow: minimum-norm restart fixes much of the numerical range problem, but the fixed contracted
geometry/basis remains the quality bottleneck.

## Audit

Independent replay verified all 189 manifest entries by byte count and SHA-256, all 16 unique
image/arm rows, exact JSON/JSONL agreement, exact N=7,000 counts, source hashes, and non-RGB array
identity.  Aggregate ratios/deltas above were independently recomputed from `metrics.json`.
`check_report_bundle.py` does not yet recognize this task-specific schema and falls through to the
maintained workflow schema; this is a report-format limitation, not a passed bundle gate.  The run
remains explicitly non-claim and no artifact was repaired after outcome access.
