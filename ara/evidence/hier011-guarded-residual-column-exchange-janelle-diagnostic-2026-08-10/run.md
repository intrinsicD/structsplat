# HIER-011 guarded residual column exchange — exposed diagnostic

## Evidence status

Diagnostic only.  C0001 is the atom-bank development view; C0004 is correlated, previously
exposed, and opened only after the bank was frozen.  The run used a dirty source tree, one
numerically nondeterministic CUDA trajectory per cell, and no distinct prospective reviewer.  It
can kill or motivate a mechanism but cannot promote a default, semantic, rate, or general quality
claim.

Portable report:
`results/hier011_guarded_residual_column_exchange_janelle_2026-08-10/index.html`.
Manifest SHA-256:
`c15d18ee3b1eca4782c4400e2f94ffe35dca6a0b383ba960c6260d396c849bf9`.

## Question and mechanism

Can HIER-005 reallocate capacity at exact N=7,000 without HIER-010's up-front reserve loss?  The
field is treated as an active set.  Every existing row receives its exact masked deletion price;
residual atoms are one-column least-squares fits at stable high-error sites.  A candidate can enter
only by replacing an unlocked, support-disjoint row whose price it exceeds.  The maintained cold
renderer commits the first ranked pair that strictly lowers raw SSE and individually does not
worsen displayed worst-pixel or worst-7x7 RMSE.  Entering rows are locked.  HIER-010's touched/new-
row PCG is an optional finish.

All bases are the persisted, hash-bound HIER-010 HIER-005 fields, so contraction nondeterminism is
not rerun.  Exact protocol and input hashes are in
`tasks/HIER-011-guarded-residual-column-exchange.md` and the report snapshot.

## Development bank screen

The separate C0001 32-pivot preflight selected the bank by lowest safe SSE exactly as declared:

| bank | final SSE | PSNR | pixel max | 7x7 max | accepted |
|---|---:|---:|---:|---:|---:|
| compact: isotropic 0.18/0.30/0.45 | 0.4413312593 | 50.345451 | 0.014497 | 0.007516 | 32 |
| multiscale: isotropic 0.30/0.45/0.60/0.75 | 0.4375248764 | 50.383070 | 0.014497 | **0.007101** | 32 |
| oriented: multiscale + 0.75x0.30 at four angles | **0.4316094323** | **50.442188** | 0.014497 | 0.007276 | 32 |

The oriented bank was frozen before C0004 execution.  This screen is not part of the frozen
two-view report and is not transfer evidence.

## Frozen outcomes

All eight cells retain exactly 7,000 rows.

| image | arm | PSNR | delta vs H005 | masked MSE | MSE reduction | MS-SSIM | LPIPS | pixel max | 7x7 max | swaps | PCG |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0001 | HIER-005 | 50.097060 | — | 9.77899e-6 | — | 0.999976337 | 2.81948e-5 | 0.026404 | 0.009518 | 0 | 0 |
| C0001 | touched projection | 50.108004 | +0.010944 | 9.75438e-6 | 0.252% | 0.999976754 | 2.79887e-5 | 0.026404 | 0.009469 | 0 | 22 |
| C0001 | guarded exchange | 50.536082 | +0.439022 | 8.83877e-6 | 9.615% | 0.999976933 | 2.22149e-5 | 0.013585 | 0.006111 | 68 | 0 |
| C0001 | exchange + projection | **50.638658** | **+0.541598** | **8.63245e-6** | **11.725%** | **0.999978781** | **2.16660e-5** | **0.013585** | **0.005999** | 68 | 47 |
| C0004 | HIER-005 | 54.374098 | — | 3.65250e-6 | — | 0.999991179 | 7.45310e-6 | 0.014847 | 0.004597 | 0 | 0 |
| C0004 | touched projection | 54.378459 | +0.004361 | 3.64883e-6 | 0.100% | 0.999991119 | 7.33756e-6 | 0.014847 | 0.004620 | 0 | 16 |
| C0004 | guarded exchange | 54.441007 | +0.066908 | 3.59666e-6 | 1.529% | **0.999991536** | 6.96297e-6 | 0.009335 | **0.003975** | 5 | 0 |
| C0004 | exchange + projection | **54.453962** | **+0.079863** | **3.58595e-6** | **1.822%** | **0.999991536** | **6.83372e-6** | **0.009335** | 0.003988 | 5 | 2 |

The search exhausts negative-cost pairs after 68/5 commits; the 128-pivot cap does not bind.
Every accepted step is monotone in SSE and both displayed local maxima.  C0001's initially failing
pixel maximum falls from 0.026404 to 0.013585, so both final cells pass the artifact gate.

The frozen full-mechanism rule nevertheless fails: C0004's `+0.079863 dB` is below the declared
`+0.10 dB` material-gain floor.  HIER-005 remains unchanged.  The result motivated HIER-012's
coefficient-scope successor; it does not authorize retuning this consumed view.

## Audit and replay

- `check_report_bundle.py --allow-dirty` passes the complete portable bundle.
- Independent cold replay checks all eight field file/canonical hashes and exact counts.  Maximum
  drift is `2.26e-7 dB` PSNR, `2.16e-13` MSE, `5.96e-8` MS-SSIM, and `9.65e-9` LPIPS; all local
  metrics reproduce exactly.
- Maximum repeated-render difference is `1.19e-7`.  Every exchange row is unique and the complete
  persisted trajectory is monotone in SSE, displayed pixel max, and displayed 7x7 max.
- The shared HIER-010/011 focused slice passes 56 tests after the successor work.
- Repository-wide Ruff passes.  The portable pytest selection, sharded to avoid the execution
  wrapper's wall-time termination, reaches 1,736 passes, 25 skips, and 514 deselections with the
  same three untouched failures recorded by HIER-010: rank-deficient affine condition number,
  SSP2E CUDA-property availability, and SSP2V opened-descriptor path-swap detection.  Hiding CUDA
  removes SSP2E; the other two remain.  `verify.sh` itself cannot enter the gate in this shell
  because it invokes `python -m ruff` while Ruff is installed only as a standalone executable.
- Visual inspection finds no gross new artifact; residual changes remain concentrated on garment
  texture, silhouettes, hair, face, and hands.  Per-arm worst crops move and are qualitative, not
  registered comparisons.
