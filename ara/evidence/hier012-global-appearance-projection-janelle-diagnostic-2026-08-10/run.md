# HIER-012 global safeguarded appearance projection — exposed diagnostic

## Evidence status

Descriptive development/attribution evidence only.  Both correlated Janelle C0001/C0004 views
were feasibility-probed before this packaging run, the source tree is dirty, and there was no
distinct prospective reviewer.  The report selects a pipeline for subsequent independent study;
it does not change a default, select Field V2 semantics, complete FIT-046, or establish a rate or
general quality claim.

Portable report:
`results/hier012_global_appearance_projection_janelle_2026-08-10/index.html`.
Manifest SHA-256:
`3abc28551be8c5a58bf4fc3a2ab4dc4acb11b731e11055e6b63f0673d2ea834b`.

## Selected pipeline

Start from the sealed HIER-005 exact-7k field and keep means, log-scales, rotations, alpha,
support, filtering, topology, and row count bit-exact.  Set the matrix-free HIER-010 PCG trainable
mask to all 7,000 RGB rows.  Solve the masked additive normal equations with a `1e-8` pull to the
incoming coefficients, tolerance `1e-6`, at most 48 iterations, and coefficient absolute limit 16.
Step zero remains the exact fallback; a checkpoint can be selected only if raw SSE and displayed
normalized artifact violation are no worse than stage zero.

The report retains HIER-005, HIER-010 touched-only, HIER-011 exchange, exchange-plus-global, and
the selected no-exchange global arm.  This shows whether topology exchange still earns its place
once coefficient scope is no longer artificially restricted.

## Exact-7k outcomes

| image | arm | PSNR | delta vs H005 | masked MSE | MSE reduction | MS-SSIM | LPIPS | pixel max | 7x7 max | PCG |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0001 | HIER-005 | 50.097060 | — | 9.77899e-6 | — | 0.999976337 | 2.81945e-5 | 0.026404 | 0.009518 | 0 |
| C0001 | touched projection | 50.108004 | +0.010944 | 9.75438e-6 | 0.252% | 0.999976754 | 2.79901e-5 | 0.026404 | 0.009469 | 22 |
| C0001 | guarded exchange | 50.536082 | +0.439022 | 8.83877e-6 | 9.615% | 0.999976933 | 2.22112e-5 | **0.013585** | 0.006111 | 0 |
| C0001 | exchange + global | 52.258653 | +2.161593 | 5.94477e-6 | 39.208% | 0.999988854 | **1.22305e-5** | **0.013585** | 0.004765 | 48 |
| C0001 | **global projection** | **52.334526** | **+2.237466** | **5.84181e-6** | **40.262%** | **0.999989331** | 1.42367e-5 | 0.016010 | **0.004586** | 5 |
| C0004 | HIER-005 | 54.374099 | — | 3.65250e-6 | — | 0.999991179 | 7.45235e-6 | 0.014847 | 0.004597 | 0 |
| C0004 | touched projection | 54.378459 | +0.004360 | 3.64883e-6 | 0.100% | 0.999991119 | 7.33783e-6 | 0.014847 | 0.004620 | 16 |
| C0004 | guarded exchange | 54.441007 | +0.066908 | 3.59666e-6 | 1.529% | 0.999991536 | 6.96275e-6 | 0.009335 | 0.003975 | 0 |
| C0004 | exchange + global | 56.455593 | +2.081495 | 2.26173e-6 | 38.077% | **0.999996662** | 2.03230e-6 | **0.008163** | 0.003234 | 8 |
| C0004 | **global projection** | **56.470211** | **+2.096112** | **2.25413e-6** | **38.285%** | **0.999996662** | **2.02287e-6** | **0.008163** | **0.003136** | 22 |

The selected arm clears the declared `+1.5 dB` floor on both views, reduces MSE by
40.262/38.285%, passes both local gates, and preserves every non-RGB array bit-for-bit.  It also
has lower MSE than exchange-plus-global on each view and a 1.04% lower geometric-mean MSE, so the
simpler pipeline wins the frozen development selection.  On C0001, exchange-plus-global retains a
better LPIPS value and isolated pixel maximum; this is an explicit objective tradeoff, not erased
by the MSE selection.

The projection itself adds 0.820/0.683 seconds to the persisted HIER-005 cumulative work in this
run.  These are unequal-work diagnostic timings, not a performance claim.

## Audit and replay

- Both selected cells use all 7,000 RGB rows; selected checkpoints are PCG 5/22.  Coefficient
  absolute maxima are 1.9931/1.1567, below the limit.
- Every selected checkpoint is finite/selectable, lowers stage-zero SSE and displayed normalized
  violation, and matches its persisted row.  Non-RGB arrays are bit-exact.
- Internal/cold renderer parity is at most `1.79e-7`; repeated-render difference is at most
  `1.19e-7`; the formal run records zero relative bilinear-identity error for all four new global
  projection solves (the CUDA feasibility probes were at most `1.05e-7`).
- Independent cold replay checks all ten field file/canonical hashes and exact counts.  Maximum
  drift is `5.16e-7 dB` PSNR, `4.34e-13` MSE, `5.96e-8` SSIM, and `1.26e-8` LPIPS; all displayed
  local metrics reproduce exactly.
- `check_report_bundle.py --allow-dirty` passes, and the focused HIER-010/011/012 slice passes 56
  tests.
- Repository-wide Ruff and every docs/ARA/task/script/workflow structural checker pass.  The
  portable pytest selection, sharded to avoid the execution wrapper's wall-time termination,
  reaches 1,736 passes, 25 skips, and 514 deselections with the same three untouched failures
  recorded by HIER-010: rank-deficient affine condition number, SSP2E CUDA-property availability,
  and SSP2V opened-descriptor path-swap detection.  With CUDA hidden, SSP2E passes and the two
  unchanged baseline failures remain.  `verify.sh` itself stops before tests because this shell has
  standalone Ruff but no importable `python -m ruff`; the equivalent direct Ruff command passes.
- Visual review of full frames and worst crops finds no gross new artifact.  The remaining errors
  are subtle and concentrated on thin hair, garment texture, silhouettes, face, and hands.  Worst
  crop positions differ across arms, so those crops are qualitative rather than registered.

## Interpretation

HIER-010's small gain was caused by its provenance restriction, not by a lack of useful
coefficient degrees of freedom.  At this exact-count additive endpoint, global fixed-geometry RGB
projection removes far more residual than reserving or exchanging topology.  This is strong
mechanism evidence on exposed views and the best observed exact-7k pipeline here, but independent
images, clean execution, distinct review, actual-rate accounting, and the governing Field V2
semantic gates remain mandatory before any broader claim or maintained dispatch.
