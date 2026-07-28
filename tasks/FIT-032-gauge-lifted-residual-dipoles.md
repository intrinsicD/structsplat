# FIT-032: Gauge-lifted residual dipoles

## Status

Completed negative on the exposed current-pipeline Janelle state. Gauge equivalence and the
finite-difference fixture pass, but dipoles lose the frozen promotion gate at all three budgets;
no production or default change is authorized.

## Context

FIT-031's tail improved its own pre-tail state by `0.522239 dB`, but spent 4,608 rows, took
about `2.1x` the clean run's wall-clock, and has no equal-count causal comparator. A
native-resolution residual audit also finds its top sites strongly clustered: 73.48% of the top
4,608 touch another selected pixel and they occupy only 1,303 distinct 8x8 cells. Its
effective-support estimate is resolution-dependent (`14,177` at max-side 1,200 versus about
`258,218` on the saved native reconstruction).

The normalized renderer has an exact gauge: replace one opacity mass by two co-located
half-opacity rows with identical geometry and color and the render is unchanged. Move the
children oppositely while giving them opposite color contrasts and the pair creates a localized
derivative-of-Gaussian detail mode for one net row.

## Goal

Test whether residual-conditioned gauge-lifted color dipoles reduce Janelle fine-detail error
substantially more per added row than the strongest ordinary-Gaussian control.

## Acceptance criteria

- [x] Implement only under `benchmarks/` and `scripts/experiments/`; leave the pipeline,
      defaults, and format untouched.
- [x] Verify co-located half-opacity render equivalence within `2e-6` absolute error.
- [x] Verify the analytic mode against centered finite differences on a deterministic fixture.
- [x] Use a deterministic local rank-one residual solve, finite trust region, spatial
      deduplication, and mask containment.
- [x] Compare equal net-row budgets `{32, 64, 128}` on the saved exposed Janelle state against
      FIT-031-style error-ranked isotropic births and moment-preserving splits under identical
      renderer, objective, and recovery work.
- [x] Log immediate/recovered foreground MSE and PSNR, tail and boundary metrics, containment,
      runtime, row count, source/device identity, and gain per added row.
- [x] Promote only if the dipole obtains at least `2x` the immediate and `1.5x` the recovered
      foreground MSE reduction per row of the strongest control at two of three budgets, with no
      protected regression. Otherwise close it as negative.
- [x] Audit the result with `structsplat-results-audit` before recording a claim.

## Depends on

FIT-007/017/019/020/025/031, BENCH-002, CORE-012, ADR-0029.

## Prior-art and negative-lineage controls

SteepGS and splitting steepest descent are the closest geometry controls. WIPES, GStex, and
adaptive per-primitive texturing occupy the explicit frequency-carrier alternative. FIT-019
confirmed opacity-split gauge equivalence but rejected its recovery allocator; this task tests
an antisymmetric color/displacement mode, not a reinterpretation of that result. FIT-017/020 and
BENCH-009/011--015 remain closed and are not retuned.

## Notes

Either displacement or color contrast alone has zero first derivative at the symmetric split;
the useful term is bilinear. Both are therefore initialized from a closed-form residual
projection rather than relying on Adam to escape the symmetric saddle. This is a local
development hypothesis, not a novelty, compression, convergence, or general-quality claim.

## Result

Dipoles passed their mechanism checks but passed `0/3` empirical budgets. At 128 net rows their
immediate foreground-MSE reduction was `2.25e-5` versus `2.77e-4` for the strongest ordinary
control, and recovered reduction was not better. Deep high-pass movement stayed about
`0.13--0.16%`. Close this primitive variation as negative for the requested detail objective.
