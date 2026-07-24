# FIT-023: Transactional safe-schedule candidates

**Status: completed (development).** Checkpoint-only wins this single-image factorial; event color
solve is not promoted. No library-default or generality claim.

## Context

The global mask-contained safe-commit schedule is the strongest observed Janelle C0001 full-field
run, while row-local refinement and pooled boundary variants are weaker. Two older mechanisms have
positive independent evidence but are not yet expressed inside the transactional schedule:

1. FIT-015 showed that an earlier same-count optimizer state can beat the terminal state.
2. FIT-005/010 showed that an exact fixed-geometry color solve can improve convergence and split
   recovery.

The schedule currently evaluates only the terminal state of each optimizer block and runs exact
color solves only before bootstrap and before polish. This task tests both mechanisms without
changing initialization, phase budgets, Gaussian capacity, renderer, or commit metrics.

## Goal

Implement and run a 2x2 development factorial on Janelle frame_00008/C0001:

| arm | Pareto-safe block checkpoints | post-topology color solve |
|---|---|---|
| control | off | off |
| checkpoint | on | off |
| event color | off | on |
| combined | on | on |

Every arm uses the global refinement policy and the same 5,000 -> 11,000 nominal schedule. CUDA
results are source/device-bound single-image development evidence, not a default claim.

## Acceptance criteria

- [x] `fit()` can optionally return state-matched field and optimizer snapshots at a fixed
      cadence without changing default trajectories or checkpoint behavior.
- [x] The safe schedule evaluates those snapshots with its full foreground/boundary/tail/hole/
      containment gate and commits only the best already-safe candidate.
- [x] A topology trial can optionally apply an exact color solve before its final gate; optimizer
      color moments match the selected solved field.
- [x] CLI, resolved config, history, and generated HTML expose both factors and
      selected-checkpoint metadata.
- [x] Tests cover default-off equivalence, earlier safe checkpoint selection, color-solve state
      handling, and argument plumbing.
- [x] Four full equal-budget Janelle arms complete with source/config/artifact provenance and a
      common `index.html`, JSON comparison, and written verdict.
- [x] Results and reader-facing images are committed with the code; any negative arm is preserved.

## Decision rule

The combined arm is only a candidate winner if its full-field foreground MSE, boundary MSE,
CVaR99, interior-hole fraction, boundary-hole fraction, and outside-mask metrics are all nonworse
than the clean control at the recorded precision, and it makes a material improvement in at least
one of them. Runtime is reported separately and can block a performance claim.

## Result

Executed sequentially on one RTX 4090 from clean commit `6e3cf0d`, using the same
frame_00008/C0001 source, seed, 5,000-row initialization, 11,000-row capacity, global policy,
budgets, renderer, and gate in all arms.

| arm | FG / boundary PSNR | CVaR99 / p99 MSE | interior / boundary holes | total |
|---|---:|---:|---:|---:|
| control | 26.566 / 10.878 dB | .172265 / .017764 | 2.544% / 31.690% | 382.5 s |
| checkpoint | **27.068 / 11.397 dB** | .153313 / **.014318** | 1.436% / **28.491%** | 419.3 s |
| event color | 26.464 / 10.794 dB | .175957 / .018526 | 2.806% / 33.413% | 412.2 s |
| combined | 27.063 / 11.388 dB | **.152707** / .014527 | **1.424%** / 28.575% | 541.9 s |

The combined arm passes its preregistered control rule, but checkpoint-only is the recommendation:
it wins foreground, boundary, p99, boundary coverage, and runtime against combined. Four earlier
state-matched checkpoints were committed. Event color was selected in 20/23 topology events in its
two enabled arms, so its negative quality/runtime outcome is not an inactivity result.

All four cold-loaded fields reproduced every audited stored metric with observed delta zero in
the audited environment; accepted-sequence gate failures were zero. CUDA is still scoped as
tolerance-reproducible, not generally bit-exact. Exact outside render/coverage remains zero.
Neither 0.1% interior nor 1% boundary hole targets were reached; `converged` means final
deterministic transaction fixed point, not total coverage.

Artifacts:
`runs/janelle_C0001_transactional_candidates_factorial_20260723/{index.html,comparison.json,report.md,audit.json}`.

## Interfaces touched

`src/structsplat/fit.py`, `src/structsplat/safe_schedule.py`,
`deprecated_scripts/fit_janelle_safe_commit_schedule.py`, comparison/experiment scripts, focused tests,
`README.md`, `tasks/INDEX.md`, and source-bound run artifacts.

## Depends on

FIT-005, FIT-010, FIT-015, FIT-021, CORE-010/011, BENCH-002, ADR-0010/0017/0019.
