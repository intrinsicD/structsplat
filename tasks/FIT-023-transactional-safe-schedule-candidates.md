# FIT-023: Transactional safe-schedule candidates

**Status: in progress.** Source-bound Janelle development experiment requested 2026-07-23.

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
- [ ] CLI, resolved config, history, and generated HTML expose both factors and
      selected-checkpoint metadata.
- [x] Tests cover default-off equivalence, earlier safe checkpoint selection, color-solve state
      handling, and argument plumbing.
- [ ] Four full equal-budget Janelle arms complete with source/config/artifact provenance and a
      common `index.html`, JSON comparison, and written verdict.
- [ ] Results and reader-facing images are committed with the code; any negative arm is preserved.

## Decision rule

The combined arm is only a candidate winner if its full-field foreground MSE, boundary MSE,
CVaR99, interior-hole fraction, boundary-hole fraction, and outside-mask metrics are all nonworse
than the clean control at the recorded precision, and it makes a material improvement in at least
one of them. Runtime is reported separately and can block a performance claim.

## Interfaces touched

`src/structsplat/fit.py`, `src/structsplat/safe_schedule.py`,
`scripts/fit_janelle_safe_commit_schedule.py`, comparison/experiment scripts, focused tests,
`README.md`, `tasks/INDEX.md`, and source-bound run artifacts.

## Depends on

FIT-005, FIT-010, FIT-015, FIT-021, CORE-010/011, BENCH-002, ADR-0010/0017/0019.
