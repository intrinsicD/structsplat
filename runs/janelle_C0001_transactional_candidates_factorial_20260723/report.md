# FIT-023 — Janelle transactional-candidate factorial

## Verdict

For the current Janelle development pipeline, enable **Pareto-safe block checkpoints** at a
50-step cadence and leave **post-topology event color solves disabled**.

The checkpoint-only arm is the best quality/performance compromise in this source-bound run. It
improves every protected quality metric over the clean control and is the strongest arm on
foreground MSE, boundary MSE, p99 MSE, and boundary undercoverage. The combined arm has slightly
better CVaR99 and interior undercoverage, but regresses the other four non-tied quality metrics
against checkpoint-only and costs 29.3% more total time. Event color solve alone is worse than
control on every measured error/coverage metric and is 7.8% slower end to end.

This is a one-image, one-seed RTX-4090 development result. It supports an opt-in Janelle
recommendation, not a repository-wide default or general convergence claim.

## Controlled design

All four arms use frame `frame_00008/C0001`, source seed `1559856117`, the same 4,500
quadtree-WSE + 500 explicit-boundary initialization, global refinement, identical phase/event
budgets, capacity 11,000, `cuda_tiled`, and the same full-resolution safe-commit gate.

| arm | Pareto checkpoints | event color solve |
|---|---:|---:|
| control | off | off |
| checkpoint | on, every 50 steps | off |
| event color | off | on |
| combined | on, every 50 steps | on |

The run was executed sequentially on one NVIDIA GeForce RTX 4090 from clean commit
`6e3cf0d1836bebeaa1c2a72227d238a53d876445`. Torch was `2.7.0+cu126`; CUDA runtime was
12.6. Every arm preserved identical source/config/environment hashes after removing only the two
factor booleans.

## Final results

Lower is better for MSE, CVaR, p99, holes, and time.

| arm | FG PSNR | boundary PSNR | CVaR99 MSE | p99 MSE | interior holes | boundary holes | total seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 26.566 dB | 10.878 dB | 0.172265 | 0.017764 | 2.544% | 31.690% | 382.5 |
| **checkpoint** | **27.068 dB** | **11.397 dB** | 0.153313 | **0.014318** | 1.436% | **28.491%** | 419.3 |
| event color | 26.464 dB | 10.794 dB | 0.175957 | 0.018526 | 2.806% | 33.413% | 412.2 |
| combined | 27.063 dB | 11.388 dB | **0.152707** | 0.014527 | **1.424%** | 28.575% | 541.9 |

Checkpoint-only versus control:

- foreground/boundary MSE: -10.92% / -11.27%;
- CVaR99/p99 MSE: -11.00% / -19.40%;
- interior/boundary undercoverage: -43.57% / -10.10% relative;
- total time: +9.61%.

Combined versus checkpoint-only:

- foreground/boundary MSE: +0.13% / +0.21% (worse);
- CVaR99: -0.40% (better), p99: +1.46% (worse);
- interior holes: -0.012 percentage points, boundary holes: +0.084 percentage points;
- total time: +29.26%, schedule time: +43.42%.

The preregistered combined-versus-control rule passes, but that rule only establishes that the
combined mechanism is viable relative to control. It does not make combined the winner once the
checkpoint-only arm is considered.

## Mechanism audit

Checkpoint-only committed four genuinely earlier state-matched snapshots:

1. bootstrap global fit at step 50 of a rejected/backtracked horizon;
2. one coverage-birth recovery at step 50 rather than its step-80 terminal state;
3. one redistribution coverage-birth recovery at step 50;
4. the safe-polish global fit at step 50.

The selected field and its Adam moments came from the same snapshot. The final checkpoint arm
contains 12 accepted coverage births, five detail births, two boundary births, one
merge→rebirth, one prune→rebirth, and five accepted global-fit blocks.

Event color solve was not merely inactive: it was selected in 20 topology commits by the
event-only arm and 23 by the combined arm. Its negative result is therefore informative. Fewer
optimizer attempts did not translate into speed: the exact color solves made the event-only arm
slower than control, and made combined substantially slower than checkpoint-only.

Every accepted non-marker transition in all four arms independently re-passes the full
foreground/boundary/CVaR/hole/outside gate against the preceding accepted state. No accepted
transition destroys previously committed quality within the configured numerical tolerances.

## Coverage and visual interpretation

The fixed-scale error images agree with the numerical result: checkpoint and combined suppress
more silhouette and high-texture residual than control/event-color, while checkpoint and combined
are visually very close. Exact support containment remains intact: rendered color and raw
coverage are both exactly zero outside the mask in every final arm.

However, **total coverage was not achieved**. The checkpoint arm ends at 1.436% reachable-interior
undercoverage and 28.491% reachable-boundary undercoverage, versus configured targets of 0.1% and
1%. `converged=true` in these artifacts means the final safe-polish transaction reached a
deterministic fixed point; it does not mean the coverage targets were met. Boundary closure and
total coverage therefore remain open method problems.

## Adversarial audit

- Four of four arms completed; all field and summary hashes match their manifests.
- Every arm records the same clean source commit and empty git status.
- Cold loading every final `.npz` and recomputing the eight audited metrics produced observed
  delta 0 for every value in this environment. CUDA replay is still treated as
  tolerance-reproducible rather than generally bit-exact.
- Accepted/attempted step sums match the schedule totals; accepted-sequence gate failures: 0.
- All four native reconstruction/error pairs are RGB PNGs at 3964×1444.
- All HTML links resolve after this report is present.
- The current cache ELF used by cold replay hashes to
  `843b37b6046b7eeaae2fdc7c32086c6f4ab428e8428257b25f2eae123c8718d2`, but the per-arm
  runner did not archive or manifest that binary. Clean source snapshots and cold replay are
  preserved; exact per-process ELF identity is not claimed.
- `--no-archive` was used. These runs make no `.rtgsv`, byte-cap, actual-rate, codec, or cold-decode
  claim.

Machine-readable details are in `audit.json`, `comparison.json`, each arm's `run_config.json`,
`schedule_history.json`, and final full-field NPZ.

## Recommended command

```bash
PYTHONPATH=src /home/alex/miniconda3/bin/python \
  scripts/fit_janelle_safe_commit_schedule.py \
  --capture-root /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric \
  --realtime-root /home/alex/Documents/realtime-gs \
  --frame frame_00008 --view-id C0001 --device cuda:0 \
  --pareto-safe-checkpoints --pareto-checkpoint-every 50 --no-archive \
  --out runs/janelle_C0001_safe_commit_pareto_checkpoint
```

Exact factorial rerun:

```bash
PYTHONPATH=src /home/alex/miniconda3/bin/python \
  scripts/run_janelle_safe_schedule_factorial.py \
  --out runs/janelle_C0001_transactional_candidates_factorial_20260723
```

Before any library-default promotion, repeat the frozen checkpoint-vs-control comparison across
multiple masked images and seeds. Keep event color solve as an experimental switch; the present
result does not justify reviving it in the recommended schedule.

## Claim disposition

| claim | kind and scope | disposition |
|---|---|---|
| State checkpoints preserve the default trajectory and pair fields with matching Adam moments. | tested plumbing | confirm |
| Checkpoint-only improves the current Janelle image over matched control. | measured single-image development | confirm |
| Event color solve improves quality or end-to-end speed. | measured single-image development | refute |
| Combined should replace checkpoint-only. | measured single-image development | refute for this run |
| Checkpoint-only should become the repository default. | asserted general/production | not authorized |
| The schedule achieved total mask coverage. | measured single-image development | refute |
| The run establishes byte-cap, codec, or actual-rate quality. | compression | not claimed; archive disabled |

GPU execution comprised the smoke run, four full arms, and cold field rescoring on the RTX 4090.
The focused/relevant test suite passed 189 tests and Ruff passed; the repository-wide 1,391-test
suite was not executed. Visual conclusions are inspection of the persisted fixed-scale native
error images. No held-out mask set or multi-seed confirmation was opened.
