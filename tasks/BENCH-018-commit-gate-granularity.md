# BENCH-018: How much should the commit gate cost? (`block_steps` granularity)

## Status

Todo. Needs GPU; no code change required — `--block-steps` is already exposed.

## Context

The transactional gate trials a whole block of `block_steps` optimization steps and rolls it back
if the metric vector regresses. `block_steps` therefore sets the **unit of discarded work**: at the
schedule default of 250, one rejection throws away up to 250 steps.

The BENCH-017 exploratory pass measured the resulting acceptance rates on the full-frame arm:

| phase | attempted | accepted | acceptance |
|---|---:|---:|---:|
| bootstrap | 7,776 | 3,200 | 41.2% |
| coverage_growth | 3,034 | 420 | 13.8% |
| detail_growth | 61,028 | 3,760 | 6.2% |
| redistribution | 26,536 | 1,840 | 6.9% |
| safe_polish | 3,276 | 0 | 0.0% |

`detail_growth` alone is ~56% of wall-clock at 6.2% acceptance. Overall, ~9% of attempted steps
commit.

This is the direct trade-off question: a smaller block discards less per rejection but evaluates
the (expensive) gate metric more often. Nobody has measured where the optimum is, and the schedule
default of 250 was inherited, not chosen against data.

## Goal

Find the `block_steps` that maximizes terminal quality per unit wall-clock, and decide whether the
schedule default should change.

## Approach (preregister before running)

1. **Arms**: `block_steps` in `{25, 50, 100, 250, 500}` at the shipped recipe, everything else
   fixed. Full-frame on Kodak-24 and masked on the dome fixture, >= 3 seeds.
2. **Equal wall-clock envelope is the point** — do not equalize step counts. Report terminal
   quality against *time*, since a smaller block buys fewer wasted steps at the price of more gate
   evaluations.
3. **Report per cell**: PSNR / MS-SSIM / LPIPS, attempted vs accepted steps, gate evaluations,
   fraction of wall-clock spent inside the gate metric, and the rejection-reason histogram.
4. **Frozen gate**: change the default only on a paired win in quality-at-equal-time with a 95% CI
   excluding zero, on both arms.

## Acceptance criteria

- [ ] Preregistered protocol, arms, and gate committed before any target is fitted.
- [ ] Per-cell config/provenance per BENCH-002.
- [ ] Gate evaluation cost isolated and reported, not inferred from total time.
- [ ] Decision recorded in `ara/logic/claims.md`; schedule default changed or explicitly kept.

## Depends on

BENCH-002, BENCH-017, FIT-023, ADR-0025

## Notes

Interacts with FIT-028: relaxing the interior-hole veto changes the rejection rate, which changes
the optimal block size. Run FIT-028 first, or run the two as a factorial if the budget allows —
but do not read a `block_steps` optimum measured under the strict veto as valid after the veto
changes.

Also interacts with FIT-027: the gate metric is dominated by SSIM, so making SSIM cheaper moves
the optimum toward smaller blocks. Measure after FIT-027 lands.
