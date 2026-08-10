# FIT-029: Decide whether `safe_polish` earns its wall-clock

## Status

Todo. **Blocked on FIT-028** by design — do not act on this task first.

## Context

In the BENCH-017 exploratory pass, `safe_polish` committed **0 of 3,276 attempted steps across 7
of 7 images** (full-frame arm, Kodak-24, capacity 5,000 and 11,000, seed 0). It costs roughly 3%
of fit wall-clock and, on this evidence, produced nothing on those images.

The phase is not obviously broken. Its tolerances are the tightest in the schedule
`(5e-3, 3e-3, 1e-3, 3e-3, 1e-3)` with `minimum_relative_gain` thresholds of `0.002 / 0.005`, so it
demands a real improvement while permitting almost no regression. That is the intended design of a
polish phase. But it shares the *unconditional* interior-hole veto with every other phase, and all
7 of its rejections were `interior_holes_regressed` with every pixel-error metric improving.

So the observed 0% acceptance has at least two candidate causes:

1. the interior-hole veto (ADR-0026) — in which case FIT-028's budget may revive the phase and
   nothing here needs changing;
2. tolerances so tight that late-fit gains cannot clear `minimum_relative_gain` — in which case
   the phase is miscalibrated rather than vetoed.

**Cutting the phase before separating these would be treating a symptom.** A phase that accepts
nothing because a shared veto blocks it is a different defect from a phase whose own thresholds
are wrong, and only the second is a reason to remove or retune it.

## Goal

Decide: keep `safe_polish` as-is, retune its tolerances, or remove it from the schedule.

## Approach

1. Re-run the FIT-028 arms and read `safe_polish` acceptance per budget value. If a nonzero
   interior budget restores nonzero acceptance, the cause is (1) and this task closes as
   "no change; superseded by ADR-0026".
2. If acceptance stays at 0% even with the veto relaxed, instrument which specific term rejects
   and whether `minimum_relative_gain` is the binding constraint.
3. Only then consider removal, and price it honestly: removing a phase that accepts nothing is a
   pure wall-clock win, but the phase may accept on the *masked* arm, where its tolerances were
   developed (FIT-023, Janelle). Measure the masked arm before removing anything.

## Acceptance criteria

- [x] `safe_polish` acceptance reported per FIT-028 arm, masked arm. *(0/1,404, 0/1,404, 31/1,872,
      0/1,404 at `0.0/1e-4/5e-4/2e-3`.)*
- [ ] Same, full-frame arm — blocked with FIT-028's full-frame arm, which has not run.
- [x] The cause identified as veto-driven or tolerance-driven, with the evidence cited.
      *(Tolerance-driven on the masked arm; `C65`.)*
- [x] Decision recorded: kept / retuned / removed, with a claim row either way. *(**Kept.**)*
- [x] If removed, ADR-0025's recipe and the schedule defaults change in one commit. *(Not removed;
      no recipe or default change.)*

## Depends on

FIT-028, ADR-0026, FIT-023, BENCH-017

## Notes

The 0-of-3,276 figure is from the unmasked arm only and is staged as O88, not claimed. The masked
arm has not been measured for this at all.

## Masked-arm disposition — 2026-08-08: keep the phase

Answered from the FIT-028 grid at no extra compute, which is why this task was blocked on FIT-028 by
design. Of the two candidate causes, the evidence separates them cleanly and selects **(2)
tolerance-driven**:

- Every `safe_polish` rejection in all 12 cells cites both `boundary_mse_regressed` and
  `cvar99_mse_regressed` — the phase's own tightest-in-the-schedule pixel-error tolerances.
- `interior_holes_regressed` appears in only 2 of 12 cells and **never alone**, so cause (1), the
  shared interior-hole veto, is not what stops the phase here.
- `no_material_gain` fires in 3 cells, so `minimum_relative_gain` binds directly.
- `budget5e4` seed 1 accepted **31 of 936** attempted steps — the first nonzero `safe_polish`
  acceptance on record, against a prior 0 of 3,276 across 7/7 unmasked images.

This is the masked measurement step 3 demanded before removing anything. The phase accepts there, so
**removal is not indicated** and nothing is cut. Retuning `minimum_relative_gain` and the polish
pixel-error tolerances is the indicated follow-up, is a new question rather than this task's, and must
not be tuned on this consumed frame.

Caveat: `safe_polish` receives only 1--2 blocks and 468--936 attempted steps per cell, so this is a
consistent pattern over small per-cell samples, not a precise acceptance rate.

Evidence `ara/evidence/fit028-hole-budget-janelle-2026-08-08/run.md`; claim `C65`; trace `N265`;
staging `O139`.

## Agent workflow

- Driver: claude-root
- Reviewer: pending-distinct
- Turn: driver
- Reviewed revision: pending

### Handoff log

The masked-arm answer is a development diagnostic from the FIT-028 grid: exposed single image, three
seeds, provisional self-review only. It supports keeping the phase and forbids removing it on current
evidence; it cannot authorize a tolerance change or close the full-frame half.
