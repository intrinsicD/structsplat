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

- [ ] `safe_polish` acceptance reported per FIT-028 arm, both ADR-0025 arms.
- [ ] The cause identified as veto-driven or tolerance-driven, with the evidence cited.
- [ ] Decision recorded: kept / retuned / removed, with a claim row either way.
- [ ] If removed, ADR-0025's recipe and the schedule defaults change in one commit.

## Depends on

FIT-028, ADR-0026, FIT-023, BENCH-017

## Notes

The 0-of-3,276 figure is from the unmasked arm only and is staged as O88, not claimed. The masked
arm has not been measured for this at all.
