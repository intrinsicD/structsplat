# BENCH-017: Screen the full-frame pipeline arm against the plain-fit path

## Status

Todo. Opened by CORE-012 as the missing evidence for the full-frame arm of ADR-0025.

## Context

CORE-012 made the safe-commit schedule reachable for unmasked images by degenerating the mask
machinery (`run_safe_schedule(mask=None)`). That is a mechanism extension with a clean argument
behind it — the commit gate is Pareto-safe, so an accepted step cannot regress the metric vector —
but it is **not** a measured result on ordinary images:

- The schedule's quality evidence (C50/C51/C52) is one masked image, one seed, one RTX 4090.
- The best-evidenced path for unmasked images is still the plain fitter: ABL-006/ADR-0013 for the
  init strategy and C12 for the pinned balanced default at its proxy regime.
- Nobody has run the two against each other at equal budget.

Until this screen runs, ADR-0025's full-frame arm is best-known-by-mechanism only, and the README
and pipeline docstring say so.

## Goal

Decide whether the full-frame arm should be the recommended path for unmasked images, or whether
`python scripts/convert.py` without `--mask`/`--mask-dir` should dispatch to the plain-fit path
instead.

## Approach (preregister before running)

1. **Arms**, all at equal final Gaussian count and equal wall-clock envelope:
   - `pipeline_full_frame`: `run_pipeline(image, mask=None)` at the shipped recipe;
   - `plain_fit_shipped`: `fit()` at the shipped `InitConfig`/`FitConfig` defaults;
   - `plain_fit_best_default`: the C12 pinned recipe, as the strongest known unmasked control.
2. **Targets**: Kodak-24 + COCO4 at the BENCH-002 fair regime, >= 3 seeds. Not the Janelle image —
   the schedule was developed on it, so it cannot also screen it.
3. **Frozen gate**, declared before the first run: the full-frame arm must win paired mean PSNR
   with a 95% CI excluding zero, must not lose MS-SSIM or LPIPS beyond a declared slack, and must
   stay inside a declared total-time multiplier. Equal-count checkpointing is part of the arm, so
   report both terminal and selected states.
4. **Cost note**: the schedule spends its budget on gated trials, so a fair comparison prices
   rejected trials as spent work. Report attempted and accepted steps separately.

## Acceptance criteria

- [ ] Preregistered protocol, arms, and gate committed before any target is fitted.
- [ ] All cells complete with per-row config/provenance (BENCH-002 rules).
- [ ] Frozen gate evaluated once; the decision recorded in `ara/logic/claims.md` either way.
- [ ] ADR-0025 amended with the outcome: the full-frame arm keeps the schedule, or dispatches to
      the plain-fit path.

## Depends on

CORE-012, ADR-0025, BENCH-002, ABL-006, FIT-015

## Notes

A negative result is a perfectly good outcome and must not trigger retuning of the schedule on
these targets: that would spend the screen. If the schedule loses, the honest fix is arm dispatch
in `pipeline.py`, which is a one-function change by construction.
