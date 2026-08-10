# FIT-028: Measure the interior coverage budget and decide its default

## Status

Mechanism implemented (ADR-0026), default `0.0` — behaviour unchanged. **The measurement below
has not run.** No claim is authorized until it does.

## Context

ADR-0026 records the finding that motivates this task: in the BENCH-017 exploratory pass,
`interior_holes_regressed` accounted for **82 of 110 rejected blocks (75%)**, and `safe_polish`
committed **0 of 3,276** attempted steps across 7 of 7 images while every pixel-error metric
improved in every rejected block.

That is a diagnosis of *waste*, not evidence that spending coverage is *better*. A block whose
pixel error improves while interior holes grow may be a genuine improvement the strict gate
refuses, or it may be the optimizer discovering that deleting coverage is a cheap way to reduce
mean error. Those two are indistinguishable in the rejection log and are distinguished only by
measuring terminal quality at a fixed budget.

## Goal

Decide whether `hole_regression_budget` should be nonzero in the ADR-0025 recipe, and if so, at
what value and for which arm.

## Approach (preregister before running)

1. **Arms**: `hole_regression_budget` in `{0.0, 1e-4, 5e-4, 2e-3}`, everything else at the shipped
   recipe. Both arms of ADR-0025 — full-frame on Kodak-24, masked on the dome fixture — because
   the interior/boundary distinction only exists in the masked case.
2. **Equal budget**: same `capacity`, same `step_scale`, same seed set (>= 3 seeds). Report
   terminal *and* Pareto-selected state.
3. **Report per cell**: PSNR / MS-SSIM / LPIPS, terminal `interior_hole_fraction`, attempted and
   accepted steps, wall-clock, and the rejection-reason histogram.
4. **Frozen gate**, declared before the first fit: a nonzero budget is promoted only if it wins
   mean PSNR with a 95% CI excluding zero, does not lose MS-SSIM or LPIPS beyond a declared slack,
   **and** terminal `interior_hole_fraction` stays below a declared absolute ceiling. Recovering
   wasted steps is not by itself a reason to promote — the terminal image is the deliverable.

## Acceptance criteria

- [x] Preregistered protocol, arms, and gate committed before any target is fitted. *(Masked arm,
      2026-08-08. The full-frame Kodak-24 arm is not preregistered and has not run.)*
- [x] All cells complete with per-row config/provenance (BENCH-002 rules). *(Masked arm: 12/12 cells,
      no error cells. Provenance caveat: the default report gate fails on config-vs-manifest
      repository divergence caused by concurrent doc edits; executed modules were frozen throughout.)*
- [x] Frozen gate evaluated once; the decision recorded in `ara/logic/claims.md` either way.
      *(Masked arm: gate not met, recorded as refuted `C64`.)*
- [x] If promoted: `pipeline.py` `RECIPE` and `PipelineConfig` updated in one commit, citing the
      claim, per the ADR-0025 update rule. If not: ADR-0026 amended to say the knob stays off.
      *(Not promoted; ADR-0026 carries the dated masked-arm Outcome section.)*
- [ ] Full-frame Kodak-24 arm preregistered, executed, and gated; distinct review obtained before any
      cross-arm or general conclusion.

## Frozen masked-arm development protocol (2026-08-08)

This freezes **only the masked arm**. The full-frame Kodak-24 arm and the promotion gate above are
untouched and still required before any default changes. There is no distinct prospective reviewer
for this run, so it is a **development diagnostic**: it may inform pipeline design and it may kill
the knob, but on its own it cannot promote `hole_regression_budget` to a nonzero default.

- **Question:** on the masked arm, does a nonzero interior coverage budget convert vetoed blocks
  into terminal quality, or does it only let the optimizer buy mean error with coverage?
- **Data:** `2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg` with
  `mask/mask_C0001.png`, the source-bound Janelle development image of FIT-023/C56/C60. Exposed
  development data in one capture group; not held out, not confirmation.
- **Regime:** `--max-side 1200` (the established `1200x1038` Janelle regime), shipped recipe,
  capacity 11,000, `quadtree_wse`, `mask_margin` 0.75, exact CUDA renderer, RTX 3050.
- **Arms:** stage `hole_budget`, variants `current` (0.0), `budget1e4` (1e-4), `budget5e4` (5e-4),
  `budget2e3` (2e-3). Nothing else moves; the registered transform touches exactly one schedule
  field and every phase budget is asserted unchanged by test.
- **Seeds:** 0, 1, 2. Pairing key is (image, seed).
- **Primary response:** terminal foreground PSNR. **Guardrails:** MS-SSIM, LPIPS, terminal
  `interior_hole_fraction`. **Mechanism telemetry:** per-phase attempted/accepted steps and the
  rejection-reason histogram, from the `gate_telemetry` surface added for this run.
- **Reading rule, declared before outcomes:** recovering vetoed steps is *not* a result. A budget
  is interesting only if terminal PSNR improves without an MS-SSIM/LPIPS loss **and** terminal
  `interior_hole_fraction` does not rise above the `current` arm's value. If PSNR rises while holes
  rise, that is the coverage-selling failure mode this task was opened to detect, and it is
  recorded as such rather than promoted.
- **Exact command:**

```bash
python scripts/stage_search.py IMAGES OUTDIR --mask-dir MASKS \
  --stage hole_budget --seeds 0 1 2 --max-side 1200 --lpips
```

- **Forbidden follow-ups:** retuning tolerances, `minimum_relative_gain`, `coverage_tau`, or the
  budget ladder on this frame after seeing outcomes; promoting a default from the masked arm alone.

## Depends on

ADR-0026, ADR-0025, BENCH-002, BENCH-017, FIT-023

## Notes

The observation that motivated this is `ara/staging/observations.yaml` O87, which is staging, not
a claim. Do not cite the 75% figure as evidence *for* the budget — it is evidence that the gate
discards work, which is the reason to run the experiment, not its result.

A negative result is a good outcome and would sharpen FIT-030: if buying pixel error with
coverage loses at every budget, then coverage is not the thing to relax and the real fix is
proposal-time repair.

## Agent workflow

- Driver: claude-root
- Reviewer: pending-distinct
- Turn: driver
- Reviewed revision: pending

### Handoff log

The masked-arm protocol above was frozen on 2026-08-08 before any target was fitted, but no
distinct prospective reviewer was available to approve its digest. The run is therefore a
development diagnostic under `structsplat-experiment`'s first classification: it can inform
pipeline design and it can kill a knob, but it cannot promote a default or close the promotion
gate, which still requires the full-frame Kodak-24 arm and distinct review.

## Masked-arm outcome — 2026-08-08

**The gate is not met and the knob stays off.** Paired deltas versus the strict `0.0` control at
`1e-4/5e-4/2e-3` are `+0.0865 / -0.0683 / +0.2116 dB` PSNR, every 95% t interval at n=3 contains
zero, and the response is non-monotonic. `budget5e4` fit seconds (`+93.3`) and `budget2e3` LPIPS
(`-0.00084`) are nominally significant in isolation but fail Bonferroni over the 15 comparisons.

The mechanism is not at fault: acceptance rises monotonically `8.71 -> 9.26 -> 9.53 -> 10.48%` and
interior-hole block citations collapse `63 -> 18 -> 6 -> 0`. But rejected blocks hold at
`73 -> 68 -> 72 -> 72`, and CVaR99 sole-cause rejections rise `6 -> 39`. Substitution is complete;
only 4 of 73 baseline rejections were hole-vetoed alone, which bounded the recoverable set in advance.

The declared reading rule decided the one ambiguous arm. `budget2e3` has the best PSNR point estimate
and the only nominal LPIPS gain, and it is the only arm retaining terminal interior holes
(`0.00131%` versus `0.00000%`) — the coverage-selling failure mode this task was opened to detect.

Design consequence: interior coverage is not the lever on the masked arm; the CVaR99 tail guard and
boundary pixel-error terms are. This sharpens FIT-030 exactly as the task's Notes anticipated.

Evidence `ara/evidence/fit028-hole-budget-janelle-2026-08-08/run.md`; claims `C64`/`C65`; trace
`N263`/`N264`/`N265`; staging `O138`/`O139`; ADR-0026 Outcome section.

## Replication-envelope addendum — 2026-08-08

BENCH-018 subsequently re-ran this task's baseline configuration at the same three seeds, producing
three same-config replicate pairs with PSNR sd `0.185 dB` and Gaussian-count differences up to 944
rows, on identical target pixels (`C67`). Seed does not pin the trajectory.

Consequently the 95% detection floor at n=3 is `~0.46 dB`, and **every FIT-028 quality estimate
above is below it** (largest `+0.2116`). The negative conclusion is unaffected and better stated:
not "the budget has no effect" but **"no effect larger than ~0.46 dB, with mechanism-level evidence
showing why none is expected"** — the acceptance ladder, the `63 -> 0` citation collapse against flat
rejected-block counts, and the 4-of-73 sole-cause bound. Those mechanism numbers are large,
monotonic, three-seed-consistent, and unaffected by the envelope; the quality nulls never carried the
conclusion.
