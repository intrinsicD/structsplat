# FIT-044 — Stage-wise parameter-group activation schedules

## Context

The fitter already exposes five per-parameter groups (means, scales, rotation, color, opacity)
with independent learning rates (`FitConfig.lr_means/lr_scales/lr_rot/lr_color/lr_opacity` via
`field.parameter_groups`), a global LR shape (`_lr_factor`), Adam/AdamW, loss warm-up, and
convergence instrumentation (PSNR history, iterations-to-target). Every group is optimized
jointly from iteration 0. The variables are strongly coupled — position compensates scale, scale
compensates opacity, opacity compensates color, anisotropy compensates position — so joint
optimization can erase the structured initialization before its value is measurable. FIT-016
gated the *loss target* coarse-to-fine and was rejected by its 500-step guard; it never tested
gating *which parameter groups move*. That is the direct test of whether structure-aware
initialization carries early-convergence value or is merely overwritten.

## Goal

An opt-in `parameter_schedule` in `FitConfig` — implemented purely as per-group learning-rate
multipliers (0/1 gating by iteration fraction) without reconstructing the optimizer — screened
against the joint baseline: does any staged activation schedule reach target PSNR in fewer
iterations, or preserve more of the structured initialization, at equal final quality and budget?

Screened schedules (phase boundaries as configurable fractions of the nominal schedule):

- `joint` — baseline, all groups active from 0 (must be a bit-for-bit no-op).
- `geometry_first` — 0–15%: means+scales+rotation only; 15–30%: +opacity; 30–100%: all.
- `appearance_first` — 0–10%: color+opacity only (geometry fixed at structured init);
  10–30%: +means; 30–100%: all.
- `progressive_anisotropy` — 0–20%: isotropic scales, frozen rotation; 20–40%: anisotropic
  scales released; 40–100%: all.

## Non-goals

- Changing any default (`joint` remains the shipped behavior).
- Loss-target curricula (FIT-016 owns that rejected axis) or pyramid schedule redesign.
- Per-field optimizer dynamics (field-specific beta2/weight-decay is a separate follow-up task).
- Topology/densification policy changes; schedules interact with densification only through the
  existing event triggers, which stay frozen across arms.

## Acceptance criteria

- [ ] `FitConfig.parameter_schedule` (+ phase-fraction fields) is opt-in; `joint` reproduces the
      current fitter trajectory exactly on a seeded regression test.
- [ ] Gating is implemented as per-group LR multipliers on the existing optimizer; a unit test
      proves a frozen group's parameters and Adam moments are bit-identical across a gated phase
      and that surviving moment state is not reset at phase boundaries.
- [ ] Phase-1 isotropy in `progressive_anisotropy` is implemented without reparameterizing the
      field (shared/averaged log-scale gradient), and a test pins `sx == sy` drift to zero during
      that phase.
- [ ] Screen on the maintained benchmark images at a fixed budget and seeds: final PSNR,
      iterations-to-target, wall-clock-to-target, displacement of original means (px), covariance
      change from initialization, fraction of original rows still contributing, densification
      event count, and PSNR at the first densification event, all from logged config + seed.
- [ ] Outcome (positive or negative) recorded as an ARA observation or claim row, and this task's
      Index status updated in the same commit.
- [ ] `./scripts/verify.sh` passes.

## Interfaces touched

`src/structsplat/fit.py` (`FitConfig`, group LR application beside `_lr_factor`),
`src/structsplat/gaussians.py` (`parameter_groups` only if a gating hook is needed),
`benchmarks/ablation.py` or a driver under `scripts/experiments/`, `tests/`,
`tasks/INDEX.md`, `ara/`.

## Depends on

FIT-016, BENCH-002

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff` and `### Review` blocks using the schema in `tasks/README.md`.
Before a formal result-bearing run, append the prospective `### Protocol review` block from that
document and bind the exact frozen protocol digest.

## Notes

Freezing is reversible by construction: multipliers return to the configured group LRs, and Adam
state persists, so a schedule can be aborted mid-run without state loss. Success is claimed only
against the frozen screen protocol (matched budgets, seeds, and images); a win authorizes a
confirmation task, not a default flip. If ABL-005's fair-regime shard has reported by execution
time, prefer its regime settings for the screen; its completion is not a prerequisite. The
realtime-gs repository is running the adjacent 3D-side initialization-preservation question; keep
terminology aligned but evidence separate.
