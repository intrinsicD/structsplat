# INIT-007: Retire the flanking default (measured answer + ADR)

**Status: done.** Completed 2026-07-07. Executes the negative answer to INIT-004/ABL-001's core
question.

## Context
The flanking hypothesis is dead. Across every decision-grade slice committed under
`ara/evidence/`, `aniso_flanking` never leads:

- 8-image exact-CUDA screen (`abl004-stage-screen-8img-cuda-2026-07-04`): `aniso_onedge` wins at
  budget 2000 (+0.24 dB paired, 7/8 wins); `quadtree_wse`/`quadtree_hybrid` win at 5000
  (+0.17/+0.16 dB, 7/8 wins).
- Fair density-control difficult-4 (`fair-density-control-difficult4-2026-07-05`): flanking+tensor
  is the weakest StructSplat row; qt-WSE/on-edge + residual growth lead.
- ABL-006 complete (`abl006-complete-2026-07-07`): flanking eliminated at stage 1. `quadtree_wse`
  is the significant budget-5000 PSNR winner and has the highest budget-10000 mean PSNR;
  `aniso_onedge` stays the low-budget/MS-SSIM alternative.

What survives is the structured-placement claim: StructSplat structured rows beat the
GaussianImage++/Image-GS analogues in all 12 fair PSNR slices. The default (`aniso_flanking`,
ADR-0009) no longer matches the evidence.

## Goal
Flip the shipped default init strategy to the measured winner, record the decision in an ADR, and
retire INIT-004 with its answer written down.

## Acceptance criteria
- [x] ADR-0013 records the default change to `quadtree_wse`, explains why this is not
      budget-conditional yet, and cites the evidence folders above + ABL-006 output.
- [x] `InitConfig.strategy` default updated in `config.py`; CLI help and README examples updated.
- [x] `benchmarks/stage_search.py` `INFLUENCE_DEFAULTS` baseline updated to the new shipped
      default (BENCH-002 rule: influence baseline == shipped default), with a note in the ADR that
      pre/post influence runs are not directly comparable.
- [x] `aniso_flanking` stays available as a strategy and stage-search axis value (it is a control
      arm now, not a thesis).
- [x] INIT-004 retired to `tasks/done/` with the measured answer; `ara/logic/claims.md` C05 kept
      in sync; README hypothesis section states the answer (started in the 2026-07-07 review PR).
- [x] `pytest -q` green; tests asserting the old default updated deliberately, not mechanically.

## 2026-07-07 result

Chose `quadtree_wse` as the single shipped default, not a budget-conditional policy. ABL-006 showed
`quadtree_wse` is significantly ahead at budget 5000 and has the best budget-10000 mean PSNR; the
budget-2000 `aniso_onedge` advantage is not statistically significant, and the MS-SSIM tradeoff is
documented rather than hidden. `InitConfig()`, `structsplat fit`, generation init, feed-forward
fallback, rate-distortion, teacher export, and stage-search baselines now resolve to
`quadtree_wse`. `flank_offset_frac` is strategy-aware when omitted: `aniso_flanking` keeps the
historical 0.5 offset, while `quadtree_wse` gets the measured 0.0 offset.

## Interfaces touched
`src/structsplat/config.py`, `src/structsplat/cli.py`, `benchmarks/stage_search.py`,
`docs/adr/0013-*.md`, `README.md`, `tasks/INIT-004*`, `ara/logic/claims.md`.

## Depends on
INIT-004 (answered), ABL-006 (complete).
