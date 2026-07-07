# INIT-004: Flanking vs on-edge placement + threshold study

**Status: done.** Both strategies implemented in `init.py`; the study ran as the 2026-07-04..07
ABL-004/ABL-006 evidence, and the answer went against flanking. Retirement of the flanking default
is recorded in ADR-0013 and implemented by INIT-007.

## Goal
Decide, empirically, whether pushing edge-Gaussian centers into the flanks beats centering on the
edge, and how sensitive it is to the flat/edge/corner thresholds and `flank_offset_frac`.

## Acceptance criteria
- [x] `aniso_onedge` and `aniso_flanking` produce valid fields differing only in edge offset.
- [x] Ablation harness supports `flank_offset_frac ∈ {0, 0.25, 0.5, 0.75, 1.0}` and threshold pairs.
- [x] Report where flanking helps (expected: low budgets / sharp high-contrast edges) and where it
      is neutral. If never better, recommend dropping it. **Measured: never better.**
      `aniso_onedge` wins at budget 2000 (+0.24 dB paired, 7/8 wins) and
      `quadtree_wse`/`quadtree_hybrid` win at ≥5000
      (`ara/evidence/abl004-stage-screen-8img-cuda-2026-07-04/`); flanking is the weakest
      StructSplat row in the fair density-control comparison
      (`ara/evidence/fair-density-control-difficult4-2026-07-05/`). ABL-006 completed the
      decision-grade confirmation (`ara/evidence/abl006-complete-2026-07-07/`): flanking was
      eliminated at stage 1, `quadtree_wse` became the high-budget PSNR default candidate, and
      `aniso_onedge` remains the low-budget/MS-SSIM alternative. Recommendation: drop flanking as
      the default; keep it as a control arm.

## Depends on
INIT-003, BENCH-001. Answer executed by INIT-007 and confirmed via ABL-006.
