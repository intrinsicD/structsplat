# INIT-004: Flanking vs on-edge placement + threshold study

**Status: answered (negative).** Both strategies implemented in `init.py`; the study ran as the
2026-07-04..06 ABL-004 slices, and the answer went against flanking. Retirement of the flanking
default (ADR-0013 + config/docs flip) is INIT-007; this file retires to `done/` with it.

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
      (`ara/evidence/fair-density-control-difficult4-2026-07-05/`). Flanking's only remaining
      niche is tiny-budget/short-fit cells, and even there only bundled with other knobs
      (`ara/evidence/merge001-coco-cuda-confirmation-2026-07-06/`, codex top1). Recommendation:
      drop flanking as the default; keep it as a control arm.

## Depends on
INIT-003, BENCH-001. Answer executed by INIT-007; confirmation completes via ABL-006.
