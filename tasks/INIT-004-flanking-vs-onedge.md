# INIT-004: Flanking vs on-edge placement + threshold study

**Status: partial.** Both strategies implemented in `init.py`; the study is ABL-001's core.

## Goal
Decide, empirically, whether pushing edge-Gaussian centers into the flanks beats centering on the
edge, and how sensitive it is to the flat/edge/corner thresholds and `flank_offset_frac`.

## Acceptance criteria
- [x] `aniso_onedge` and `aniso_flanking` produce valid fields differing only in edge offset.
- [ ] Ablation over `flank_offset_frac ∈ {0, 0.25, 0.5, 0.75, 1.0}` and threshold pairs.
- [ ] Report where flanking helps (expected: low budgets / sharp high-contrast edges) and where it
      is neutral. If never better, recommend dropping it.

## Depends on
INIT-003, BENCH-001.
