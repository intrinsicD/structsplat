# FIT-009: Factor the refine axis into orthogonal sub-axes

**Status: done.** Completed 2026-07-07.

## Context
`refine` mode strings like `residual_tensor_add_nms_residual_color` bundled three orthogonal
choices: **where** capacity is added (residual / residual x tensor / absgrad / ranked /
freq-violation), **how** it is instantiated (duplicate / fp_duplicate / moment_preserving /
sampled add), and **dedup** (NMS on/off). Evidence from FIT-004/FIT-006/FIT-007 suggested the
winning ingredients might live in different bundles, and `residual_tensor x moment_preserving`
could not be written before this task.

## Outcome
- Added `FitConfig.refine_site`, `FitConfig.refine_primitive`, and `FitConfig.refine_nms` while
  keeping `split_mode` as a legacy alias.
- Added alias resolution for every core legacy split mode, including NMS aliases, with a CPU test
  proving legacy split aliases and equivalent factored fields produce bit-identical fitted fields.
- Changed the fit loop to dispatch by factored site/primitive, enabling new combinations such as
  `refine_site=residual_tensor, refine_primitive=moment_preserving`.
- Factored `benchmarks/stage_search.py` into explicit refine axes:
  `refine_site`, `refine_primitive`, `refine_nms`, plus orthogonal `refine_color`,
  `refine_prune`, and `refine_relocate` flags. Legacy `--refine-modes` remains accepted and is
  normalized into the new schema.
- Preserved BENCH-002 equal-final-capacity accounting: adding refine arms still start below the
  cell budget and cap at the requested budget.
- Updated benchmark docs, ADR-0010, architecture notes, and the local benchmark skill.

## Evidence
- Focused tests:
  `python -m pytest tests/test_fit_dynamics.py::test_legacy_split_alias_matches_factored_refine_fields tests/test_fit_dynamics.py::test_residual_tensor_site_can_use_moment_preserving_primitive tests/test_stage_search.py -q`
  passed 32/32.
- Full fair-regime slice:
  `ara/evidence/fit009-refine-axis-factor-2026-07-07/run.md`.

The fair-regime difficult-four slice covered `kodim01`, `kodim07`, `kodim13`, and `kodim19`,
budget 2000, seed 0, 1500 iterations, max-side 768, exact CUDA renderer, and equal final N.

| refine | cells | mean PSNR | mean MS-SSIM | mean AUC |
|---|---:|---:|---:|---:|
| `moment_preserving` | 4 | 24.3194 | 0.88102 | 24.4891 |
| `residual_add` | 4 | 24.2549 | 0.88177 | 24.4723 |
| `residual_tensor_add` | 4 | 24.2213 | 0.87992 | 24.5915 |
| `residual_tensor_moment_preserving` | 4 | 24.0325 | 0.87685 | 24.4541 |

Conclusion: the interface works and the cross-product is now searchable, but this slice does not
support promoting `residual_tensor x moment_preserving`. Keep it as an explicit search candidate;
plain `moment_preserving` had the best mean PSNR here and `residual_tensor_add` had the best mean
AUC.

## Interfaces touched
`src/structsplat/config.py`, `src/structsplat/fit.py`, `benchmarks/stage_search.py`,
`tests/test_fit_dynamics.py`, `tests/test_stage_search.py`, `.claude/skills/benchmark/`,
`benchmarks/README.md`, `docs/adr/0010-stage-influence-protocol.md`, `docs/architecture.md`.

## Depends on
FIT-004, FIT-006, FIT-007.
