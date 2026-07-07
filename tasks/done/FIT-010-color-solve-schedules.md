# FIT-010: Cheap color-solve schedules (init / final / on-split)

**Status: done.** Completed 2026-07-07.

## Context
FIT-005 showed `color_solve_every=10` could buy about +0.5 dB, but at much higher fit time. This
task tested whether event-triggered solves could capture most of that quality at a smaller cost.

## Outcome
- Added `FitConfig.color_solve_schedule`, a `+`-composable schedule over `none`, `every`, `init`,
  `final`, and `on_split`.
- Preserved `color_solve_every` compatibility: `color_solve_every=N` still means periodic
  `every<N>` behavior.
- Added stage-search parsing for `none`, `every<N>`, `init`, `final`, `on_split`, and compositions
  such as `init+on_split` or `every10+on_split`.
- Color-solve events now record a `trigger` field in `history["color_solve_events"]`.
- `on_split` runs after split/relocate/adaptive-growth optimizer-state carry, then resets color
  optimizer state.
- `final` runs before final metrics and is included in `fit_seconds`.
- The FIT-005 restrictions remain enforced: normalized renderers only and `color_basis=constant`.

## Evidence
- Focused tests:
  `python -m pytest tests/test_fit_dynamics.py::test_fit_color_solve_runs_in_loop_and_fails_closed_for_other_renderers tests/test_fit_dynamics.py::test_color_solve_init_final_and_on_split_schedules tests/test_stage_search.py::test_color_solve_kwargs_parse_stage_modes tests/test_stage_search.py::test_color_solve_is_stage_axis tests/test_stage_search.py::test_new_stage_smoke_matrix_records_expected_events_and_html -q`
  passed 5/5.
- Schedule smoke: `ara/evidence/fit010-color-solve-schedules-2026-07-07/run.md`.

Schedule smoke summary:

| color solve | mean PSNR | delta PSNR vs none | mean AUC | extra fit s | mean events |
|---|---:|---:|---:|---:|---:|
| `none` | 23.3878 | +0.0000 | 21.6988 | +0.0000 | 0.00 |
| `every10` | 23.8762 | +0.4884 | 22.0057 | +0.5872 | 6.00 |
| `init` | 22.8631 | -0.5247 | 21.8042 | +0.1131 | 1.00 |
| `on_split` | 23.5811 | +0.1933 | 21.9235 | +0.1187 | 1.00 |
| `init+on_split` | 23.0872 | -0.3006 | 21.9854 | +0.2723 | 2.00 |

Promotion target: event schedules needed at least 70% of `every10`'s PSNR delta (+0.3419 dB) at
no more than 30% of `every10`'s extra fit time (+0.1762 s). `on_split` met the cost target but not
the PSNR target. No event schedule replaces `every<N>` as the promoted quality arm.

Split-dip interaction was positive for `on_split`: logged post-split delta improved from
-0.8055 dB (`none`) to +0.8418 dB (`on_split`) with zero recovery lag. Keep `on_split` searchable
for split-recovery work, but do not promote it as a default.

## Interfaces touched
`src/structsplat/config.py`, `src/structsplat/fit.py`, `benchmarks/stage_search.py`,
`tests/test_fit_dynamics.py`, `tests/test_stage_search.py`, `.claude/skills/benchmark/`,
`benchmarks/README.md`.

## Depends on
FIT-005. Pairs with FIT-009 and ABL-005.
