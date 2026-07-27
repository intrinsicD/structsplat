# CORE-012 Janelle mask-margin provenance audit — 2026-07-27

## Question

The maintained pipeline had two defaults for the same containment parameter:
`PipelineConfig.mask_margin=1.5` in `src/structsplat/pipeline.py`, while the report-producing
workflows and the historical Janelle safe-schedule runners used `0.75`. This audit asks which value
reproduces the evidence-bearing Janelle recipe. It does not estimate which margin has better
general image quality.

## Audit

The resolved `run_config.json` files in the local FIT-023, FIT-024, and FIT-025 result bundles were
searched for `mask_margin`. All ten evidence-bearing arms record `0.75`:

| evidence family | arms audited | resolved margin |
|---|---:|---:|
| FIT-023 / C50 checkpoint factorial | 4 / 4 | 0.75 |
| FIT-024 / C51 storage A/B | 3 / 3 | 0.75 |
| FIT-025 / C52 detail-tail comparison | 3 / 3 | 0.75 |

`audit.json` records each source path, SHA-256, extracted value, and disposition.

The tracked source agrees with those resolved records:

- `deprecated_scripts/fit_janelle_safe_commit_schedule.py` defaults `--mask-margin` to `0.75`;
- `deprecated_scripts/fit_janelle_complete_refinement.py` defaults it to `0.75`;
- the exact FIT-024 command in
  `ara/evidence/fit024-transactional-fixed-capacity-janelle-2026-07-24/run.md` does not override
  the runner default.

The tracked `1.5` result is
`ara/evidence/core010-c0001-densified-fit-2026-07-22/run.md`. That run used a different 20,000-row,
L1 + 0.3 SSIM densification procedure on another GPU. It is neither the safe-commit schedule nor a
matched margin arm.

## Disposition

`0.75` is the only value bound to the executed Janelle safe-schedule evidence behind C50, C51, and
C52, so it is the reproducible default for that named recipe. The `1.5` value in `PipelineConfig`
was configuration drift when the evidence-bearing procedure was composed into a maintained API.

There is no controlled `0.75` versus `1.5` comparison. This audit therefore does **not** establish
that `0.75` is universally better, nor does it authorize a quality delta, general default, or
margin sensitivity claim. A future quality ranking requires a frozen paired margin experiment;
until then, `1.5` remains interpretable as a more conservative containment setting available by
explicit override.

## Implementation binding

ADR-0028 aligns `PipelineConfig.mask_margin` and all current-profile parser defaults at `0.75`,
bumps the recipe version, and keeps the `>=0.72` containment floor explicit. Tests assert the
recipe default, parser derivation, manifest provenance, and validation floor.
