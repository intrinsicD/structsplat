# ADR-0025: One maintained entrypoint for the current best pipeline

## Context

The repository records what it has rejected far better than what its accepted results compose
into. A 2026-07-25 review found three separate, partially conflicting definitions of "best", none
of which is the end-to-end pipeline:

- **Shipped library defaults** (`config.py`, ADR-0009 + ADR-0013). Explicit and machine-readable,
  but ADR-0009 chose them to be *conservative*: "every Codex breadth axis defaults to its
  conservative option". They are the safe pipeline, not the measured one.
- **`structsplat_best_default`** (`benchmarks/fair_density_control_compare.py`, claim C12). A
  pinned recipe — `aniso_onedge` + WSE, feature cap `12@160`, tensor-aware residual growth — that
  disagrees with the shipped default on the init strategy (ADR-0013 ships `quadtree_wse`) and on
  scale caps (C02/ADR-0012 reject them as a default). Both positions are correct in their own
  regime; nothing stated the regime boundary.
- **The safe-commit schedule** (`safe_schedule.py`), which describes itself as "the production
  schedule" and carries the best measured quality (C50/C51/C52), but was reachable only through
  `scripts/fit_janelle_*.py`, absent from the `docs/architecture.md` pipeline diagram, and
  governed by no ADR — ADR-0020..0023 cover its *storage*, never its phase order or commit gate.

Meanwhile several confirmed wins ship default-off for good reasons (FIT-015/C18, INIT-009/C25,
PORT-002/C53), so assembling "what should I run today" required cross-referencing the claim
ledger, task status lines, ADRs, and README prose. Dome (alpha-matted) capture work needs a stable
answer to that question, and needs the same answer to keep working when the recipe changes.

## Decision

`src/structsplat/pipeline.py` is the single maintained definition of the current best pipeline, and
`structsplat convert` is its command-line entrypoint.

1. **`run_pipeline(image, mask=None, cfg=PipelineConfig())`** composes the safe-commit schedule
   with the measured-best initialization. `PipelineConfig`'s defaults *are* the recipe; `RECIPE`
   records its name, version, per-choice evidence, and the scope that evidence reaches.
2. **The mask argument selects the arm, and nothing else does.** `mask` given -> the masked arm
   (containment, boundary-tangent initialization, boundary closure). `mask=None` -> the full-frame
   arm.
3. **The full-frame arm degenerates the mask machinery rather than replacing it.** `mask=None`
   builds an all-true frame mask: `mask.signed_distance` already clips an empty complement to the
   image diagonal, so the projection interior is the whole frame, per-row caps are inert, the
   boundary band `(0, boundary_band]` is empty, and `_visible_boundary` is empty. Boundary closure
   is then skipped explicitly rather than spending its budget on unallocatable proposals. Every
   other phase, the Pareto commit gate, and the metric vector are shared verbatim.
4. **The recipe is deliberately not the library default.** `pipeline.py` sets
   `wse_progressive_order=True` (C25), `pareto_safe_checkpoints=True` every 50 steps (C50), and
   `event_color_solve=False` (C50) while `InitConfig`/`FitConfig` keep their conservative values.
   Promoting a knob to a library default still requires its own confirmation and ADR.
5. **Changing the recipe is a single reviewable diff.** Edit `RECIPE` and the `PipelineConfig`
   defaults together, bump `RECIPE["version"]`, and cite the authorizing claim. It is a
   results-bearing change: it needs `structsplat-results-audit` and an `ara/logic/claims.md` row.

`structsplat fit` is unchanged and remains the knob-level research command.

## Consequences

+ "What is the current best pipeline?" has one machine-readable answer, and dome work has a stable
  entrypoint that survives recipe changes.
+ The full-frame arm reuses the masked arm's evidence-bearing mechanism instead of forking a second
  method, so a future schedule improvement lands in both arms at once.
+ Every run is self-describing: the result carries `recipe`, `arm`, `init`, `device`, and the
  resolved `pipeline_config`.
- **The full-frame arm has no benchmark screen.** The schedule's quality evidence (C50/C51/C52) is
  one masked image, one seed, one GPU. The arm is a mechanism extension, and this ADR does not
  claim it beats the ABL-006/C12-evidenced plain-fit path on ordinary images. `BENCH-017` is that
  screen; until it runs, the full-frame arm ships as best-known-by-mechanism, not by measurement.
- A second definition of "best" now exists next to `structsplat_best_default`. That benchmark pin
  stays: it is the comparison baseline for the fair-density harness at its own proxy regime, and
  CORE-012 does not re-pin it.
- `PipelineConfig` must stay torch-free (the CLI parser reads its defaults for `--help`), so
  `pipeline.py` imports `safe_schedule` inside functions.

## Links

Composes ADR-0013 (init default), ADR-0017/0019 (mask containment and anisotropic caps),
ADR-0020..0023 (pooled/transactional storage), ADR-0011 (renderer). Does not supersede ADR-0009:
the library defaults it records stay conservative by design. Realizes CORE-012; BENCH-017 owns the
full-frame screen. Evidence: C25, C50, C51, C52.
