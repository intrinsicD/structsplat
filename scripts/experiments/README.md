# scripts/experiments/

One-off, experiment-specific scripts live here. `scripts/` itself is reserved for durable
repository tooling that every agent and CI run may need.

## Which directory does my script go in?

| Script | Goes in |
|---|---|
| Runs on every commit, or is part of `verify.sh` / CI | `scripts/` |
| Reusable across experiments (a validator, an env bootstrap, a figure renderer) | `scripts/` |
| Drives one task, one protocol, one image, or one ablation | `scripts/experiments/` |
| Throwaway, never to be committed | the session scratchpad, not the repo |

`scripts/check_script_layout.py` enforces this: any new top-level file in `scripts/` must be
added to the `DURABLE_SCRIPTS` allowlist in that checker, with a reason. If your script is
experiment-specific, put it here instead and no allowlist entry is needed.

## Grandfathered scripts (do not move)

The task-specific drivers currently at the top level of `scripts/` are pinned there and
allowlisted with that reason:

- `fit_janelle_*.py`, `run_janelle_*.py`, `compare_janelle_*.py` — the FIT-021 through FIT-026
  Janelle lineage
- `run_abl004_full_ablation.sh`, `run_abl005_*.sh`, `run_stage_*.sh`, `prepare_abl004_images.py`,
  `run_storage_budget_168k.sh` — the ABL-004/ABL-005 and stage-search lanes
- `setup_native_*_env.sh` — BENCH-005 native-reference environment bootstraps

They are referenced from `README.md`, from `tests/`, from `ara/evidence/*/run.md` command
records, and — for the Janelle scripts — copied verbatim into
`runs/*/provenance/` as the source binding for committed evidence bundles. Moving them would
make committed evidence commands wrong and break the provenance copies' correspondence to a
real path, which `structsplat-results-audit` forbids.

The lifecycle policy applies to new scripts. Existing evidence provenance is never rewritten to
satisfy a layout rule.

## Conventions for scripts here

- Name the task: `<TASK-ID>_<what>.py` (e.g. `fit027_boundary_sweep.py`).
- Put the exact invocation in the module docstring, and pin `InitConfig.seed`.
- Reference the `ara/evidence/` bundle or `tasks/` entry the script produced, so a reader can
  get from script to evidence and back.
- When the task closes, leave the script here. It is provenance, not clutter.
