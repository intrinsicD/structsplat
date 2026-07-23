# DOCS-003 — Verification spine (docs_sync + verify.sh + pytest markers/seeding + CI)

## Context
StructSplat ships a `docs-sync` skill and a definition-of-done that says "docs updated in the same
commit", but had no executable checker, no `verify.sh`, no CI, no registered pytest markers, and no
central RNG-seeding fixture (despite 100+ test files). rtgs ships exactly this engineering spine;
this task ports it, adapted to StructSplat's flat module layout and the NumPy/torch import split.

## Goal
One CPU-only verification entry point that lint-checks, runs the deterministic test suite, and
structurally verifies docs<->code — mirrored in CI — without breaking core invariant 1 (init math
importable without torch).

## Acceptance criteria
- `scripts/docs_sync.py` (torch-free, AST-based) checks: required docs exist; every skill is listed
  in CLAUDE.md; CLI subcommands + aliases are documented in README; every `init.STRATEGIES` entry is
  documented; CLAUDE.md path tokens exist; every `src/structsplat` module has a docstring. Exit 0
  when clean, 1 on drift.
- `scripts/verify.sh` runs `ruff check` (pinned `select = ["E9", "F"]` correctness baseline) +
  `pytest -m "not slow and not integration"` + `docs_sync`.
- `pyproject.toml` registers `cuda`/`slow`/`integration` markers and enables `--strict-markers`.
- `tests/conftest.py` seeds NumPy every test and torch only when importable (invariant 1 preserved),
  and auto-marks the environment/evidence-coupled modules `integration`.
- `tests/test_docs_sync.py` asserts the checker is clean.
- `.github/workflows/ci.yml` runs the spine on CPU.

## Interfaces touched
`scripts/docs_sync.py`, `scripts/verify.sh`, `pyproject.toml` (`[tool.pytest.ini_options]`),
`tests/conftest.py`, `tests/test_docs_sync.py`, `.github/workflows/ci.yml`, `README.md` (documented
the `feedforward` strategy), `CLAUDE.md` (verification note).

## Depends on
DOCS-001 (docs-sync backfill).

## Notes
- Lint is pinned to a correctness baseline `select = ["E9", "F"]` (syntax + pyflakes) that the whole
  tree already passes clean. The config previously pinned no `select`, so it silently inherited
  ruff's evolving *default* rule set — which is how the spine's first CI lint surfaced ~1147
  pre-existing style findings (I/B/UP/RUF/SIM/…). Those broader style/import families and
  `ruff format --check` (97/136 tracked files predate ruff-format) are intentionally NOT enforced
  yet: adopting them is a separate, repo-wide lint/format ratchet, not this spine.
  **Follow-up task:** stage the style ratchet — `ruff check --fix` the ~774 auto-fixable, triage the
  rest (B023 loop-var capture, BLE001 blind-except may be real), then widen the pinned `select`.
- The full suite is not fresh-checkout portable: ~416 tests (`ssp2e`/`ssp2v`/`ssp2f` actual-coder,
  `local_linear_reproducing_full_assay`, `native_sad_frontier`, `gauge_free_covariance_assay`) need
  committed `results/` evidence bundles (gitignored, absent from a fresh clone or CI runner), a
  Landlock-capable kernel, or a thread-pinned worker env. `conftest.py` auto-marks those modules
  `integration`; the default gate runs the remaining ~1474 tests (verified green — every failure in
  a full run was inside an `integration` module). Run `-m integration` in a provisioned environment.
- Pre-existing drift observed (left untouched — it lives in the frozen evidence machinery):
  `test_gauge_free_covariance_assay` expects the source manifest to cover the whole tree, but the
  frozen manifest predates `triage.py`/`pool.py`/`mask.py`/`batch.py`/`safe_schedule.py`/`viewer.py`.
- Developed on `claude/repos-preallocation-pool-strategy-yhdq9j` (a cross-repo infrastructure
  sweep), not the per-area `docs/003-*` branch convention.
- StructSplat half of a bidirectional infra transfer with rtgs; the rtgs half wires StructSplat
  mechanisms (structure-tensor/WSE init, pooled row lifecycle, checkpoint policy) as opt-in tested
  modules before any default-changing experiment.
