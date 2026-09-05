# DOCS-004 — Staged lint/format ratchet (widen `select`, adopt `ruff format`)

## Context
`DOCS-003` built the verification spine and deliberately pinned `ruff` lint to a correctness
baseline (`select = ["E9", "F"]`) that the whole tree already passes. It left the broader style /
import families and `ruff format --check` out of the gate, with a "Follow-up task" note but no
tracked task ID and no expiry. That note has now been open long enough to become permanent by
default, which is exactly how a deferred ratchet turns into an un-owned exception.

This task is that follow-up, with the work split into stages so no single commit mixes a
repo-wide mechanical reformat with a semantic change.

Measured on the tree at the time this task was opened (`ruff` over
`src tests scripts benchmarks`):

| Signal | Count |
|---|---|
| Findings under `select = E,W,F,I,B,UP,RUF,SIM,C4` | 1472 |
| Auto-fixable (`--fix`) | 751 (+229 behind `--unsafe-fixes`) |
| `RUF100` unused-noqa | 437 |
| `E501` line-too-long | 378 |
| `I001` unsorted-imports | 181 |
| `B023` function-uses-loop-variable | 64 |
| `B905` zip-without-explicit-strict | 50 |
| Files `ruff format` would rewrite | 176 of 248 |

`B023` (loop-variable capture in a closure) and `B904`/`BLE001` are the families that may be
**real defects** rather than style, so they get triage rather than a bulk fix.

## Goal
Widen the pinned `select` and add `ruff format --check` to `scripts/verify.sh`, in stages, without
ever mixing a mechanical reformat with a behavior change, and without weakening the existing
correctness baseline.

## Acceptance criteria
Each stage is its own commit and leaves `./scripts/verify.sh` green.

- **Stage 1 — `RUF100` + `I001` (auto-fix only).** `ruff check --select RUF100,I001 --fix`; add
  `RUF100` and `I001` to the pinned `select`. Purely mechanical; no line rewrapping.
- **Stage 2 — `ruff format`.** Run `ruff format` over the tree in a single commit that contains
  **nothing else**, then add `ruff format --check` to `verify.sh` and CI. Confirm no test that
  asserts a frozen source hash changes verdict — `test_gauge_free_covariance_assay` and the
  `ssp2*` replay assays read source manifests, so re-check them explicitly and record the outcome
  in this task before promoting the gate.
- **Stage 3 — `E501`.** `line-length = 100` is already configured; resolve the 378 long lines
  (Stage 2 removes many) and add `E501`.
- **Stage 4 — safe auto-fixable families.** `UP035`, `RUF022`, `C420`, `RUF036`, `UP018`, `UP034`,
  `SIM114`, `SIM300`, `B009`: fix and add to `select`.
- **Stage 5 — triage the correctness-relevant families.** For each `B023` (64), `B905` (50),
  `B904` (2), and `RUF043` (20) site, decide fix-or-`noqa`-with-reason. A `noqa` needs an inline
  reason. Add these codes to `select` only once the tree is clean. Any site that turns out to be a
  live defect gets its own `FIT-`/`CORE-` bug task rather than being fixed inline here.
- **Stage 6 — close.** `select` reflects the adopted families, `verify.sh` and CI run
  `ruff check` + `ruff format --check`, and `DOCS-003`'s deferral note points here as resolved.

## Interfaces touched
`pyproject.toml` (`[tool.ruff.lint] select`), `scripts/verify.sh`,
`.github/workflows/ci.yml`, and — in Stage 2 only — most files under `src/`, `tests/`,
`scripts/`, `benchmarks/`.

## Depends on
DOCS-003 (verification spine).

## Expiry

### 2026-09-05 disposition (AI-authored workflow decision)

The old requirement that Stages1/2 precede the next results-bearing task closure was missed;
multiple historical tasks closed while these stages remained undone. This is not retroactive
compliance or a user-requested waiver. Retire that expired cross-task closure dependency now:
a repository-wide mechanical rewrite is separate from completing source-bound research and
would enlarge the current change without validating a method. Current verification/CI rules
remain unchanged and mandatory. No stage is claimed complete and no replacement deadline is
invented. Keep the staged adoption plan as explicit backlog, including correctness-relevant
B023/B904 and related triage, not merely formatting. Any live defect found still gets its own
scoped task. The distinct code-research reviewer accepted this explicit disposition.

## Notes
- Stage 2 is the risky one, and only because of frozen-source-hash assays — not because
  formatting changes behavior. Verify those assays before, not after, promoting the gate.
- Do not lower `line-length`, and do not adopt `D` (pydocstyle) as part of this task;
  `docs_sync.py` already enforces module docstrings where the repo wants them.
- `--unsafe-fixes` is out of scope. If a family only clears with unsafe fixes, leave it out of
  `select` and note it here.
