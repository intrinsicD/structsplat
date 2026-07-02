---
name: docs-sync
description: Use whenever a StructSplat code change affects documented behavior, decisions, or task status, to keep README, docs/adr, docs/architecture.md, and tasks/INDEX.md consistent in the same commit. Trigger on closing a task, making a design decision, renaming, or changing a public interface.
---

# Docs sync

Docs drift is a bug. When code changes, update docs in the **same commit**.

## Triggers -> what to update
- **Closed/advanced a task** -> `tasks/INDEX.md` status + the task file's checkboxes.
- **Made a design decision / reversed one** -> add or amend an ADR in `docs/adr/`
  (`NNNN-title.md`: Context / Decision / Consequences). Never edit a superseded ADR's decision;
  add a new one that supersedes it and link both.
- **Changed the layer map or a public interface** -> `docs/architecture.md` and the `core` skill.
- **Changed how to run something** -> `README.md` usage + the relevant skill.
- **Renamed the project/package** -> `pyproject.toml`, imports, `README`, `core` skill, this repo's
  headers — one atomic commit.

## ADR discipline
One decision per ADR. State the decision in one sentence, then the trade-off and what it rules out.
The code references ADRs by number in module docstrings — keep those numbers valid.

## Quick audit before PR
Grep for the changed symbol/behavior across `docs/`, `README.md`, `.claude/skills/`, `tasks/`.
If a doc names old behavior, fix it now.
