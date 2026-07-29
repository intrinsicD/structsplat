---
name: structsplat-docs-sync
description: Use whenever a StructSplat code change affects documented behavior, decisions, or task status, to keep README, docs/adr, docs/architecture.md, and tasks/INDEX.md consistent in the same commit. Trigger on closing a task, making a design decision, renaming, or changing a public interface.
---

# Docs sync

Docs drift is a bug. When code changes, update docs in the **same commit**.

## Triggers -> what to update
- **Closed/advanced a task** -> `tasks/INDEX.md` status + task workflow/review state, then
  regenerate `tasks/SESSION-BRIEF.md`.
- **Made a design decision / reversed one** -> add or amend an ADR in `docs/adr/`
  (`NNNN-title.md`: Context / Decision / Consequences). Never edit a superseded ADR's decision;
  add a new one that supersedes it and link both.
- **Changed the layer map or a public interface** -> `docs/architecture.md` and the `structsplat-core` skill.
- **Changed how to run something** -> `README.md` usage + the relevant skill.
- **Renamed the project/package** -> `pyproject.toml`, imports, `README`, `structsplat-core` skill, this repo's
  headers — one atomic commit.

## ADR discipline
One decision per ADR. State the decision in one sentence, then the trade-off and what it rules out.
The code references ADRs by number in module docstrings — keep those numbers valid.

- **Produced, changed, or refuted a claim** -> add or update its row in `ara/logic/claims.md`
  with a `Proof` binding to a tracked artifact under `ara/evidence/`. Stage a not-yet-promoted
  finding as an `O<NN>` entry in `ara/staging/observations.yaml` instead. See the "Evidence and
  claims" section of `CLAUDE.md`.
- **Added a doc under `docs/`** -> make it discoverable. Top-level `docs/*.md` must be reachable
  from `CLAUDE.md`/`README.md`/`AGENTS.md`; an ADR must be cited somewhere as `ADR-NNNN`; a
  `docs/research/` note must be referenced from its task or `benchmarks/README.md`.
- **Added a script under `scripts/`** -> a one-off task driver goes in `scripts/experiments/`;
  durable tooling goes at the top level and needs a `DURABLE_SCRIPTS` entry with a reason.

## Quick audit before PR
Grep for the changed symbol/behavior across `docs/`, `README.md`, `.claude/skills/`, `tasks/`.
If a doc names old behavior, fix it now.

Then run the structural gates — they catch the drift grep misses:

```bash
python scripts/docs_sync.py           # docs<->code, doc reachability, skills listed
python scripts/check_ara.py           # claim ledger structure and proof paths
python scripts/check_task_policy.py   # tasks/ tree vs INDEX.md
python scripts/check_script_layout.py # scripts/ vs scripts/experiments/
python scripts/check_agent_workflow.py # guides/hooks/skills/verify/CI/generated brief
```

`./scripts/verify.sh` runs all five plus lint and the portable test gate. Maintained portable
reports use the separate, parameterized `python scripts/check_report_bundle.py RESULTS_DIR` gate.
