# Agent workflow

StructSplat uses one repository-native workflow across Claude Code, Codex, and other agent
harnesses. The workflow is intentionally small: `tasks/INDEX.md` is the outcome authority,
individual task files carry execution context and review history, and `ara/` is the claim and
evidence authority. Generated views and harness configuration may expose those records, but may
not become competing state stores.

ADR-0031 records this authority decision. The dated source audit is
`docs/research/2026-07-29-agentic-workflow-audit.md`.

## Start a session

1. Read `AGENTS.md`, then the canonical `CLAUDE.md`.
2. Read the generated `tasks/SESSION-BRIEF.md`; use it to find current work, then confirm the
   selected row in `tasks/INDEX.md` and read the task itself.
3. Load `structsplat-core` and the task-specific skills from the routing table.
4. Inspect `git status` before editing. Preserve unrelated user changes.
5. For a new substantial unit of work, create or pick up a task using `tasks/TEMPLATE.md`.

The Claude `SessionStart` hook prints the brief and, in a remote environment, prepares the
project virtual environment. Other harnesses get the same repository context by following
`AGENTS.md`; no harness-specific current-task file is authoritative.

## Execute and hand off a task

A picked-up task records stable `Driver`, `Reviewer`, `Turn`, and `Reviewed revision` fields in
its `## Agent workflow` section. The Driver owns implementation and focused checks. Before review,
the Driver appends a structured Handoff, changes the Index disposition to `in-review`, sets
`Turn: reviewer`, and binds the exact commit or tree being reviewed.

The Reviewer appends a structured verdict. `Revision required` returns the turn to the Driver;
after two consecutive unsuccessful revision rounds, use `Turn: human` and stop. A maintainer can
record a bounded Revision authorization before another round; cycling without that record fails
task validation. Terminal work moves to `tasks/done/`, moves from the Active to Retired Index
table, and records `Turn: none`. `Turn` is a protocol marker rather than a filesystem lock, so
agents sharing a worktree act serially.

Self-review is allowed only as an explicit fallback and must use
`Provisionally accepted (self-reviewed)`. It is not independent approval. Research claims,
default changes, scientific or architectural ADRs, critical numerical code, and changes to the
verification policy require a distinct Reviewer before they are described as accepted.
`tasks/README.md` contains the exact Handoff and Review schemas.

## Bind results and claims

Freeze a results-bearing protocol in its task before formal execution: question, data and split
roles, controls, budgets, seeds, metrics, stop or killing rule, exact command, and source
identity. Before execution, a distinct prospective reviewer records an Approved protocol digest
and confirms that sealed outcomes were not accessed. The task-specific runner remains responsible
for recomputing that digest; the generic task checker validates the review envelope. Run formal
evidence from a clean commit. A dirty-source run may be retained only as a diagnostic with its
exact source preserved; it is not claim-ready by default.

Maintained current-pipeline reports are checked with:

```bash
python scripts/check_report_bundle.py RESULTS_DIR
```

That structural gate checks the report manifest, raw-table agreement, finite metrics, stable
cell identity, contained artifacts and links, hashes, and clean repository binding. It does not
judge scientific validity. Quantitative or promoted results still require
`structsplat-results-audit`, an ARA observation or claim, and the relevant evidence bundle.

## Verify and review

Run focused tests while iterating. Before a commit or handoff, run:

```bash
./scripts/verify.sh
```

The command selects the project `.venv` Python when available (or
`$STRUCTSPLAT_PYTHON` when explicitly set), runs the pinned Ruff correctness rules, the portable
pytest gate, then the docs, ARA, task, script-layout, and agent-workflow structural checkers. CI
runs the same command and also exposes the torch-free structural checks as a separate readable
Python 3.10–3.13 matrix, covering the package's declared interpreter floor without multiplying
the full numerical suite.

Use `structsplat-review` for the semantic pass: confirm task scope, invariants, tests, docs,
failure paths, and claim boundaries. The workflow checker proves that the coordination surfaces
agree; it cannot prove that a design is correct or an experiment is persuasive.

## Keep derived workflow state fresh

After changing a task lifecycle row, regenerate the execution view:

```bash
python scripts/generate_session_brief.py
python scripts/generate_session_brief.py --check
```

`scripts/check_agent_workflow.py` validates the brief plus the guide adapter, Claude settings and
hook, skill mirrors, verification stage order, CI contract, and pull-request template. This is a
drift detector, not a second task checker.

Periodically rerun the source-audit method from the dated research note when agent tooling,
supported harnesses, or the formal evidence workflow changes. New machinery is adopted only when
it closes a demonstrated StructSplat gap without duplicating `tasks/INDEX.md`, task files, or
ARA state.
