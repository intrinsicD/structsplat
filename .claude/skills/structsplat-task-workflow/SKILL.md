---
name: structsplat-task-workflow
description: Use when picking up, implementing, or closing a StructSplat task from the tasks/ directory. Covers the task-file lifecycle, branch naming, the definition of done, and how a task links to ADRs, tests, and the benchmark. Trigger on "implement TASK-ID", "work on the next task", or opening anything under tasks/.
---

# Task workflow

Tasks live in `tasks/` as `AREA-NNN-slug.md` and are tracked in `tasks/INDEX.md`. Areas:
CORE, INIT, FIT, HIER (hierarchy), BENCH, ABL (ablation), FF (feed-forward), COMP (compression),
PORT (CUDA/engine), DOCS (repository/documentation workflow). `tasks/SESSION-BRIEF.md` is a
generated startup view; the Index remains the outcome authority. Use `tasks/TEMPLATE.md` for new
work and `tasks/README.md` for the exact handoff/review schemas.

## Lifecycle
1. **Read the task file end to end** plus every ADR/task it references. Load the `structsplat-core` skill first.
2. Set status `in-progress` in `tasks/INDEX.md`. Add `## Agent workflow` with stable Driver and
   Reviewer labels, `Turn: driver`, and `Reviewed revision: pending`. Branch:
   `area/NNN-slug` (e.g. `init/003-wse`).
3. Implement against the acceptance criteria only. Resist scope creep — extra ideas become new
   task files, not silent additions (this repo favors small, reviewable diffs).
4. Add/extend tests under `tests/` for every new behavior (`structsplat-review` skill lists what to check).
5. Run `pytest -q` and, if the change can affect quality, the relevant `structsplat-benchmark` slice.
6. Append a structured Handoff. Set status `in-review`, `Turn: reviewer`, and bind the exact
   commit/tree in `Reviewed revision` before independent review.
7. The Reviewer reproduces important checks and appends a Review verdict. `Revision required`
   returns `Turn: driver`; after two consecutive unsuccessful revision rounds, use `Turn: human`
   and record any maintainer-authorized bounded extra round rather than looping.
   If no distinct reviewer is available, use only
   `Provisionally accepted (self-reviewed)`—never present it as independent approval.
8. Update docs in the **same** commit (`structsplat-docs-sync`): task status, Index, generated
   session brief, and any ADR/ARA consequence.
9. Run `./scripts/verify.sh`. Terminal work sets `Turn: none`, moves to `tasks/done/`, and moves
   its Index row to Retired.
10. Open a PR using the repository template, including task scope, exact verification, claims,
    handoff roles, reviewed revision, verdict, risks, and follow-ups.

Research claims, default changes, scientific/architectural ADRs, critical numerical code, and
verification-policy changes require a distinct Reviewer before the result is called accepted.
Self-review can close the implementation record provisionally, but cannot satisfy that independent
approval requirement.

`Turn` is not a filesystem lock. Agents sharing a worktree act serially. A Driver handoff must
state protected actions not taken; the Reviewer treats it as untrusted orientation and recomputes
important checks. Before a formal result-bearing run, add the distinct prospective
`### Protocol review` block from `tasks/README.md`, bind its exact digest, and keep the reviewer
away from sealed outcomes.

## Definition of done
- Acceptance criteria in the task file are all satisfiable and tested.
- No unrelated churn; NumPy/torch import split preserved (see `structsplat-core` invariant 1).
- If a design decision was made, an ADR exists or was updated.
- Results (if any) are reproducible from a logged config + seed.
- Any claim the task produced or refuted has a row in `ara/logic/claims.md` bound to evidence.
- Handoff, reviewed revision, and verdict identify the exact review boundary.
- `./scripts/verify.sh` is green.

## Writing a new task
Start from `tasks/TEMPLATE.md`. Keep Context / Goal / Non-goals / Acceptance criteria /
Interfaces touched / Depends on / Agent workflow / Notes short and testable.

Name the file `AREA-NNN-slug.md` and add a row to the Active table of `tasks/INDEX.md` in the same
commit — `check_task_policy.py` fails on a task file the index does not list, and on an index row
with no file. Retiring a task means moving it to `tasks/done/` and moving its row to the Retired
table with the new path. `Depends on` entries must name real task IDs or `ADR-NNNN` decisions that
exist; the checker resolves both.

Regenerate `tasks/SESSION-BRIEF.md` after the Index changes. Status vocabulary for the Active
table (a qualifier may follow the first word, as in
`implemented/screened — rejected by the 500-step guard`): `todo`, `partial`, `in-progress`,
`in-review`, `revision-required`,
`implemented`, `implemented/screened`, `implemented/confirmed`, `design-only`, `completed`,
`blocked`, `terminal`, `repaired`, `superseded`, `abandoned`. Adding a new word means editing
`STATUS_WORDS` in `scripts/check_task_policy.py` deliberately.
