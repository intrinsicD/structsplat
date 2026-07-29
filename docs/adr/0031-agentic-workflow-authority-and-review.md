# ADR-0031: Agent workflow authority and review

- Status: proposed — provisionally self-reviewed; independent policy review pending
- Date: 2026-07-29
- Task: DOCS-005
- Related: ADR-0027, ADR-0028

## Context

StructSplat already has two mature repository authorities: `tasks/INDEX.md` plus task files for
work outcomes, and `ara/` for claims, evidence, and research trace. It also has a canonical guide,
repository-prefixed skills, structural checkers, and a portable verification command. What it
lacked was a mechanically checked execution view and a durable way to identify who implemented
and reviewed a task, which revision was reviewed, and what verdict was reached.

The sibling-repository audit found useful coordination mechanisms, but several came packaged as
parallel current-task, backlog, or research-state trees. Importing those stores would make two
places answer the same question and would weaken reconstruction rather than improve it.

## Decision

Keep `tasks/INDEX.md`, task files, and `ara/` as the sole task and evidence authorities; add only
a generated session brief, in-task Driver/Reviewer/Turn/handoff/verdict records, a
repository-workflow drift checker, and a structural report-bundle gate.

Specifically:

1. `tasks/SESSION-BRIEF.md` is deterministically generated from `tasks/INDEX.md`. It is an
   execution view, never an authorization or outcome authority.
2. Newly picked-up tasks opt into a ratcheted `## Agent workflow` schema. Historical task records
   are not rewritten. Self-review is visibly provisional; high-risk changes require a distinct
   reviewer before acceptance.
3. `scripts/check_agent_workflow.py` checks agreement among agent-facing configuration and
   generated context. Existing domain checkers remain responsible for docs, claims, task state,
   and script placement.
4. `scripts/check_report_bundle.py` supplies structural integrity and clean-source checks for
   maintained reports. It does not replace the semantic `structsplat-results-audit` or ARA.
5. New formal result tasks record a distinct prospective protocol review and exact digest in the
   task. Task-specific tooling recomputes that digest; no generic experiment registry is added.
6. ARA proof paths resolve strictly inside the repository, and new workflow checkers are exercised
   across the declared Python 3.10 floor through current 3.13 in the lightweight CI lane.
7. No `.agents/state/`, second backlog, generic experiment registry, or duplicate claim ledger is
   introduced.

## Consequences

- A session can recover current work without scanning the full Index, while every displayed item
  remains traceable to the canonical row.
- A handoff records the exact review boundary and terminal work carries a machine-checkable
  verdict. Independent approval and self-review can no longer be presented as equivalent.
- Agent settings, hooks, skill mirrors, verification stages, CI, and PR guidance fail together
  when they drift instead of decaying independently.
- Report portability and internal consistency become checkable before scientific review; a
  structurally valid report may still be scientifically weak or unauthorized.
- The workflow remains lightweight for historical and trivial records. The cost of the richer
  schema is paid when a task is picked up, where ownership and review ambiguity matter.
