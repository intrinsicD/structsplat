# DOCS-005 — Agentic workflow maturity

## Context
StructSplat already has strong scientific invariants, an ARA claim ledger, task and docs
checkers, repository-prefixed skills, and one portable verification command. Its coordination
and handoff layer is thinner: task pickup does not persist ownership or a reviewed revision,
ordinary substantial work has no structured verdict, the large task index has no generated
session-start view, current-pipeline report bundles have no standalone integrity gate, and no
checker validates the agent settings/hook/skill mirrors/PR contract/CI spine as one system.

The requested audit compares the current agentic workflows in `agent-kit`, `IntrinsicEngine`,
`realtime-gs`, and `prospect`. The result must adopt only mechanisms that fit StructSplat's
existing authorities, not install a second state tree or replace its research-specific gates.
The source-qualified findings are in
`docs/research/2026-07-29-agentic-workflow-audit.md`.

## Goal
Make the repository's agent workflow reconstructable, reviewable, mechanically checked, and
portable across agent harnesses while keeping `tasks/INDEX.md` and `ara/` as the sole task and
evidence authorities.

## Acceptance criteria
- A dated current-head audit records every material transferable mechanism considered, the
  source repository, the StructSplat gap, and an explicit adopt/adapt/reject decision.
- One ADR defines the authority model: generated views may derive from `tasks/INDEX.md`, but no
  parallel backlog/current-task/evidence ledger is introduced.
- Task pickup records Driver, Reviewer, Turn, and reviewed revision in the task itself; terminal
  workflow-governed tasks require a structured verdict, with self-review visibly downgraded to
  provisional rather than presented as independent approval.
- A deterministic `tasks/SESSION-BRIEF.md` exposes in-progress, actionable, and blocked tasks from
  the task index. CI fails when it is stale.
- Task validation expands abbreviated dependencies such as `FIT-005/006/007`, rejects unresolved
  dependencies and dependency cycles, validates protected-action handoffs, prospective protocol
  reviews, and bounded revision authorization, without rewriting historical task records.
- ARA validation rejects proof traversal/symlink escape, unknown or duplicate fields, empty or
  punctuation-only status, and inconsistent punctuated status handling.
- A repository-level agent-workflow checker validates the canonical guide adapter, Claude
  settings and hook, skill mirrors, verification stage order, CI invocation, generated brief,
  PR template, and the Python 3.10–3.13 structural matrix.
- A standalone current-pipeline report-bundle checker rejects incomplete, inconsistent,
  non-finite, dirty-source, or non-portable results before they can be handed off as evidence.
- Task and PR templates, the session hook, relevant skills, README, and canonical guides describe
  the same lifecycle and commands.
- Focused adversarial tests cover the new task, session-brief, workflow, and report-bundle gates;
  `./scripts/verify.sh` passes.

## Interfaces touched
`AGENTS.md`, `CLAUDE.md`, `README.md`, `.claude/`, `.github/`, `docs/`, `tasks/`,
`scripts/`, `src/structsplat/workflows.py`, `tests/`, and the workflow-facing project skills. No
renderer, fitter, codec, or research result is changed.

## Depends on
DOCS-003.

## Agent workflow
- Driver: codex
- Reviewer: codex
- Turn: none
- Reviewed revision: tree `720a4284aaffdd7000fc7c21551d01ce787ecd1a`

### Handoff log

### Handoff

#### Objective
Integrate the genuinely missing coordination and evidence-handoff mechanisms from the four
audited repositories without creating a second task, experiment, or claim authority.

#### Changes
Added the source-qualified audit and ADR-0031; task ownership/handoff/review validation; compact
dependency expansion and cycle detection; a generated session brief; workflow drift and report
bundle checkers; task and PR templates; session-hook/CI/verification integration; artifact links
in maintained reports; aligned guides and skills; and adversarial tests. Retired stale DOCS-003
state while leaving the open format ratchet with DOCS-004.

#### Evidence
The pre-change repository gate passed with 1,496 tests, 4 skips, and all structural checks.
Post-change focused validation passed 24 tests across the new checkers, docs sync, and pipeline
workflow surface. Ruff, shell syntax, Python compilation, task/ARA/docs/script/workflow checkers,
and diff whitespace checks passed. The full gate remains the post-closure check.

#### Assumptions
`tasks/INDEX.md` and task files remain the only work authority; ARA remains the only claim and
evidence ledger. Historical tasks and old report bundles are not rewritten to the new schemas.

#### Uncertainties
No distinct reviewer was available in this run. Report integrity was exercised with adversarial
portable fixtures and the maintained HTML generator, not a new GPU result bundle.

#### Review focus
Check for duplicate authorities, over-broad ceremony, task state-transition loopholes, report
schema mismatches, misleading capability language, stale generated context, and configuration or
CI drift that can escape the new meta-checker.

#### Protected actions not taken
No sibling repository was modified. No result-bearing experiment, protected dataset, external
write, commit, push, or independent-review claim was made.

#### Recommended next action
Run the full repository gate after mechanical task closure, then obtain independent review of the
recorded tree before calling this verification-policy change accepted.

### Review

#### Verdict
Provisionally accepted (self-reviewed)

#### Self-reviewed
Yes

#### Correctness
The implementation preserves the existing task and ARA authorities and makes each selected gap
executable. Self-review found and fixed a missing report-artifact reachability integration, a
terminal pending-reviewer loophole, clean-status hash inconsistency, malformed-settings handling,
and accidental formatter churn.

#### Evidence quality
Focused adversarial tests cover dependency expansion/cycles, ownership and self-review semantics,
session derivation, duplicate verification stages, report completeness, clean/dirty/error
qualification, cross-format agreement, non-finite values, escaping links, artifact reachability,
and the live repository checkers. This is strong structural evidence, not independent semantic
approval.

#### Simplicity
The design adds derived views and focused checkers around existing authorities. It deliberately
does not add `.agents/state`, a generic experiment registry, a duplicate evidence ledger, or
universal sealed adjudication.

#### Missing cases
The checker cannot prove that two reviewer labels identify independent people or processes, judge
scientific validity, or retroactively qualify historical reports. Detached clean commits are
supported, but no new remote/GPU report was generated for this workflow-only task.

#### Required changes
Independent review is still required before this verification-policy change may be described as
accepted rather than provisionally self-reviewed.

#### Optional improvements
When agent tooling changes materially, repeat the dated source audit and add only failures that
the current authority and drift gates do not already catch.

### Reopened

The final provenance check found that `agent-kit` and `IntrinsicEngine` had advanced after the
initial audit and that `realtime-gs` had gained an uncommitted workflow layer. DOCS-005 was
reopened to revalidate those exact current states and integrate only newly demonstrated gaps.

### Handoff

#### Objective
Revalidate the agent-workflow integration against a bounded snapshot of all four live reference
worktrees, close any newly demonstrated structural loopholes, and bind the exact final
implementation tree for review.

#### Changes
Updated the audit to the exact bounded source snapshot; incorporated hostile ARA field/status/path
cases, the Python 3.10–3.13 structural matrix, protected-action disclosure, prospective protocol
review, and ordered human revision authorization. Self-review then closed retroactive
authorization, task/report symlink escape, and absolute report-artifact serialization loopholes,
and kept ADR-0031 proposed while independent policy review is pending.

#### Evidence
The source audit is pinned to four exact revisions plus explicit dirty-worktree qualifications.
Thirty-one focused workflow, report, and maintained-pipeline tests pass. Ruff, Python and shell
syntax, docs sync, ARA, task, script-layout, workflow, and diff-whitespace checks pass. An earlier
full-tree pass completed with 1,506 tests, 4 skips, and all structural gates; the mechanically
closed tree still requires the final full gate.

#### Assumptions
The bounded source snapshot is the audit boundary; later sibling-repository activity is not
silently attributed to it. Task files and the Index remain the sole work authority, and ARA
remains the sole claim/evidence authority.

#### Uncertainties
No distinct reviewer was available under this run's non-delegated execution constraint. The
report gate is covered by hostile portable fixtures and maintained serializer tests, not a new
GPU result bundle.

#### Review focus
Check authority duplication, role/turn transition loopholes, authorization ordering, source and
artifact containment, report relocation behavior, generated-view freshness, CI/version drift,
and any language that overstates provisional policy review.

#### Protected actions not taken
No sibling repository was modified. No result-bearing experiment, protected dataset, external
write, commit, push, destructive repository action, or independent-acceptance claim was made.

#### Recommended next action
Run the full repository gate on the mechanically closed tree, then obtain a distinct policy
review before promoting ADR-0031 or describing DOCS-005 as independently accepted.

### Review

#### Verdict
Provisionally accepted (self-reviewed)

#### Self-reviewed
Yes

#### Correctness
The final tree preserves one task authority and one claim/evidence authority while making the
selected coordination, review, provenance, and report-integrity gaps executable. The second
self-review caught and fixed live-source drift, retroactive review authorization, task/report
symlink escape, absolute in-bundle artifact serialization, and a misleading accepted ADR status.

#### Evidence quality
The audit distinguishes committed revisions from dirty current context and records explicit
adopt/adapt/reject decisions. Hostile tests exercise dependency cycles and containment,
self-review identity, review ordering, protocol envelopes, session derivation, CI coverage, ARA
metadata/path bypasses, cross-format report consistency, dirty/error qualification, non-finite
data, absolute/escaping paths, and maintained report serialization. This remains structural
evidence rather than independent semantic approval.

#### Simplicity
The implementation extends existing task, ARA, report, and verification authorities with derived
views and focused checkers. It does not import an installer, parallel state tree, generic
experiment registry, duplicate evidence ledger, or generic maturity vocabulary.

#### Missing cases
Structural checks cannot authenticate reviewer identities, judge scientific validity, recompute
task-specific protocol digests, guarantee remote branch policy, or retroactively qualify
historical reports. No Python 3.10–3.13 CI run or new GPU report was produced locally; CI owns the
multi-version reproduction and report fixtures cover the portable contract.

#### Required changes
Independent review remains required before ADR-0031 is accepted or this verification-policy
change is described as independently approved.

#### Optional improvements
Repeat the dated mechanism audit when supported harnesses or formal evidence workflows materially
change; add only failures that remain outside the current authority and drift gates.

## Notes
- `agent-kit`'s parallel `.agents/state/` tree is explicitly out of scope because it would
  duplicate `tasks/INDEX.md`, `tasks/`, and ARA trace state.
- IntrinsicEngine's C++/Vulkan-specific layers, task maturity taxonomy, and large validator suite
  are evidence sources, not transplant targets.
- Prospect's single-use formal adjudication is reserved for protocols that actually need that
  assurance; this task does not impose it on ordinary development.

## Completion
Completed provisionally on 2026-07-29 at reviewed implementation tree
`720a4284aaffdd7000fc7c21551d01ce787ecd1a` after a bounded source revalidation and two explicit
self-review passes. No independent acceptance is claimed; ADR-0031 remains proposed pending a
distinct policy review.
