# DOCS-006 — Repository-native experiment workflow skill

## Context
StructSplat already has task-first protocol review, maintained benchmark workflows, portable
report bundles, a standalone bundle checker, semantic result audit, and the ARA claim ledger.
Those pieces are distributed across the canonical guide and several focused skills. The
`realtime-gs` repository has an experiment-routing skill, but copying it verbatim would introduce
commands and an experiment registry that StructSplat deliberately does not have.

## Goal
Add one repository-native `structsplat-experiment` skill that routes comparative research work
through StructSplat's existing task, report, audit, and evidence authorities.

## Non-goals
- Do not add a second experiment registry, task state tree, or evidence ledger.
- Do not change fitter, renderer, benchmark, report-generator, result, claim, or default behavior.
- Do not run a result-bearing experiment as part of this workflow-only task.

## Acceptance criteria
- [x] `.claude/skills/structsplat-experiment/SKILL.md` defines the scratch/formal boundary and the
      full task → protocol review → run → report-bundle gate → semantic audit → ARA lifecycle.
- [x] The skill describes the current portable `index.html` contents without claiming unavailable
      test curves, topology telemetry, or an interactive 3D viewer.
- [x] The canonical skill has valid OpenAI interface metadata and an exact `.agents/skills/`
      symlink mirror.
- [x] `CLAUDE.md` routes experiment questions to the new skill while preserving the existing
      benchmark and results-audit responsibilities.
- [x] Focused skill/workflow/docs checks pass.
- [ ] `./scripts/verify.sh` passes in a complete StructSplat development environment.
- [x] Documentation and task state are synchronized.

## Interfaces touched
`.claude/skills/structsplat-experiment/`, `.agents/skills/structsplat-experiment`, `CLAUDE.md`,
`tasks/INDEX.md`, `tasks/SESSION-BRIEF.md`, and this task record.

## Depends on
DOCS-005, BENCH-002/003.

## Agent workflow
- Driver: codex
- Reviewer: codex
- Turn: driver
- Reviewed revision: pending

### Handoff log

### Handoff

#### Objective
Add a repository-native experiment-routing skill without importing realtime-gs-only commands or
creating a second task, protocol, or evidence authority.

#### Changes
Added the canonical `structsplat-experiment` skill and OpenAI interface metadata, its exact
`.agents/skills/` symlink mirror, the ninth-skill routing entry and results-bearing flow in
`CLAUDE.md`, and this durable task record. The skill defines diagnostic versus formal runs,
prospective protocol review, maintained and one-off execution surfaces, comparison validity,
portable report bundles, the exact current 2D `index.html` surface, structural and semantic
gates, and ARA disposition.

#### Evidence
The skill-creator validator, `check_agent_workflow.py`, `docs_sync.py`, `check_task_policy.py`,
`check_ara.py`, `check_script_layout.py`, and `git diff --check` pass. Ruff also passes. The
workspace's base StructSplat runtime completes the portable gate with 1,484 passes and 32 skips;
two unchanged tests fail reproducibly both inside and outside the filesystem sandbox:
`test_rank_deficient_reproduction_design_is_diagnostic_total_and_rejected` reports an infinite
condition number under Torch 2.7, and
`test_opened_descriptor_rejects_path_swap_and_in_place_mutation` does not observe the expected
metadata change on the local filesystem. Neither owning implementation is touched by this diff,
so no full green `verify.sh` result is claimed.

#### Assumptions
The current maintained workflow and `check_report_bundle.py` define the report contract. The new
skill documents and routes that behavior; it does not change the report generator itself.

#### Uncertainties
No independent reviewer was used. The filesystem-sensitive SSP2V descriptor-mutation test blocks
a fully green local gate; the Torch-version-sensitive affine-carrier diagnostic also fails. Neither
failure is changed here, and self-review remains provisional.

#### Review focus
Check that the skill neither duplicates authority nor overstates the static 2D report, especially
the distinction between fitted-target curves, held-out test metrics, native-baseline reports, and
interactive 3D viewers.

#### Protected actions not taken
DOCS-006 performed no result-bearing experiment, sealed-data access, source/default change, or
dependency installation. A separate ABL-002 four-image development diagnostic was executed under
the owning task and remains explicitly non-claim-ready. Before the user-authorized publishing
turn, no commit, push, external write, or modification of the pre-existing untracked `.codex/`
directory was performed.

#### Recommended next action
Disposition the unrelated affine-carrier and SSP2V failures in their owning tasks, rerun
`./scripts/verify.sh`, then bind the reviewed tree and close DOCS-006 after an independent or
visibly provisional review.

## Notes
- `tasks/INDEX.md` and task files remain the sole work/protocol authority.
- `ara/` remains the sole claim and evidence authority.
- The realtime-gs experiment skill is a workflow reference, not an authority copied into this
  repository.
