# Agentic workflow audit — 2026-07-29

## Question and method

Which mechanisms in the current agent workflows of `agent-kit`, `IntrinsicEngine`,
`realtime-gs`, and `prospect` close a real StructSplat gap without duplicating its task or ARA
authorities?

The audit inspected each repository's guides, agent settings and hooks, task schemas and
validators, verification and CI entrypoints, review/handoff policy, experiment/result contracts,
and research-evidence rules. It compared those mechanisms against StructSplat at base revision
`ebf860bcf29d15e33d1be32315c6856baa30abb5`. A pre-delivery provenance check found that two
repositories had advanced and a third had acquired uncommitted workflow work, so the audit was
reopened and revalidated against the exact states below. The table is a bounded read-only
snapshot taken at `2026-07-29T12:58:52+02:00`; later sibling-worktree activity is outside this
audit.

| Repository | Inspected revision | Worktree qualification |
|---|---|---|
| `agent-kit` | `206ef8dae269b57063070a034975b6f9e4a54cd0` | clean `task/007-installer-v2`; accepted Task 006 and Task 007's independent round-one `Revision required` verdict were distinguished |
| `IntrinsicEngine` | `9954895d0dbd10c1100ab3be5c45c1241efba779` | dirty runtime/task/ARA work was inspected as uncommitted context; the committed and dirty runtime deltas did not change the workflow mechanisms under comparison |
| `realtime-gs` | `548e04a497669a0ce17af80161b90a7a8e1f2425` | uncommitted agent-workflow/task-skill/prospective-protocol-review work and experiment-contract edits were inspected as current context, not attributed to the revision |
| `prospect` | `21b16952b4ebe930bde312c9f0bb0468ed46fc6e` | dirty research/evidence work was preserved and inspected as uncommitted context, not attributed to the revision |

This is a mechanism audit, not a ranking of repositories. Domain-specific machinery was accepted
only when StructSplat had the corresponding problem.

## Baseline already present in StructSplat

StructSplat already had:

- a canonical cross-harness guide (`CLAUDE.md` via `AGENTS.md`) and repository-prefixed mirrored
  skills;
- `tasks/INDEX.md` plus task files as a checked work-outcome ledger;
- ARA staging, trace, claim, and evidence layers with a structural checker;
- scientific review and result-audit skills;
- one pre-commit verification command mirrored by CPU CI;
- deterministic method invariants and source/config/seed expectations;
- portable report-producing maintained workflows.

Those are retained. A mature integration should reinforce these authorities, not layer generic
state files over them.

## Transfer matrix

| Source mechanism | StructSplat finding | Decision and integration |
|---|---|---|
| `agent-kit`: explicit Driver/Reviewer/Turn, persisted handoff, terminal verdict, self-review downgrade, serial worktree use, and bounded revision loop with recorded human override | Genuine gap: task files did not identify ownership, review boundary, protected actions, or verdict; the initial two-round limit had no explicit authorized-override path | **Adapt.** Add an opt-in ratcheted schema to task files, enforce independent-vs-self review semantics, require protected-action disclosure, and allow only recorded bounded Revision authorization after escalation |
| `agent-kit` Task 006: hostile ARA checker fixtures for punctuated/empty status, unknown/duplicate fields, proof traversal/symlink escape, and commented duplicate gate commands | Genuine gap: StructSplat's claim checker trusted any existing joined path, ignored punctuated unknown fields, and normalized status inconsistently; the new verify-stage checker already handled commented duplicates | **Adopt.** Resolve proof and PAPER paths strictly inside the repository, use one status normalizer, reject empty/punctuation-only status and unknown/duplicate fields, and add adversarial fixtures |
| `agent-kit` Task 007: install groups, slug rendering, dry-run, collision rollback, receipts, doctor, and an independent review that found renderer-boundary, impossible-plan, and filesystem-node false negatives | No installable workflow distribution exists here; its review quality reinforces the distinct-review mechanism, but its findings are specific to an installer product StructSplat does not have | **Reject the product surface.** Do not add an installer, receipt authority, or repair surface without a distribution consumer; retain exact skill-mirror checks and the adapted independent-review boundary |
| `agent-kit`: `.agents/state/current-task.md`, backlog, decisions, and research state | Duplicates `tasks/INDEX.md`, task files, ADRs, and ARA trace | **Reject.** No parallel state authority |
| `IntrinsicEngine`: generated session brief from task metadata/dependencies | Genuine gap: the large Index was authoritative but costly to scan; compact dependency forms were not fully interpreted | **Adapt.** Generate `tasks/SESSION-BRIEF.md` from the Index; expand abbreviated dependencies and reject actionable cycles |
| `IntrinsicEngine`: strict validator over guides, hooks, settings, skill mirrors, verify stages, CI, and PR shape | Genuine gap: StructSplat checked each domain but not their orchestration | **Adapt.** Add `check_agent_workflow.py` and a PR template; retain the existing focused checkers |
| `IntrinsicEngine`: task templates, maturity/right-sizing rules, periodic drift/output audits | Partial gap: no task template and review did not explicitly look for oversized or half-integrated agent output | **Adapt narrowly.** Add task/PR templates and review checks; use one workflow drift checker instead of importing the C++-specific task taxonomy |
| `IntrinsicEngine` and `agent-kit`: split lightweight structural lanes and multi-version checker coverage | StructSplat's full numerical gate is intentionally one CPU lane, but its torch-free workflow checkers should cover the declared Python floor | **Adapt narrowly.** Keep one full Python 3.11 gate and run the structural job on Python 3.10–3.13 |
| `IntrinsicEngine`: C++ target matrices, touched-scope machinery, and engine task maturity taxonomy | These solve compiled-engine scheduling and ownership problems rather than a StructSplat workflow gap | **Reject.** Do not import engine-specific lanes or a second maturity vocabulary |
| `realtime-gs`: task-first frozen experiment JSON, dirty-worktree refusal, immutable run roots, result-bundle validation and portable viewer receipt | Genuine report-integrity gap, but experiment questions already belong in StructSplat tasks | **Adapt.** Add a clean-source-by-default structural checker for maintained report bundles; do not add a second experiment registry |
| `realtime-gs`: immutable diagnostic/run-root conventions | Valuable for formal results, but existing evidence bundles and task protocols vary historically | **Adopt as policy, not a historical rewrite.** New formal evidence freezes commands/source in the task and does not overwrite executed artifacts |
| uncommitted `realtime-gs` plus `prospect`: distinct prospective protocol review, exact digest, and sealed-outcome boundary before formal execution | Genuine scientific-workflow gap: StructSplat had frozen tasks and post-outcome audit, but no explicit non-author pre-execution review envelope | **Adapt without a generic registry.** Define and structurally validate an optional task-local Protocol review block; benchmark policy requires it before a new formal run, while task-specific tooling recomputes the declared digest |
| uncommitted `realtime-gs`: generic Scaffolded→Claim-ready maturity ladder | StructSplat already distinguishes implemented/screened/confirmed status, development/held-out evidence, defaults, and ARA claim scope | **Reject.** A second generic maturity vocabulary would blur the repository's more specific evidence language |
| `prospect`: semantic/evidence contract, source/environment binding, non-author adjudication, checkpoint restoration, negative-result preservation | Mostly already covered by ARA and `structsplat-results-audit`; distinct review semantics were missing | **Reuse existing gates and strengthen linkage.** Report validation is structural; semantic promotion still goes through results audit and ARA |
| `prospect`: single-use formal execution/adjudication ceremony | Appropriate for sealed high-stakes protocols, disproportionate for ordinary implementation | **Reject as a universal workflow.** A task may opt into it when its protocol genuinely requires that assurance |

## Gaps closed

The integrated workflow closes eight demonstrated gaps:

1. persistent task ownership, review turn, reviewed revision, protected-action handoff, terminal
   verdict, serial worktree rule, and bounded human-authorized revision escalation;
2. a deterministic small session-start view derived from the existing task authority;
3. correct abbreviated-dependency expansion plus unresolved-reference and cycle checks;
4. a meta-checker for guides, settings, hooks, skill mirrors, verification, CI, PR structure, and
   the Python 3.10–3.13 structural matrix;
5. ARA status/field strictness and repository-contained proof/PAPER paths with hostile fixtures;
6. task and PR templates plus explicit right-sizing and agent-output review criteria;
7. relative serialization of report-owned artifacts plus a standalone clean-source, consistency,
   hash, and portability gate for current-pipeline report bundles;
8. a distinct task-local prospective protocol-review envelope before new formal evidence runs,
   without a second experiment authority.

The bounded executable capability and its falsification boundary are recorded as C61.

It also retires stale DOCS-003 state and leaves the still-open lint/format ratchet correctly owned
by DOCS-004.

## Deliberate non-integrations

No sibling repository is treated as a wholesale template. In particular, this change does not
introduce a parallel `.agents/state` tree, generic experiment JSON authority, duplicate evidence
ledger, workflow installer/receipt, automatic multi-agent requirement, C++/Vulkan task layers,
generic maturity ladder, or universal sealed adjudication. Each would either conflict with the
existing authority map or impose ceremony without a demonstrated StructSplat failure.

## Validation and limitations

The workflow and bundle contracts are covered by adversarial unit tests and the full repository
gate. Structural validation can establish consistency, freshness, and provenance shape; it
cannot establish that a review is insightful, that a claimed independent identity belongs to a
different human or process, recompute a task-specific protocol digest without its declared tool,
or establish that an experiment is scientifically persuasive. High-risk work therefore remains
subject to a distinct reviewer, task-specific digest verification, `structsplat-results-audit`,
and ARA evidence review.

This task itself was implemented without delegated sub-agents, so its terminal verdict is
recorded as provisional self-review rather than independent acceptance.
