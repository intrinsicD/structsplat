# StructSplat task workflow

`tasks/INDEX.md` is the sole current outcome authority. Task files hold bounded context,
acceptance criteria, protocol text, handoffs, and reviews. `tasks/SESSION-BRIEF.md` is generated
from the Index for session startup; it never overrides the Index.

Use `tasks/TEMPLATE.md` for new work. The authoritative operational procedure is the
`structsplat-task-workflow` skill and the cross-harness summary is
`docs/agent_workflow.md`.

## Lifecycle

1. Add `tasks/AREA-NNN-slug.md` and its Active Index row in the same change.
2. When work starts, set the Index disposition to `in-progress`, assign stable Driver and
   Reviewer labels in `## Agent workflow`, set `Turn: driver`, and work on
   `area/NNN-slug`.
3. The Driver implements the smallest accepted scope, runs focused checks, and appends a
   structured Handoff. Set the disposition to `in-review`, bind `Reviewed revision`, and set
   `Turn: reviewer` before independent review.
4. The Reviewer reproduces the important evidence and appends a structured Review. A revision
   returns `Turn: driver`; after two consecutive unsuccessful rounds, set the task to `blocked`,
   use `Turn: human`, and record any explicitly authorized additional round.
5. A terminal task moves to `tasks/done/` and its row moves to Retired. Set `Turn: none` and keep
   the review record with the task.

The checker allows self-review when no independent reviewer is available, but it must be recorded
as `Provisionally accepted (self-reviewed)`. It is not independent approval. Research claims,
default changes, ADRs that change scientific or architectural behavior, critical numerical code,
and verification-policy changes require a distinct Reviewer before being described as accepted.
`Turn` is a durable marker, not a filesystem lock: agents sharing one worktree must act serially.

## Workflow fields

Every picked-up task carries:

```markdown
## Agent workflow
- Driver: stable-agent-label
- Reviewer: stable-agent-label-or-pending
- Turn: driver | reviewer | human | none
- Reviewed revision: commit/tree identifier or pending
```

Use these nested Handoff Log blocks:

```markdown
### Handoff

#### Objective
#### Changes
#### Evidence
#### Assumptions
#### Uncertainties
#### Review focus
#### Protected actions not taken
#### Recommended next action
```

The terminal Review block is:

```markdown
### Review

#### Verdict
Accepted | Accepted with follow-up | Revision required | Rejected | Inconclusive |
Provisionally accepted (self-reviewed)

#### Self-reviewed
Yes | No

#### Correctness
#### Evidence quality
#### Simplicity
#### Missing cases
#### Required changes
#### Optional improvements
```

After two consecutive `Revision required` verdicts, a maintainer may authorize only a bounded
additional round. Record it before work resumes:

```markdown
### Revision authorization

#### Authorized by
stable-human-or-maintainer-label

#### Additional rounds
1

#### Decision
Exact bounded correction authorized.

#### Date
YYYY-MM-DD
```

`scripts/check_task_policy.py` validates the fields, turn transitions, reviewed revision, review
semantics, bounded revision authorization, optional prospective protocol-review blocks,
abbreviated dependencies, and actionable dependency cycles. Historical records without an Agent
workflow section remain valid; once a task adopts the section, the ratchet is permanent.

## Results-bearing work

Freeze the question, data/splits, controls, budgets, seeds, metrics, killing rule, source identity,
and exact command in the task before formal execution. A distinct prospective reviewer must
approve the exact digest without executing the run or inspecting sealed outcomes:

```markdown
### Protocol review

#### Reviewer
stable-label-distinct-from-driver

#### Verdict
Approved | Rejected

#### Protocol digest
lowercase-sha256

#### Digest scope
Exact files/sections covered by the digest.

#### Outcomes accessed
No

#### Review focus
Controls, leakage, budgets, metrics, and killing rule.
```

The checker validates any such block; task-specific protocol tooling remains responsible for
recomputing the declared digest. Do not repair or overwrite a formal result in place. The
maintained benchmark/ablation/stage-search workflows produce portable reports; gate them with:

```bash
python scripts/check_report_bundle.py RESULTS_DIR
```

A dirty-source or error-cell report is diagnostic rather than claim-ready. Quantitative promotion
still requires `structsplat-results-audit` and the ARA claim/evidence ledger.

## Generated session view

Regenerate after any task/index lifecycle change:

```bash
python scripts/generate_session_brief.py
python scripts/generate_session_brief.py --check
```

The full repository gate runs the freshness check through `check_agent_workflow.py`.
