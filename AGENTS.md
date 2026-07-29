# Agent guide

The canonical agent guide for this repository is [`CLAUDE.md`](CLAUDE.md) — read it fully before
making changes. It covers the skill routing table, the non-negotiable invariants (NumPy/torch
import split, image and position conventions, the normalized renderer, reproducibility), the
`ara/` claim and evidence ledger, the repository layout, and the verification gate.

At session start, read the generated [`tasks/SESSION-BRIEF.md`](tasks/SESSION-BRIEF.md), then
confirm selected work in `tasks/INDEX.md` and its task file. The brief is derived context, not a
second task authority. [`docs/agent_workflow.md`](docs/agent_workflow.md) defines the
cross-harness Driver/Reviewer/Turn, handoff, review-verdict, clean-results, and self-review
policy.

Project skills live in `.claude/skills/` (mirrored for Agent Skills / Codex discovery in
`.agents/skills/`). Every skill name is `structsplat-`prefixed so it cannot collide with a
sibling repository's skill when more than one repo is open in a session. Load them by task per
the routing table in `CLAUDE.md`; the authoritative bodies are the `SKILL.md` files themselves.

If your harness does not auto-discover skills, treat the files under `.claude/skills/` as
required reading for the task at hand.

Before every commit:

```bash
./scripts/verify.sh
```

That runs lint, the portable test gate, and the structural checkers (`docs_sync`, `check_ara`,
`check_task_policy`, `check_script_layout`, `check_agent_workflow`). CI mirrors the same sequence
on CPU. Maintained report bundles are additionally checked on demand with
`python scripts/check_report_bundle.py RESULTS_DIR`.
