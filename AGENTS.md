# Agent guide

The canonical agent guide for this repository is [`CLAUDE.md`](CLAUDE.md) — read it fully before
making changes. It covers the skill routing table, the non-negotiable invariants (NumPy/torch
import split, image and position conventions, the normalized renderer, reproducibility), the
`ara/` claim and evidence ledger, the repository layout, and the verification gate.

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
`check_task_policy`, `check_script_layout`). CI mirrors the same sequence on CPU.
