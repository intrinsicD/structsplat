---
name: task-workflow
description: Use when picking up, implementing, or closing a StructSplat task from the tasks/ directory. Covers the task-file lifecycle, branch naming, the definition of done, and how a task links to ADRs, tests, and the benchmark. Trigger on "implement TASK-ID", "work on the next task", or opening anything under tasks/.
---

# Task workflow

Tasks live in `tasks/` as `AREA-NNN-slug.md` and are tracked in `tasks/INDEX.md`. Areas:
CORE, INIT, FIT, HIER (hierarchy), BENCH, ABL (ablation), FF (feed-forward), COMP (compression),
PORT (CUDA/engine).

## Lifecycle
1. **Read the task file end to end** plus every ADR/task it references. Load the `core` skill first.
2. Set status `in-progress` in `tasks/INDEX.md`. Branch: `area/NNN-slug` (e.g. `init/003-wse`).
3. Implement against the acceptance criteria only. Resist scope creep — extra ideas become new
   task files, not silent additions (this repo favors small, reviewable diffs).
4. Add/extend tests under `tests/` for every new behavior (`review` skill lists what to check).
5. Run `pytest -q` and, if the change can affect quality, the relevant `benchmark` slice.
6. Update docs in the **same** commit (`docs-sync`): task status, INDEX, any ADR consequence.
7. Open a PR summarizing: what changed, which acceptance criteria are met, benchmark delta.

## Definition of done
- Acceptance criteria in the task file are all satisfiable and tested.
- No unrelated churn; NumPy/torch import split preserved (see `core` invariant 1).
- If a design decision was made, an ADR exists or was updated.
- Results (if any) are reproducible from a logged config + seed.

## Writing a new task
Copy an existing file's shape: Context / Goal / Acceptance criteria / Interfaces touched /
Depends on / Notes. Keep it short and testable.
