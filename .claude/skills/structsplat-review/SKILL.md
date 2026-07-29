---
name: structsplat-review
description: Use when reviewing a StructSplat diff or PR, or self-reviewing before commit. A correctness-first checklist for differentiable graphics code — gradient safety, numerical stability, determinism, and performance regressions. Trigger on "review this", "check my changes", or before opening a PR.
---

# Review checklist

## Correctness (highest priority)
- **Math matches an oracle.** Structure-tensor / conic / render formulas have NumPy checks — if you
  changed one, update or add the mirror. Prefer a closed-form or NumPy reference over "looks right".
- **Gradients flow** where intended: renderer must stay differentiable w.r.t. `means`, `conics`,
  `colors`. No `.item()`/`.detach()`/`.long()` on the loss path. `radii` is a tiling quantity and
  may be detached.
- **Shapes and coords**: `(H, W, 3)`, positions `(x, y)`. Guard image-boundary indexing.
- **Determinism**: same seed -> same init. New randomness must thread `InitConfig.seed`.

## Stability
- `log_scales` clamped so Gaussians can't collapse (`fit.py`) or explode past image size.
- Divisions guarded (`+ eps`). No NaNs from `sqrt`/`log` of non-positive values.

## Performance (reference is allowed to be slow, but watch regressions)
- Renderer stays chunked and sorted-by-radius (bounded memory). Flag O(N*H*W) dense paths.
- Sampling stays grid-accelerated; flag accidental O(M^2) neighbor scans.

## Hygiene
- Diff is scoped to the task. NumPy/torch split intact. Public signatures documented.
- Any new user-facing behavior has a test and, if quality-relevant, a benchmark number.
- No new one-off driver at the top level of `scripts/` — those go in `scripts/experiments/`.
- No committed evidence bundle under `ara/evidence/` overwritten or rewritten.
- New machinery is right-sized for a demonstrated failure. Flag duplicate authorities, generic
  frameworks with no current consumer, and configuration that is more complex than the behavior.
- Trace every new file and option to a caller, test, guide, task acceptance criterion, or explicit
  follow-up. Flag half-integrated agent output: unused helpers, placeholder TODOs without a task,
  copied policy that names the wrong repository, schemas no checker reads, and docs that promise a
  gate the verification command does not run.
- For workflow-governed tasks, confirm the Driver/Reviewer/Turn state, reviewed revision, latest
  Handoff, protected actions not taken, and verdict describe the actual boundary. Treat the
  producer-authored Handoff as orientation, not proof. Self-review must remain provisional.

## Claim hygiene
Prose is a claim surface. If the diff adds a number or a capability statement to `README.md`,
`docs/`, a task status, or an ADR:

- Is it bound to a row in `ara/logic/claims.md` whose `Proof` cites an artifact that exists?
- Is the wording no stronger than the evidence class — proxy vs full-resolution, development vs
  held-out, screened vs confirmed, one seed vs paired seeds?
- Does a changed default cite the assay that justifies it, and is the `Status` qualifier honest
  about scope (`implemented/screened`, not `implemented`)?

`python scripts/check_ara.py` checks the structure; it cannot tell whether a sentence overstates
its artifact. That is this step, and `structsplat-results-audit` for anything promoted.

## Gate
```bash
./scripts/verify.sh
```
Lint, the portable test gate, and five structural checkers (`docs_sync`, `check_ara`,
`check_task_policy`, `check_script_layout`, `check_agent_workflow`). Maintained report bundles
also run `python scripts/check_report_bundle.py RESULTS_DIR`. Results-bearing changes go through
`structsplat-results-audit` before the claim lands.
