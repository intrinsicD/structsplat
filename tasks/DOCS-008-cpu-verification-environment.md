# DOCS-008 — Reproducible CPU verification environment

## Context
The user requested both repositories committed and pushed for use at work. GitHub run
34067002355 installed a newer PyTorch release and failed four exact shared-tail parity checks;
the local PyTorch 2.9.0+cu128 CPU gate passed all 2,447 portable tests. The failure log is retained
in the private local handoff bundle. Preserve the exact assertions and the source-bound research.

## Goal
Use the exercised PyTorch 2.9.0 release for CI and the documented CPU setup.

## Non-goals
No renderer, quantile implementation, tolerance, default, frozen experiment or result change.
Compatibility with newer PyTorch releases remains outside this bounded environment repair.

## Acceptance criteria
- [x] CI and the documented CPU setup select torch 2.9.0.
- [x] Numerical assertions and experimental source remain unchanged.
- [x] Task and generated session brief are synchronized.
- [ ] The post-pin hosted gate passes; local full verification is a pre-commit requirement.

## Interfaces touched
`.github/workflows/ci.yml`, `README.md`, and this workflow record.

## Depends on
DOCS-003

## Agent workflow
- Driver: Codex-delivery
- Reviewer: Codex-delivery
- Turn: reviewer
- Reviewed revision: 8e201fe plus CI/README byte-concatenation SHA-256 8f5474ea2045bf0fac6a9e4295c444b20396a683622ca04aa0205c6af7925487

### Handoff

#### Objective
Deliver a reproducible CPU verification environment without changing frozen numerical behavior.
#### Changes
Pin the CI CPU wheel and the documented setup to torch 2.9.0.
#### Evidence
The existing exact tests passed locally on 2.9.0. Hosted failure is run 34067002355; no assertion
is weakened. Run the mandatory full gate before committing this environment-only repair.
#### Assumptions
A tested dependency version is the verification baseline, not proof of all-version compatibility.
#### Uncertainties
The post-pin hosted result is pending. No newer-release parity or reconstruction rerun is claimed.
#### Review focus
Environment pin consistency, unchanged assertions/source/evidence, and CPU-first setup.
#### Protected actions not taken
No scientific result/default promotion, old-evidence rewrite or independent acceptance.
#### Recommended next action
Verify, push the bounded repair under the user's delivery authorization, and inspect hosted CI.

### Review

#### Verdict
Provisionally accepted (self-reviewed)
#### Self-reviewed
Yes
#### Correctness
Both setup surfaces select the already exercised release; the mathematical implementation is unchanged.
#### Evidence quality
Direct hosted failure and local full-gate evidence; post-pin hosted confirmation remains pending.
#### Simplicity
A dependency pin avoids changing exact parity semantics during artifact delivery.
#### Missing cases
Newer PyTorch compatibility requires a separate investigation.
#### Required changes
Pass the mandatory pre-commit gate and preserve this provisional review scope.
#### Optional improvements
A later dependency-upgrade task can assess the changed quantile arithmetic explicitly.

## Notes
This is an environment repair within the user's commit/merge/push request, not a new experiment.

### Local verification (2026-09-07)
The mandatory full `./scripts/verify.sh` gate passed on the pinned local environment after this
repair. The log is retained locally with the private work-handoff bundle. Hosted confirmation
remains pending; the review disposition remains explicitly provisional.
