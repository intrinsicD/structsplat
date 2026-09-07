# CORE-021 — Portable verification corrections

## Context

The pinned CPU gate exposed two pre-existing failures: a rank-deficient affine-design test
assumes LAPACK returns a nonzero trailing singular value, and the current decode worker relies
on opened-descriptor timestamps to notice a replaced artifact pathname. A rename may preserve
those timestamps on the active filesystem.

## Goal

Make the affine diagnostic test portable and detect artifact path replacement independently
of opened-descriptor timestamp changes.

## Non-goals

- No affine-carrier numerical implementation, metric, gate, threshold, codec or renderer change.
- No edits to frozen BENCH-014/COMP-011 task text, protocol constants, captured source snapshots
  or existing evidence; no experiment, replay rerun, protected data access or scientific claim.
- No changes to the independently reviewed CORE-020 implementation or tests.

## Acceptance criteria

- [x] Rank-deficient designs remain rejected, with positive non-NaN condition diagnostics whether
      the trailing singular value is exactly zero or a machine-dependent nonzero residue.
- [x] After reading an artifact, its relative path is rewalked from the original root descriptor
      without following symlinks and its current leaf identity is compared with the opened file;
      leaf/parent replacement, removal and symlink substitution fail closed at that boundary.
- [x] Synthetic tests cover path changes while the opened descriptor's stat snapshot is unchanged,
      and existing in-place mutation, no-follow access and sealed-byte loading checks pass.
- [x] A distinct reviewer checks the exact diff; task/index/session state is synchronized.
- [x] `./scripts/verify.sh` passes on the pinned CPU environment.

## Interfaces touched

`benchmarks/ssp2v_decode_worker.py`, `tests/test_ssp2v_decode_worker.py`,
`tests/test_affine_carrier_core.py`, this task, `tasks/INDEX.md`, and `tasks/SESSION-BRIEF.md`.

## Depends on

BENCH-014, COMP-011

## Agent workflow

- Driver: Codex-structsplat-engineering
- Reviewer: Codex-cross-repo-review
- Turn: none
- Reviewed revision: source/test diff SHA-256 e2e20c8fbbf8ec945ee155b9c33f9a3c36327ba1c36f2b22bf82bac535b46d46

### Handoff log

### Handoff

#### Objective
Resolve the two baseline verification failures while retaining rejection and artifact identity checks.
#### Changes
Permit the mathematically valid infinite condition number for an exactly singular design while
requiring rejection and severe ill-conditioning. Rewalk the validated artifact path from the root
without following symlinks after reading; compare the leaf identity with the original opened file.
#### Evidence
Both baseline failures and six deterministic path-swap controls reproduced before repair. The two
focused test files pass 62 tests afterward. Ruff and structural checks pass; the full gate runs.
#### Assumptions
Read bytes stay bound to the original descriptor. No replacement file is opened for content.
#### Uncertainties
The final path check observes one boundary and does not lock the filesystem against later mutation.
#### Review focus
Descriptor cleanup, nested parent/leaf replacement, symlink/removal controls, unchanged-stat
counterexamples, singular-matrix rejection, and preservation of historical protocol/evidence bytes.
#### Protected actions not taken
No formal experiment, sealed outcome access, scientific promotion, historical source snapshot
mutation, commit, push, or external transfer.
#### Recommended next action
Independently review the exact source/test diff and close only after the full portable gate passes.

### Review

#### Verdict
Accepted
#### Self-reviewed
No
#### Correctness
Codex-cross-repo-review found no blocker. Revalidation starts at the original root and never
follows replacement symlinks or reads replacement content. Success and error paths release
owned descriptors. The SVD test permits mathematically valid positive infinity while retaining
rank-one diagnosis, severe ill-conditioning and solver rejection (ARA C77).
#### Evidence quality
The reviewer independently reproduced 62 focused passes, 20 leaf/ancestor substitution probes,
repeated success/error descriptor-cleanup probes, and four singular-design probes. Substitution
probes include identical-byte replacements, symlinks, dangling links, FIFOs and removal while
holding the original descriptor stat constant.
#### Simplicity
One shared parent-walk helper; unchanged production affine mathematics and protocol constants.
#### Missing cases
Identity is observed at the read boundary; subsequent filesystem mutation is not prevented.
#### Required changes
None. The independent review's full-gate condition is now satisfied.
#### Optional improvements
None required.

### Final verification (2026-09-07)
The complete portable verify.sh gate passed under torch 2.9.0+cpu and Ruff 0.15.20 with both
repository source trees on PYTHONPATH, MKL_THREADING_LAYER=GNU, and two BLAS/OpenMP threads.
Its local receipt is /tmp/structsplat-rtgs-collab/structsplat-verify-final.log. All reviewed
source/test digests remained unchanged. Tracked tests and the exact review boundary are the
durable evidence; local independent probe receipts remain under /tmp/structsplat-rtgs-collab/.
No commit, push, scientific/default promotion, or external artifact transfer was performed.
Claude review remains unperformed pending specific approval for the prepared code-only payload.

### Claude static review (2026-09-07)

After the user's explicit code-payload approval, Claude (`claude-fable-5-1`, tools disabled)
reviewed the cross-repository patch and reported: "No blocking findings within this patch scope."
This was a static review, with no tests executed by Claude. The patch packet SHA-256 is
66de54dc3b22162caccb5664e484d0f1082d41c063eb603c6522a464cdbc1527;
the verbatim review SHA-256 is
6aecbb0b1a517f94edceaf7758ca6ec381cf729f0d3970733df3500655b81ad7.
Local review and invocation receipts are in /tmp/structsplat-rtgs-collab/claude-review.md
and claude-review-receipt.json. Only approved source/test code was transmitted.

Claude confirmed no-follow traversal, leaf identity checking, and descriptor cleanup. Its
nonblocking diagnostic note is acknowledged: an OSError such as descriptor exhaustion is
reported as a changed path, while the original error is retained as the exception cause.
Rejection remains correct. The documented end-of-read check does not prevent a later swap or
detect a swap-and-restore interval. Claude required no CORE-021 source/test changes.

### Verification fixture follow-up (2026-09-07)

The full gate rerun after CORE-020's test-only Claude suggestion exposed a nondeterministic
pre-existing in-place mutation fixture: 1 failed, 2,494 passed, 77 skipped, 515 deselected.
Its receipt is /tmp/structsplat-rtgs-collab/structsplat-verify-claude-followup.log.
An implementation-agent probe reproduced unchanged mtime/ctime in 44 of 64 same-size writes;
the other 20 changed timestamps and were rejected. A stat comparison cannot infer a write
whose inode, size and timestamps all remain identical; sealed-content hash checks remain
authoritative for byte identity. This does not change the bounded path-identity contract.

The existing fixture now backdates mtime before opening, then performs its actual in-place
write and fsync. It still requires rejection and additionally verifies the mutation occurred,
inode and size stayed unchanged, mtime changed, and the resulting bytes are exact. There are
no sleeps or production changes. All 45 worker tests and 64 repeated regression invocations
passed. Independent reviewer Codex-cross-repo-review accepted the exact fixture-only delta
(self-reviewed: No), confirmed the worker and affine-test hashes were unchanged, reproduced
45 worker passes and 64 regression repetitions with no descriptor leaks, and passed diff checks.
Its local receipt is /tmp/structsplat-rtgs-collab/core021_fixture_followup_review.json.
The final source/test diff SHA-256 is
e2e20c8fbbf8ec945ee155b9c33f9a3c36327ba1c36f2b22bf82bac535b46d46;
the earlier implementation review was bound to
c122d7a77cd763301195340e20cd364e68f35857ffbc01f7facd7686afa7f56c.
The complete portable verification rerun passed: 2,495 passed, 77 skipped, 515 deselected in
327.12 seconds, followed by all lint/documentation/claim/task/workflow checks. Its local receipt
is /tmp/structsplat-rtgs-collab/structsplat-verify-claude-final.log. This satisfies the independent
follow-up review's full-gate condition. No production code changed after Claude's static review;
no commit or push was made.

## Notes

Changes apply to the current development source and synthetic tests. Historical source-bound
artifacts retain their original worker bytes and scientific dispositions. The new check observes
the root-relative path at the end of the read; it does not lock the filesystem against later
mutation and never reads replacement bytes.
