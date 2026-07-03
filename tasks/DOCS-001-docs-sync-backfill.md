# DOCS-001: Docs-sync backfill (post-merge staleness, missing ADRs, ara scaffold)

**Status: todo.** From the 2026-07-03 repo review. The last feature commit (3f9646c, +4,942
lines) bypassed the repo's own docs-sync rule: zero ADRs, zero task files, no INDEX updates.
This task pays that debt in one pass.

## Context
1. **Stale/contradictory task status.** MERGE-001 is simultaneously 'todo' (tasks/INDEX.md),
   'partial' (its task file, line 3), and de-facto merged (PR #1 in git history). PORT-001
   says 'todo' although an exact CUDA renderer (forward + analytic backward + autograd +
   parity tests) exists and is the default renderer of `cross_repo_matrix_compare.py`.
   FIT-001's "Optional opacity parameterization" checkbox is open although fully implemented.
2. **Stale docs.** README, docs/architecture.md, benchmarks/README.md, and ABL-002 omit the
   three quadtree strategies, the scale-cap axis, the residual_tensor refinement modes, and
   the three new renderer modes (cuda/cuda_additive/gsplat).
3. **Missing ADRs** for two real decisions that shipped in 3f9646c: the owned exact CUDA
   renderer (+ the gsplat comparator's non-equivalent alpha semantics) and per-Gaussian scale
   caps (`GaussianField.scale_max`).
4. **ara/ scaffold is partially hollow.** PAPER.md's layer index lists `ara/src/configs/` and
   `ara/src/kernel/` which do not exist; `logic/problem.md` and `logic/claims.md` are empty
   headers despite 14 observations and 3 staged claims; N13/N15 cite evidence IDs absent from
   the evidence index; N13–N17 mix ID-based and raw-path citation conventions; the blockers
   doc mutates in place ("Status after implementation" inside recommendation 5), blurring
   analysis-time claims vs later results; comparison.md's "33 passed in 108.55s" and the
   session trace's "79–81 tests in ~5s" describe different code states with no annotation.

## Goal
Docs, tasks, ADRs, and the research trace describe the repository that actually exists.

## Acceptance criteria
- [ ] MERGE-001 status reconciled (INDEX + task file agree; note records that the PR merged
      before the confirmation gate, with the remaining gate re-stated).
- [ ] PORT-001 updated to 'partial': records the existing exact (non-tiled) extension as an
      interim milestone with its parity tests, and re-scopes remaining work — tiled/culled
      gather kernel, shared-memory reduction of backward partials, block sizing by tile area,
      exact ellipse–tile intersection (Speedy-Splat-style; AABBs are worst-case for this
      repo's signature anisotropic Gaussians), optional top-k normalized mode (Image-GS),
      deterministic-accumulation option, throughput targets, RHI port.
- [ ] FIT-001's opacity checkbox closed with a pointer to the implementing commit/ADR-0009.
- [ ] README / docs/architecture.md / benchmarks/README.md / ABL-002 refreshed: 8 strategies,
      7 sampling modes, 5 renderer modes, scale-cap axis, refinement modes, new benchmark
      scripts (one line each; BENCH-003 owns the deeper benchmark README).
- [ ] ADR-0011 (owned exact CUDA renderer; gsplat comparator semantics) and ADR-0012
      (per-Gaussian scale caps) written; docs/architecture.md references them.
- [ ] ara/: problem.md and claims.md written (the prose exists in README/theory.md; staged
      claims promoted or explicitly parked with caveats); PAPER.md layer index matches the
      directories that exist; evidence index gains entries for N13–N17 (commands,
      environment, key numbers) with one citation convention; blockers doc post-analysis
      updates moved to a dated addendum/trace link; pytest evidence entries annotated with
      the commit hash they ran against.
- [ ] `grep -ri "todo" tasks/INDEX.md` matches only genuinely-open work.

## Interfaces touched
`tasks/INDEX.md`, `tasks/MERGE-001*.md`, `tasks/PORT-001*.md`, `tasks/FIT-001*.md`,
`README.md`, `docs/architecture.md`, `docs/adr/0011-*.md` (new), `docs/adr/0012-*.md` (new),
`docs/blockers_and_external_techniques.md`, `docs/comparison.md`, `benchmarks/README.md`,
`ara/**`. Docs only — no code.

## Depends on
— (independent; ABL-004 owns committing new empirical evidence).
