# HIER-034 — Fixed-geometry basis cache

## Context
The fixed-geometry color projection repeats tile and weight construction on every operator call.

## Goal
Compare streaming, cached scatter, and sparse operators with build cost, memory, and parity.

## Non-goals
- No maintained default change, count/rate claim, or cache reuse after geometry changes.

## Acceptance criteria
- [ ] Bounded immutable cache and forward/transpose oracle tests.
- [ ] Frozen protocol, distinct prospective review, and clean-source evidence.
- [ ] Portable report, independent audit, synchronized docs/ARA, full verification.

## Interfaces touched
`additive_basis.py`, `contraction_refinement.py`, tests and task-scoped experiment.

## Depends on
HIER-031/032/014, ADR-0006

## Agent workflow
- Driver: codex-root
- Reviewer: codex-overnight-protocol-reviewer
- Turn: driver
- Reviewed revision: pending

### Handoff log
Protocol and implementation are prospectively approved at the digest below. The clean source
checkpoint is4b2c79f2a97e0bde5109d11ab717edd5881a1ca1. Formal timing remains unlaunched while an
unrelated GPU workload is active; no other process was stopped and no speed claim is made.

## Notes
Overnight research authorized September 5, 2026.

## Frozen protocol

The executable PROTOCOL in scripts/experiments/hier034_basis_cache.py is authoritative for
the matrix and resolved choices. Its digest additionally binds the driver, report/checker,
projection/cache, field adapters, metrics, maintained reference, and owned CUDA sources.
Recompute with: python scripts/experiments/hier034_basis_cache.py --print-protocol-digest.

- Question/null: does retaining finite-support weights reduce complete fixed-geometry projection
  time? Null: neither cache achieves at least 1.1x median paired speedup with every parity gate.
- Arms: unchanged off; cached scatter; two-direction CSR. Same PCG defaults (48 maximum
  iterations, tolerance 1e-6, ridge 1e-8, input start/regularization, transaction selection,
  absolute coefficient bound 16), C0 fade, three-sigma support, float32, CUDA additive, chunk256.
- Data: three procedural 128x128/144-row families (smooth overlapping, rotated thin, irregular
  mask), seeds 0/1/2; one hash-bound exposed HIER-031 1200x1038/N7000 field with colors multiplied
  by 0.97. This is a color-solver workload, not a geometry/quality improvement or held-out assay.
  Original geometry and raw mask remain fixed. Existing HIER-031 diagnostic lineage persists.
- Pairing: family/seed/repeat. Six repeats cover all six lexicographic backend-order permutations.
  Each cell runs in a new process with all three libraries/backends warmed on synthetic seed77.
  Warmup, input IO, post-run perceptual scoring, export, and cold validation are outside the
  primary interval; the complete projection call, including cache build, all PCG/checkpoint work,
  and maintained replay, is inside it. GPU synchronization brackets that interval.
- Resources: RTX3050, one torch CPU thread, 256MiB retained-cache ceiling, 600-second worker
  timeout. Peak allocated/reserved GPU memory spans the entire projection; process peak RSS also
  includes imports/warmup/input loading. Neither is represented as cache storage alone.
- Integrity: exact count and geometry, cold decoded coefficients, raw active-pixel cold/maintained
  parity <=2e-4. Cached-versus-off active-pixel difference <=2e-4 and raw SSE discrepancy
  <=1e-4*max(off_SSE,1e-6). Inspect selected-iteration/transaction disagreement; any disagreement
  prevents a blanket interchangeable-backend conclusion. Record all coefficient differences.
  Compare every checkpoint's iteration, bounded/selectable/transaction booleans and trace length;
  normalized-violation disagreement beyond absolute 1e-6 also excludes interchangeability.
- Report: raw masked PSNR/SSE, display black-matted MS-SSIM/LPIPS, attempted/selected iterations,
  forward/transpose counts, complete checkpoint traces, build time, retained bytes, measured wall
  time and memory; target/field/reconstruction/error/curve artifacts and source/input hashes.
- Decision: workload-specific median paired projection acceleration only if all six pairs are
  complete, pass parity, agree on selection, and execute at least two iterations. Report ratios
  and dispersion within each workload; do not treat repeats as independent source images.
- Missing/OOM/timeout/nonfinite/budget errors remain explicit rows and exclude that workload from
  positive selection. No limit increase, selective rerun, in-place repair, or threshold rescue.
- Formal run uses a fresh results/hier034_basis_cache_2026-09-05 directory, from a clean reviewed
  commit/worktree, with --base-bundle pointing to the frozen HIER-031 bundle and
  --approved-protocol-digest set to the distinct review receipt. --smoke uses synthetic seed77
  only, one backend order and three iterations, with preserved dirty source; it is wiring only.

## Design sources
docs/research/2026-09-05-overnight-research-portfolio.md records the research alternatives.

### Protocol review

#### Reviewer
codex-overnight-protocol-reviewer

#### Verdict
Approved

#### Protocol digest
a5be4997c39a13d59eada1962a67837ff447bf9145e44b5dcca3eaaaacb66436

#### Digest scope
Canonical executable PROTOCOL plus SHA256 of every file in the driver's SOURCES list,
including projection/cache, semantic adapters, maintained render/CUDA sources, metrics and
portable report/checker. Recomputed before and after independent review, unchanged.

#### Outcomes accessed
No

#### Review focus
Paired saved-state integrity; all-checkpoint decisions; build-inclusive timing; six-repeat
counterbalancing/completeness; retained and peak memory scope; error preservation; prospective
source/commit identity; fault-injection tests. Independent focused gate: 32 tests passed.
Approval is prospective and workload-specific, not a result or default approval. Initial
intermediate digest 24bb46bdc7153e8bda77d2210cdf1b620eb48acc7fa9505d20efe0df009d4a9a was not
approved because decision and provenance gates were incomplete; the corrections above close
those issues. Post-run independent evidence audit remains required.
