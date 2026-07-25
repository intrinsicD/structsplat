---
name: structsplat-results-audit
description: Adversarial referee pass ("scientist pass") over StructSplat's results, quantitative claims, benchmark artifacts, and derivations. Use before publishing or promoting a claim, after an experiment or benchmark run, before changing a default or task/ADR status, when reviewing a results-bearing change, or whenever actual-rate, renderer-parity, development/held-out, provenance, statistics, or ARA claim integrity needs an independent check. Do not use for generating new research directions (that is the structsplat-research-ideation skill).
license: MIT
metadata:
  version: "1.0.0"
---

# Results Audit ("scientist pass")

> **Provenance.** Distilled 2026-07-15 from StructSplat's BENCH-002
> validity repairs, the BENCH-007 actual-rate killing gate, the 168 KiB
> count-versus-stream-rate correction, exact-CUDA cold-field parity work, and
> FIT-018–020's source-bound negative-result practice. It incorporates the
> sibling-repository referee pattern for finding errors after a run appears
> complete.

## Stance

Act as a referee, not the producing author. Assume at least one number, scope
label, comparison, hash binding, or interpretation is wrong until checked. The
deliverable is corrections, narrower wording, explicit negative results, and
updated ARA claims, never reassurance. "Everything checks out" requires a
completed claim table with every row bound to independently inspected evidence.

Load `structsplat-core` first, `structsplat-benchmark` for harness-specific rules, `structsplat-review` before
completion, and `structsplat-docs-sync` for claim updates. This skill audits evidence; it
does not authorize a rescue experiment, protected confirmation data, or a new
method.

## When to run

- Before a quantitative, compression, quality, speed, novelty, or default claim
  enters `README.md`, an ADR/task, or `ara/logic/claims.md`.
- After an ablation, actual-rate run, mechanism guard, native comparison, or
  evidence session.
- Before task closure/default promotion or review of a results-bearing PR.
- On request: "audit", "referee pass", "scientist pass", "verify the claims".
- Periodically over old claims and completed gates, not only the newest rows.

## Procedure

### 1. Inventory the claims

Sweep `README.md`, `CLAUDE.md`, relevant tasks/ADRs, `benchmarks/README.md`,
result configs/manifests, `ara/evidence/`, and `ara/logic/claims.md`. Build:

| # | Claim | Kind + scope | Evidence path | Config/source bound? | Executed now? |
|---|---|---|---|---|---|

Kind is derived, measured, descriptive, or asserted. Scope is plumbing,
calibration, development, held-out, external/native context, or production.
An asserted claim, missing split label, or public claim without a bounded ARA
row is already a finding. Reconcile statement, status, falsification criteria,
proof, dependencies, and provenance; never silently promote maturity.

### 2. Verify actual-rate accounting

For compression/RD claims require a persisted self-contained stream and
recompute `bpp = 8 * complete_stream_bytes / (original_width * original_height)`.
Count headers, ranges, opacity/framing, attributes, and entropy payload. Gaussian
count, parameter BPP, analytical payload, nominal bits, or interpolated bytes are
not actual rate. Verify native pixel dimensions/hashes, exact integer byte caps,
explicit missing points, independent fits per count, equal candidate/search
budgets, cold decode, central rescoring, nondominated envelopes, and BD-rate only
on the common measured interval without extrapolation.

### 3. Enforce frozen gates and data splits

Treat BENCH-007 Stage 1 as a development killing pilot, not held-out evidence.
Its gate failed quality-at-both-rates, time, and texture guards against the
strongest local gradient control, so `stage2_authorized=false` and DIV2K
validation remains untouched. Do not rescue the claim with the favorable
0.5-bpp slice, tune exposed images, or open Stage 2. A materially new formulation
needs a new null, disjoint development screen, frozen gate, and authorization
before confirmation.

### 4. Audit renderer and decoded-field parity

Freeze the equation separately from its implementation. `renderer=normalized`
is the PyTorch oracle; `renderer=cuda` is the owned exact-CUDA implementation of
that normalized equation. `gsplat`, additive comparators, and local paper-
mechanism transplants are not interchangeable. Verify identical renderer,
fitter, horizon/checkpoint rule, optimizer/count budget, codec/search, QAT,
clamp/scoring policy, and decoded pixels across arms.

Validate cold parity at the decoded-field boundary: persisted bytes reproduce
the decoded-state hash and satisfy the frozen tolerance. Do not use two exact-
CUDA renders as a bit-exact oracle; atomic accumulation order can vary by ulps.
CPU replay may be bit-exact; GPU evidence is device/version/source-bound.

### 5. Bind configuration, source, and artifacts

Verify manifest and source SHA-256s, input/decoded-pixel hashes, resolved config,
seed, renderer/device, commit/branch, dirty flag, tracked diff, untracked source
snapshot, and library/CUDA versions. A dirty run is usable only when its exact
executed source is preserved. On resumed journals select the latest valid row per
stable cell key and include error/missing rows in completeness accounting. Never
copy a number from prose or a stale journal when raw tables can be recomputed.

### 6. Recompute statistics and resources

Recompute deltas, directions, ratios, units, and decision predicates from raw
rows. Preserve pairing; average correlated seeds/budgets within source image
before image-cluster bootstrap. Rate points are not independent samples. Check
repeat counts, intervals, multiple-comparison adjustment, strongest-control
selection, missing curves, and per-image/family heterogeneity.

Charge the resource scope actually claimed: initialization, fit, QAT, candidate
search, encode, cold decode, render, RSS, and VRAM. An exhaustive oracle work
ceiling, proxy FLOP/visit count, or faster isolated renderer is not an end-to-end
speedup. Quality, convergence, actual RD, and implementation speed are separate
claims.

### 7. Audit comparison and interpretation scope

Distinguish common-harness analogues from official native executions; label local
transplants `local_<mechanism>_control`. Keep plumbing/calibration/development,
procedural/single-field, descriptive/post-hoc, and held-out evidence distinct. A
win at one metric, budget, horizon, image, or renderer is not global dominance.
Controls and secondary diagnostics cannot rescue a failed primary gate. Verify
figures and public prose carry the same scope as their raw artifact.

### 8. Dispose of every claim in the same change

For each row: **confirm**, **narrow**, **refute**, or **retire**. Update public
wording with `ara/logic/claims.md`, `ara/evidence/README.md`, the task decision,
ADR consequences, and trace/staging records as applicable. Preserve failed gates
and negative results as history. Stage interpretation until the ARA closure rules
permit crystallization; do not fabricate a closure signal.

### 9. Report

End with the claim table, raw recalculations, corrections, integrity failures,
development-versus-held-out state, exact rerun commands, and unresolved gaps.
State what ran on CPU, what ran on which GPU, what was only inspected, and what
remains unauthorized.

## Anti-patterns (hard nos)

- Tuning a frozen gate/tolerance or consuming confirmation after a failed gate.
- Comparing analytical/count rate with complete-stream actual rate.
- Calling a local transplant an upstream/native method execution.
- Treating two nondeterministic CUDA renders as cold-stream parity.
- Deleting failed/missing rows, negative claims, or historical contradictions.
- Promoting a claim without config, source, split, and raw-evidence binding.
- Reporting a work ceiling or proxy counter as measured acceleration.

## Repository anchors

`CLAUDE.md` + `structsplat-core` — invariants and routing · `structsplat-benchmark` and
`benchmarks/README.md` — validity protocol ·
`tasks/BENCH-007-actual-rate-structure-phase-diagram.md` — frozen actual-rate
gate · `benchmarks/actual_rate_phase_diagram.py` — executable accounting ·
ADR-0003/0011 — normalized equation and exact CUDA · `ara/logic/claims.md` —
claim ledger · `ara/evidence/` — source-bound proof · `structsplat-docs-sync` — same-change
documentation discipline.
