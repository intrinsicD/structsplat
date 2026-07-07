# HIER-003: Pyramid equal-iteration diagnosis (fix or retire HIER-001)

**Status: todo.** The pyramid currently *loses* -1.0 to -1.4 dB under equal budgets; determine
whether that is the idea failing or the iteration accounting failing, then fix or retire.

## Context
`pyramid=pyramid` scored 20.29 vs 21.69 single-stage in the fit004 controls
(`fit004-residual-add-controls-2026-07-03`) and has lost every comparison since. But the
comparison may be structurally unfair to the pyramid: stage-search runs 2 levels with fractions
(0.35, 0.65) and splits the *same total iteration count* across levels, so the full field is only
co-trained for the final level's share of iterations, and the level-0 prefix is trained at full
LR against a target it can never match (it holds 35% of the budget). Residual-driven
densification is also the mechanism behind `residual_add` refine — which *wins* at the fair
regime — so the primitive is sound; the schedule may not be.

## Goal
A controlled diagnosis separating the densification idea from its scheduling, ending in either a
competitive pyramid default-candidate or an honest retirement of HIER-001.

## Protocol
Fair regime (exact CUDA, max-side 768, difficult-4, budgets {2000, 5000}), arms:
1. single-stage baseline (1500 iters);
2. pyramid as-is (iteration split, 1500 total) — reproduces the known loss;
3. pyramid with equal *full-field* iterations (coarse levels get extra iterations on top, full
   field still gets 1500 — measures the ceiling if scheduling were free);
4. pyramid with cosine LR spanning the whole run (ADR-0010 note) and level fractions
   (0.1, 0.9) — a cheap schedule fix;
5. `refine=residual_add` with a growth schedule matched to the pyramid's level budgets — the
   refine-mode twin of the same idea, isolating "pyramid ceremony" (re-init, per-level tensors)
   from "progressive capacity".

## Acceptance criteria
- [ ] All five arms run and committed under `ara/evidence/hier003-*/` with paired deltas.
- [ ] A one-paragraph verdict in the evidence: mechanism (scheduling vs idea) identified.
- [ ] If arm 3 or 4 reaches parity (±0.1 dB): a follow-up task defines the fixed schedule and
      HIER-001 stays open. If not: HIER-001 status set to failed-with-findings, pyramid demoted
      to a prefix-LOD/streaming feature (its remaining genuine use), docs updated.
- [ ] Either way `tasks/INDEX.md`, HIER-001, and the `benchmark` skill notes updated in the same
      commit.

## Interfaces touched
`src/structsplat/pyramid.py`, `benchmarks/stage_search.py` (arm 3 needs an
iteration-accounting flag), `ara/evidence/`, `tasks/HIER-001-progressive-pyramid.md`.

## Depends on
HIER-001, HIER-002, FIT-004.
