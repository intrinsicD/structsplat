# ABL-005: Fitter-knob influence pass at the fair regime

**Status: todo.** The largest unclaimed deltas in the 2026-07 evidence are fitter knobs, not init
strategies — but none has been isolated at a decision-grade regime.

## Context
`codex_stage_top1` beat the shipped default by +0.26 dB with 108/120 paired wins on the MERGE-001
COCO confirmation (`merge001-coco-cuda-confirmation-2026-07-06`) using exactly three knobs:
`loss=charbonnier`, `density=variance`, `opacity=constant`. That paired delta is as large as any
init-strategy difference — yet it was measured only as a bundle, and only at the tiny 40-iter /
max-side-160 regime, which is known to rank refine modes opposite to the 1500-iter fair regime.
Separately, smokes showed `refine=moment_preserving` (FIT-007), `color_solve` (FIT-005), and
`color_basis=affine` (CORE-006) as promising, and `lr_schedule=cosine` has never been tested at
scale. ADR-0010's influence mode exists precisely for this and has not been pointed at these knobs.

## Goal
One-factor-at-a-time paired deltas for the seven promising fitter/config knobs at the fair regime,
so promotion decisions (each via its own ADR) rest on isolated, regime-correct evidence.

## Protocol (fixed before running; BENCH-002 rules apply)
- Mode: `structsplat stage-search --mode influence` around the shipped ADR-0009 baseline.
- Variant axes (one non-baseline value each): `loss=charbonnier`, `density=variance`,
  `opacity=constant`, `refine=moment_preserving`, `lr_schedule=cosine`, `color_solve=every10`,
  `color_basis=affine`. 8 arms total including baseline.
- Fair regime: exact CUDA renderer, 1500 iters, max-side 768.
- Images: the 8-image screen set (`kodim01,04,07,10,13,16,19,22`) for continuity with
  `abl004-stage-screen-8img-cuda-2026-07-04`.
- Budgets {2000, 5000, 10000}; seeds {0, 1}; `--target-psnrs 28 30 32` (never a lone 35).
- Cost estimate: 8 x 8 x 3 x 2 = 384 cells, ~7 GPU-h on the RTX 3050. Shard with `--resume`.

## Acceptance criteria
- [ ] `influence.md` with paired ΔPSNR / ΔMS-SSIM / ΔAUC / Δiters-to-target / Δfit-s per knob,
      per budget, committed under `ara/evidence/abl005-*/` with `config.json`.
- [ ] A follow-up combo run: baseline vs the additive composition of every knob that won its
      paired test (knobs can interact; the combo must be measured, not assumed).
- [ ] Each knob promoted to default gets its own ADR citing this run; knobs that lose are recorded
      as parked in `ara/logic/claims.md`.
- [ ] `tasks/INDEX.md` and README updated in the same commit as any default change.

## Interfaces touched
`benchmarks/stage_search.py` (no new code expected — protocol only), `ara/evidence/`,
`docs/adr/` for promotions.

## Depends on
ADR-0010, FIT-005/006/007 (axes exist), CORE-006, BENCH-002. Pairs with ABL-006 (shares GPU
budget; this task is the higher-information spend and should run first).
