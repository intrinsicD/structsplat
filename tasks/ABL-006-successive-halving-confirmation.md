# ABL-006: Successive-halving execution of the remaining confirmation

**Status: partial.** Harness and stage-1 decision are complete; stage 2/3 GPU runs and README
winner update remain open. Supersedes the flat execution plan of the ABL-004 confirmation manifest
(the
protocol and arms are unchanged; only the run order and stopping rule change).

## Context
The ABL-004 confirmation (28 images x 3 seeds x 3 budgets x 6 arms = 1,512 cells) is 108 cells
(7%) complete after six bounded shards, at an estimated ~26 GPU-hours remaining on the local
RTX 3050. Meanwhile the strategy question is already directionally answered (INIT-007): the
interesting remaining questions are *which* structured arm wins per budget and whether
Floyd-Steinberg/iso controls stay dominated. A flat sweep spends most of its compute on arms and
budgets whose outcome is no longer in doubt.

## Goal
Finish the confirmation with the same per-cell protocol (1500 iters, max-side 768, exact CUDA,
`--resume`-compatible with the existing 108 cells) but staged as successive halving, cutting cost
by roughly half to two-thirds without weakening the statistical claim for the arms that matter.

## Protocol
1. **Stage 1 — all 6 arms, budget 2000, 28 images, seeds {0,1}** (336 cells; existing cells
   reused). Rank by paired ΔPSNR vs the current leader with bootstrap CIs (the harness already
   computes `paired_deltas_vs_baseline.csv`).
2. **Stage 2 — survivors only** (arms whose CI overlaps the leader), budget 5000, 28 images,
   seeds {0,1}.
3. **Stage 3 — finalists**, budget 10000, plus seed 2 for every finalist at every budget so the
   headline claim is 3-seed as originally specified.
4. Eliminated arms are reported as eliminated-at-stage-k with their paired delta — not silently
   dropped (BENCH-002 no-silent-caps rule).

## Acceptance criteria
- [x] A `plan`/`run` mode in `benchmarks/abl004_confirmation.py` (or a thin wrapper) that emits
      the staged cell list and honors elimination decisions recorded in a committed JSON.
- [x] Elimination rule stated in the config artifact before stage 1 completes (no post-hoc
      tuning): CI method, confidence level, pairing key.
- [ ] Final `summary.md` + `leaderboard.csv` + elimination trail committed under
      `ara/evidence/abl006-*/`; ABL-004's remaining acceptance boxes ticked by this evidence.
- [ ] README hypothesis status paragraph updated with the per-budget winners.

## 2026-07-07 partial
Added `halving-plan` / `halving-run` subcommands to `benchmarks/abl004_confirmation.py`.
The staged plan writes `abl006_plan.csv`, `abl006_config.json`,
`abl006_elimination_decisions.json`, `abl006_run_groups.json`, and
`abl006_elimination_trail.csv`. The analysis path now supports staged expected cells so eliminated
arms are not counted as missing high-budget cells. Harness evidence is committed under
`ara/evidence/abl006-halving-harness-2026-07-07/`.

Ran two bounded stage-1 warm-up shards after preparing the full Kodak-24 + COCO4 image set:
`kodim01` and `kodim02`, budget 2000, seeds {0,1}, all six arms. Evidence is under
`ara/evidence/abl006-stage1-shard1-2026-07-07/` and
`ara/evidence/abl006-stage1-shard2-2026-07-07/`.

Then completed the remaining stage-1 cells in one resumable shard. Stage 1 is 336/336 complete.
The leader is `quadtree_wse` at 26.5477 dB mean PSNR, statistically tied with `aniso_onedge`
(-0.0004 dB, 95% CI [-0.1453, 0.1351]). Stage-2 survivors are `quadtree_wse` and
`aniso_onedge`; `aniso_flanking`, `quadtree_hybrid`, `iso_blue_noise`, and `floyd_steinberg` are
eliminated by the frozen CI rule. Evidence is under
`ara/evidence/abl006-stage1-complete-2026-07-07/`. The staged analysis is now 336/448 complete
with 112 stage-2 cells pending.

## Interfaces touched
`benchmarks/abl004_confirmation.py`, `ara/evidence/`, `tasks/ABL-004-controls-and-full-run.md`,
`README.md`.

## Depends on
ABL-004 (protocol + first 108 cells), BENCH-002. Feeds INIT-007 (default flip cites this run).
