# BENCH-004: Sweep-cost controls (plateau exit, multi-target tables, proxy regime)

**Status: todo.** Compute is the binding constraint on every open experiment; make each GPU-hour
buy more decisions without weakening the protocol.

## Context
The confirmation manifest alone is ~26 remaining GPU-hours and 7% complete; ABL-005/006,
HIER-003, and INIT-008 all queue behind the same RTX 3050. Three protocol-level costs are
avoidable: (1) cells run all 1500 iterations even when PSNR plateaued at 600; (2) the headline
tables still key convergence to a lone `target_psnr=35` that is reached in ≤1/8 screen cells, so
the convergence half of the protocol is silently empty (the multi-target machinery already
exists); (3) screening happens either at a 40-iter/160-px regime that provably mis-ranks refine
modes, or at the full fair regime that costs ~30 GPU-min per image-budget — there is no
validated cheap regime in between.

## Goal
Three harness upgrades, each preserving BENCH-002 validity guarantees.

## Design
1. **Plateau early exit** (`--early-exit`, default off): stop a cell when the best-PSNR
   improvement over the last W iterations falls below eps (suggested W=150, eps=0.02 dB);
   record `stopped_at`, report AUC over the *nominal* horizon by holding the last value (states
   its convention in the row — one metric convention per row). Refine-mode cells must not exit
   before the final growth wave lands at budget (equal-budget rule). Validate by re-running one
   completed 8-image screen slice: rankings and paired-delta signs unchanged, wall-clock saving
   reported.
2. **Multi-target headline tables**: `summary.md`/leaderboard writers use `target_psnrs`
   {28, 30, 32} everywhere `iters_to_target` appears today; a lone unreachable 35 disappears
   from headline tables (kept in raw rows for continuity).
3. **Calibrated proxy regime**: run the 11-arm 8-image screen at candidate proxy regimes
   (max-side 384 / 500 iters first) and compute Spearman rank correlation + paired-delta sign
   agreement against the committed full-regime screen
   (`abl004-stage-screen-8img-cuda-2026-07-04`). Acceptance for a proxy: rho >= 0.9 on strategy
   ranking and >= 90% sign agreement on paired deltas |Δ| > 0.1 dB. The validated proxy gets
   documented in the `benchmark` skill as the designated screening regime; full fair regime
   stays mandatory for promotions.

## Acceptance criteria
- [ ] Early exit implemented in `fit.py` history tracking + benchmark plumbing, off by default,
      with the validation rerun committed as evidence.
- [ ] Multi-target tables land in `abl004_confirmation`, `stage_search`, and
      `fair_density_control_compare` summary writers; tests updated.
- [ ] Proxy calibration evidence committed under `ara/evidence/bench004-*/` with the rank
      correlation table; `benchmark` skill updated with the proxy-regime contract (screen on
      proxy, decide on fair).
- [ ] BENCH-002 checklist re-audited after the changes (no silent caps, one convention per row).

## Interfaces touched
`src/structsplat/fit.py`, `benchmarks/common.py`, `benchmarks/stage_search.py`,
`benchmarks/abl004_confirmation.py`, `benchmarks/fair_density_control_compare.py`,
`.claude/skills/benchmark/`, `tests/`.

## Depends on
BENCH-002, FIT-003. Unblocks cheaper ABL-005/ABL-006/HIER-003/INIT-008 runs — schedule first
among the benchmark tasks.
