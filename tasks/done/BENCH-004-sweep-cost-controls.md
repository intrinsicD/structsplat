# BENCH-004: Sweep-cost controls (plateau exit, multi-target tables, proxy regime)

**Status: done.** Compute is the binding constraint on every open experiment; make each GPU-hour
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
- [x] Early exit implemented in `fit.py` history tracking + benchmark plumbing, off by default,
      with the validation rerun committed as evidence.
- [x] Multi-target tables land in `abl004_confirmation`, `stage_search`, and
      `fair_density_control_compare` summary writers; tests updated.
- [x] Proxy calibration evidence committed under `ara/evidence/bench004-*/` with the rank
      correlation table; `benchmark` skill updated with the proxy-regime contract (screen on
      proxy, decide on fair).
- [x] BENCH-002 checklist re-audited after the changes (no silent caps, one convention per row).

## Result
Implemented 2026-07-07. The accepted cheap screen is max-side 512 / 750 iterations. Evidence:

- `ara/evidence/bench004-proxy-calibration-2026-07-07/`: rejected 384/500 because the 5k budget
  missed the acceptance gate (`rho=0.8636`, sign agreement 89.8%).
- `ara/evidence/bench004-proxy-calibration-512-750-2026-07-07/`: accepted 512/750 against the
  committed full 8-image screen (`rho=0.9364/0.9182`, sign agreement 91.8%/93.9% at 2k/5k).
- `ara/evidence/bench004-early-exit-validation-512-750-5k-2026-07-07/`: opt-in early exit on the
  5k 8-image slice kept the top AUC/PSNR arm and preserved all AUC paired signs above 0.1 dB;
  final PSNR retained top-1 but swapped two close pairs, so early-exit rows are for convergence
  screening and promotion still uses full-horizon fair runs. The slice saved 1.1% iterations and
  1.7% fit seconds at `--early-exit-window 150 --early-exit-min-delta 0.02`.

BENCH-002 re-audit: row-level `n_gaussians` remains capped to `budget`; early-exit is opt-in and
records `iterations_run`/`stopped_at`; AUC rows state the convention via `auc_psnr_horizon`; configs
record resolved targets, early-exit settings, device, renderer, and versions; raw 35 dB
iters-to-target remains available but headline summaries use 28/30/32.

## Interfaces touched
`src/structsplat/fit.py`, `benchmarks/common.py`, `benchmarks/stage_search.py`,
`benchmarks/abl004_confirmation.py`, `benchmarks/fair_density_control_compare.py`,
`.claude/skills/benchmark/`, `tests/`.

## Depends on
BENCH-002, FIT-003. Unblocks cheaper ABL-005/ABL-006/HIER-003/INIT-008 runs — schedule first
among the benchmark tasks.
