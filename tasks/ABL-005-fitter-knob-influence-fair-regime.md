# ABL-005: Fitter-knob influence pass at the fair regime

**Status: partial.** The largest unclaimed deltas in the 2026-07 evidence are fitter knobs,
not init strategies — but none has been isolated at a decision-grade regime. The fair protocol is
now split into CUDA-native knobs and an affine quality-only shard so the known affine renderer
fallback cannot contaminate speed claims.

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
- Mode: `structsplat stage-search --mode influence` around the shipped ADR-0009 + ADR-0013
  baseline.
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

## Notes

- 2026-07-07: Added the missing harness support needed to attempt the run: `stage-search --resume`
  / `--max-new-cells`, CUDA-compatible color solve for normalized renderer modes, and an affine
  color fallback to the exact PyTorch reference on CUDA. Evidence:
  `ara/evidence/abl005-harness-dryrun-and-fair-blocker-2026-07-07/`.
- The exact 8-arm command dry-runs successfully at tiny scale, but the fair-regime all-seven run is
  not decision-grade yet: the `color_basis=affine` arm falls back to the reference renderer because
  the custom CUDA extension has no affine-color backward kernel. In the first fair shard, three
  normal CUDA 2k cells finished at ~5 fit seconds each; the affine 2k cell was interrupted after
  ~3 minutes without finishing. This makes the affine speed delta implementation-confounded.
- 2026-07-09: Unblocked the public `structsplat stage-search` workflow for the six CUDA-native
  ABL-005 knobs by exposing the benchmark module's resume/shard flags, factored refine axes, and
  early-exit controls in `src/structsplat/cli.py`, and by making console-script benchmark imports
  find the repo-level `benchmarks` package. A tiny CPU influence smoke through the public command
  completed 14/14 cells and wrote `index.html`; evidence:
  `ara/evidence/abl005-cli-unblock-2026-07-09/`. The HTML report now marks the best paired-delta
  variants for PSNR, MS-SSIM, AUC, and fit time. This does not resolve the affine-speed blocker.
- 2026-07-10: Split the blocked protocol into two reproducible shard scripts. Use
  `scripts/run_abl005_cuda_native_influence.sh` for the six CUDA-native knobs
  (`loss=charbonnier`, `density=variance`, `opacity=constant`, `refine=moment_preserving`,
  `lr_schedule=cosine`, `color_solve=every10`) under `renderer=cuda`; it writes `index.html` and
  paired `influence.md` and is valid for quality, convergence, and fit-time deltas. Use
  `scripts/run_abl005_affine_quality_influence.sh` for `color_basis=affine`; it pins the exact
  reference renderer by default and is quality/convergence evidence only until native CUDA affine
  backward exists. A tiny two-image smoke completed both scripts and wrote
  `results/abl005_cuda_native_influence_smoke/index.html` plus
  `results/abl005_affine_quality_influence_smoke/index.html`; committed smoke note:
  `ara/evidence/abl005-split-protocol-smoke-2026-07-10/run.md`.
- 2026-07-10: Started the full fair-regime CUDA-native run on the Kodak screen. The first shard
  completed three `kodim01`/2k/seed0 cells (baseline, `density=variance`, `opacity=constant`) and
  wrote a partial `results/abl005_cuda_native_influence/index.html`; the next cell
  (`color_solve=every10`) was interrupted after proving too slow for a broad 21-cell shard. The
  CUDA-native runner now supports per-axis env overrides (`DENSITY_MODES`, `OPACITY_MODES`,
  `COLOR_SOLVE_MODES`, `PIXEL_LOSSES`, `LR_SCHEDULES`, `REFINE_MODES`) so the remaining fair run
  can be resumed by knob group rather than blocking all progress on the slow color-solve arm.
  A fast-axis resume then completed `loss=charbonnier`, `lr_schedule=cosine`, and
  `refine=moment_preserving` for the same cell, bringing the shard to 6/336 CUDA-native fair
  cells. Evidence: `ara/evidence/abl005-cuda-native-fair-shard-2026-07-10/`. Single-cell deltas
  are not promotion evidence, but they show the shard is producing paired metrics:
  `opacity=constant` (+1.5168 dB PSNR, +0.3783 AUC, +1.411 s fit),
  `lr_schedule=cosine` (+0.9614 dB, +0.0778 AUC, +0.245 s fit), `loss=charbonnier`
  (+0.3367 dB, -0.0910 AUC), `moment_preserving` (+0.4769 dB, -0.0019 AUC), and
  `density=variance` (-0.0203 dB, -0.2181 AUC). A follow-up fast-axis resume completed the
  matching `kodim01`/2k/seed1 rows, bringing the shard to 12/336 CUDA-native fair cells. The
  two-seed paired deltas are still not promotion evidence but are now: `opacity=constant`
  (+1.5633 dB PSNR, +0.3759 AUC, +0.319 s fit), `lr_schedule=cosine` (+1.1433 dB,
  +0.1595 AUC, -0.237 s fit), `loss=charbonnier` (+0.3719 dB, -0.0469 AUC),
  `moment_preserving` (+0.5602 dB, -0.0377 AUC), and `density=variance` (+0.4040 dB,
  -0.0547 AUC).
- Next action: run/resume the CUDA-native shard on the 8-image Kodak screen at the fixed fair
  regime, then run the affine quality-only shard separately and promote knobs only from paired
  metrics with no quality/convergence/performance regression.

## Interfaces touched
`benchmarks/stage_search.py` (protocol runner), `scripts/run_abl005_*_influence.sh`,
`ara/evidence/`, `docs/adr/` for promotions.

## Depends on
ADR-0010, FIT-005/006/007 (axes exist), CORE-006, BENCH-002. Pairs with ABL-006 (shares GPU
budget; this task is the higher-information spend and should run first).
