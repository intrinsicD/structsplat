# ABL-005: Fitter-knob influence pass at the fair regime

**Status: blocked/partial.** The largest unclaimed deltas in the 2026-07 evidence are fitter knobs,
not init strategies — but none has been isolated at a decision-grade regime.

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
- Next action: either implement native CUDA affine-color forward/backward, or split ABL-005 into
  six CUDA-native knobs plus a separate affine quality-only run that explicitly excludes speed
  claims until native CUDA affine exists.

## Interfaces touched
`benchmarks/stage_search.py` (no new code expected — protocol only), `ara/evidence/`,
`docs/adr/` for promotions.

## Depends on
ADR-0010, FIT-005/006/007 (axes exist), CORE-006, BENCH-002. Pairs with ABL-006 (shares GPU
budget; this task is the higher-information spend and should run first).
