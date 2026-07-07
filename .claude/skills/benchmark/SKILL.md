---
name: benchmark
description: Use when running or extending the StructSplat benchmark / ablation, interpreting results, or wiring outputs as a fitness signal for algorithm discovery. Covers the metric protocol (PSNR / MS-SSIM / LPIPS + iterations-to-target) and the strategy x budget sweep. Trigger on "run the ablation", "benchmark", "fitness", or editing benchmarks/.
---

# Benchmark & fitness

The headline experiment (`ABL-001`) is `{init strategy} x {budget}` scored on fixed images. It
answers turn-3's question empirically and is the **fitness signal** for a co-scientist loop over
init/sampling variants.

## Protocol (keep fixed — this is the contract)
- Metrics: **PSNR**, **MS-SSIM**, **LPIPS** (optional dep), PSNR AUC, and
  **iters-to-target**. Headline convergence tables use target PSNRs **28 / 30 / 32**; keep the
  legacy 35 dB target in raw rows when requested, but do not use it as the only headline target.
  Report quality at fixed budget *and* convergence speed — they can disagree.
- Same fitter config, same iteration count, same images for every cell in a comparison.
- Seeds logged; multiple seeds per cell when claiming a difference is real (report mean ± std).
- Output: a tidy JSON/CSV row per (image, strategy, budget, seed) + a markdown summary table.
- Plateau early exit is opt-in only (`--early-exit`). Rows must record `iterations_run`,
  `stopped_early`, `stopped_at`, and `auc_psnr_horizon`. When early exit is enabled, AUC holds the
  last logged PSNR to the nominal horizon (`auc_psnr_horizon=nominal_hold_last`) so cells remain
  comparable. Do not use early-exit rows to claim exact final-PSNR ordering among near ties; use
  them for convergence/AUC screening, then confirm promoted decisions at the full horizon.
- Cheap screen regime: use max-side **512** and **750** fit iterations for the 11-arm 8-image Kodak
  screen before spending the full fair regime. BENCH-004 evidence rejects 384/500 at 5k
  (`rho=0.8636`, sign agreement 89.8%) and accepts 512/750 (`rho=0.9364/0.9182`, sign agreement
  91.8%/93.9% at 2k/5k). Promotion/default changes still require the full fair regime.
- `color_basis=affine` under `renderer=cuda` currently falls back to the exact PyTorch reference on
  CUDA because the custom CUDA extension lacks an affine-color backward kernel. Quality/AUC rows are
  valid, but fit-time deltas for that arm are implementation-confounded until native CUDA affine
  exists.

## Running
`structsplat ablation <images-or-dir> --budgets 2000 5000 10000 20000 --iters 1500 --target-psnr 35`
or `python -m benchmarks.ablation`. Use a small image set (Kodak-style) to start.

## As a fitness function
`run_ablation` returns per-cell metrics; expose the aggregate (e.g. mean PSNR at a target budget,
or area-under-PSNR-vs-iters) as the scalar a search/optimizer maximizes. Keep the harness
task-agnostic: the scorer reads only the metrics dict, never the method internals (mirror the
IntrinsicEngine core/harness split).

## Reading results
Expected shape (hypothesis, not fact): `aniso_flanking` >= `aniso_onedge` > `iso_blue_noise` >
`grid` > `random` at **low** budgets, gap shrinking as budget grows. If flanking never wins, it has
no niche — record that honestly and prefer the simpler strategy.

## Stage influence (ABL-002 / ADR-0010)
`structsplat stage-search --mode influence` measures each stage's isolated contribution:
one-factor-at-a-time around the baseline (= first value of every stage axis; defaults are the
ADR-0009 production config). `influence.md` reports **paired** deltas per stage option —
ΔPSNR / ΔMS-SSIM / ΔAUC (quality + convergence) and Δiters-to-target / Δinit / Δfit seconds
(speed) — so max-quality, max-convergence and max-speed candidates come from one run. Factorial
mode (`--mode factorial`, default) stays the best-combination search; both dedupe configs whose
differing stage is provably inert. Pass `--target-psnr` or iters-to-target stays empty.

## Experimental-validity rules (BENCH-002 — a sweep result is trustworthy by construction)
- **Equal budgets.** Refine (adding) arms are capped at the cell budget and start below it so
  their planned additions land *at* budget — never compare a refine arm that carries +split_count
  more capacity than the baseline. Every result row carries `n_gaussians`; no arm exceeds its
  `budget`.
- **Reproducible from its own artifacts.** Every harness writes `config.json` (resolved args +
  device + torch/numpy/structsplat versions) into its outdir. `rate_distortion.py` also writes
  `rate_distortion_config.json`. The baseline of `stage-search --mode influence` is the *shipped*
  ADR-0009 default of each stage (e.g. `scale_cap=none`), not a diverged value.
- **Resumable.** `stage_search.py` appends each cell to `stage_search.jsonl` as it completes and
  isolates per-cell failures (a broken arm becomes a `status="error"` row; the sweep finishes and
  the summary skips it). Summary writers never raise `StatisticsError` on an empty method.
- **Honest aggregation.** `fitness()` and the psnr-vs-budget plot use the *best config per
  strategy*, not the pooled mean over hyperparameter variants (which a wide sweep would dilute).
- **One metric convention per row**, stated: fit-time metrics are on the raw render; codec RD
  metrics clamp to `[0,1]` (display-referred). Do not mix clamped and unclamped columns in one row.
- **GPU nondeterminism caveat.** The CUDA renderer uses atomic accumulation (`atomicAdd` /
  `index_add`), so GPU renders are **not** bit-reproducible run to run. Reproducibility from seed
  is exact only on CPU; on GPU the renderer/device/versions in `config.json` bound the variation.
  Record them; do not claim bit-exact GPU repro.
- **No personal paths.** Datasets and the optional Instant-GI module come from CLI args /
  `STRUCTSPLAT_INSTANT_GI`; there are no `/home/...` defaults (`grep -r /home benchmarks/` is empty).
