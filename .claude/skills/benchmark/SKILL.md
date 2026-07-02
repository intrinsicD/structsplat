---
name: benchmark
description: Use when running or extending the StructSplat benchmark / ablation, interpreting results, or wiring outputs as a fitness signal for algorithm discovery. Covers the metric protocol (PSNR / MS-SSIM / LPIPS + iterations-to-target) and the strategy x budget sweep. Trigger on "run the ablation", "benchmark", "fitness", or editing benchmarks/.
---

# Benchmark & fitness

The headline experiment (`ABL-001`) is `{init strategy} x {budget}` scored on fixed images. It
answers turn-3's question empirically and is the **fitness signal** for a co-scientist loop over
init/sampling variants.

## Protocol (keep fixed — this is the contract)
- Metrics: **PSNR**, **MS-SSIM**, **LPIPS** (optional dep), and **iters-to-target** at a fixed
  target PSNR. Report quality at fixed budget *and* convergence speed — they can disagree.
- Same fitter config, same iteration count, same images for every cell in a comparison.
- Seeds logged; multiple seeds per cell when claiming a difference is real (report mean ± std).
- Output: a tidy JSON/CSV row per (image, strategy, budget, seed) + a markdown summary table.

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
