# FIT-003: Fit-loop speed (device-side target tracking, SSIM hygiene, fused SSIM)

**Status: done.** Completed 2026-07-03. From the 2026-07-03 repo review. Ordered by payoff/effort; items 1–2 are
pure wins with zero semantic change.

## Context
1. **Per-iteration GPU→CPU sync.** `M.psnr` returns `float(...)` and is called every iteration
   whenever any target PSNR is set (the ablation default), serializing the training pipeline.
   (`src/structsplat/fit.py:344-349`, `src/structsplat/metrics.py:25-27`)
2. **SSIM overhead.** The 11×11 Gaussian window is rebuilt on-device on every `ssim()` call —
   twice per iteration (loss + logging paths) across every fit in every sweep — and the SSIM
   forward+backward runs even when `ssim_weight == 0`.
   (`src/structsplat/metrics.py:30-41`, `src/structsplat/fit.py:333`)
3. **Fused SSIM.** With the exact CUDA renderer landed (~6.5× render speedup recorded in the
   trace), the PyTorch SSIM is a dominant remaining per-iteration cost. Taming-3DGS's fused
   CUDA SSIM kernel (rahul-goel/fused-ssim; arXiv 2406.15643) is a drop-in backend worth
   1.2–1.8× fit-loop wall-clock on GPU.

## Goal
Cut per-iteration overhead that is not the renderer, without changing training semantics.

## Acceptance criteria
- [x] Target-PSNR crossings tracked on-device: keep MSE as a tensor, precompute per-target MSE
      thresholds (10^(−t/10)), update a device-side boolean per iteration, sync only at log
      points; `iters_to_targets` results identical on a fixed-seed fit.
- [x] SSIM window memoized keyed by (win, sigma, device, dtype); loss skips the SSIM term
      entirely when `ssim_weight == 0`; fixed-seed loss trajectory identical when
      `ssim_weight > 0`.
- [x] Optional fused-ssim backend in `metrics.py` (try-import, silent fallback to the built-in
      implementation), gated like `renderer='cuda'`; parity test vs built-in SSIM within
      tolerance; wall-clock delta recorded on one GPU fit.
- [x] Before/after seconds-per-iteration table (CPU and GPU, 512 and 20k budgets) in notes.

## Completion notes

Evidence lives in `ara/evidence/fit003-speed-2026-07-03/`:
- `summary.md` records the CPU/GPU before/after seconds-per-iteration table.
- `speed_table.csv` mirrors the table for scripts.
- `raw.json` contains the per-row fit outputs.
- `config.json` records the image, budget, iteration, target, and renderer settings.

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/metrics.py`. No ADR (no math change; fused SSIM is
an opt-in backend).

## Depends on
FIT-001, BENCH-001.
