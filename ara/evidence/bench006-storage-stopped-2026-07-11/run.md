# BENCH-006 stopped-run record — 2026-07-11

## Status

The committed 168 KiB all-method convergence runner was launched for the declared 328-cell matrix
(4 COCO images × 41 methods × 2 seeds). The user then explicitly requested that execution stop and
all task changes be committed. The process was terminated during cell 2; 1/328 cells completed and
the ignored journal remains resumable at `results/storage_budget_168k_all_methods/metrics.jsonl`.
No complete benchmark conclusion or canonical lane/root `index.html` was produced.

## Completed cell

| Image | Method | Seed | Iterations | Stop | G | Payload bytes | PSNR | MS-SSIM | LPIPS | AUC | SSPL1 bytes | Fit s |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO_train2014_000000000009 | structsplat_best_default | 0 | 8,201 | PSNR plateau | 5,376 | 172,032 | 49.9428 | 0.999787 | 0.0000606 | 45.4467 | 48,542 | 21.233 |

Run protocol SHA-256: `c2baabdb311e195b7f6c43b457095b2e5101a2b57c9585f96ce050a04562f416`.

This single cell validates execution/accounting only. It is not evidence for an all-method ranking.
