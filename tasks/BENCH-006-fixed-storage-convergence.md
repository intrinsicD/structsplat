# BENCH-006 — Fixed-storage all-method convergence lane

## Goal

Add a decision-auditable benchmark lane that reports the serialized source-image size for every
image and compares every registered fair-harness method at one exact 168 KiB analytical Gaussian
payload until a declared plateau criterion or a visible maximum horizon.

## Protocol

- Interpret `168 KiB` as `172,032` bytes, not SI `168 kB`.
- Common frozen representation: `(mean_x, mean_y, log_sx, log_sy, rotation, r, g, b)` in float32,
  exactly 32 bytes/Gaussian and 5,376 Gaussians.
- Keep source, target PNG, reconstruction PNG, decoded array, analytical payload, and actual SSPL1
  stream bytes in separately named fields.
- Freeze the available external-present fair-harness registry; the completed run contained 40
  methods because Instant-GI was absent from the declared external root. Use COCO4, max-side 160,
  seeds `{0,1}`, exact CUDA, LPIPS, and cold-decode SSPL1 metrics.
- Ceiling 10,000 iterations. Complete all structural growth before iteration 6,500, then stop
  after six consecutive 100-step evaluations without a 0.005 dB PSNR gain. Hold a stopped run's
  scored endpoint to the nominal horizon for convergence curves and AUC. Mark max-horizon cells
  right-censored.
- Local analogues only. Do not describe the 32-byte payload as a native codec/checkpoint size.

## Deliverables

- `benchmarks/storage_budget.py`: exact storage accounting.
- `benchmarks/storage_budget_compare.py`: frozen run wrapper and report manifest.
- `benchmarks/results_index.py`: explicit-input portable root dashboard.
- Fair-harness per-image byte/quality tables, exact-capacity status, codec metrics, convergence
  tables, cache artifact hashes, and mixed failure visibility.
- Regression tests, full benchmark artifacts, and curated ARA evidence.

## Status

Implementation, validation, and the external-present execution are complete. The first 41-method
attempt was stopped after 1/328 cells and remains preserved at
`ara/evidence/bench006-storage-stopped-2026-07-11/run.md`. The subsequently frozen available-repo
run excluded absent Instant-GI and completed 320/320 cells (40 methods × four COCO images × two
seeds), with 296 exact-capacity and 24 explicitly overfilled rows:
`ara/evidence/bench001-external-complete-2026-07-13/run.md` and
`results/storage_budget_168k_external_present/`.

Scientific status: completed as a **high-rate local optimizer/policy diagnostic**, not a
compression or SOTA benchmark. At the prepared image sizes the 172,032-byte analytical field is
71.68–81.15 bpp; the actual SSPL1 streams are about 22 bpp, while the prepared lossless PNGs
average about 17.99 bpp. Most paper-name rows are common-harness analogues, not native executions.
Compression decisions move to BENCH-007; this report remains immutable evidence.

## Depends on

BENCH-001/002/003/004, COMP-002, FIT-008/013/014/015/016, ABL-004/005.
