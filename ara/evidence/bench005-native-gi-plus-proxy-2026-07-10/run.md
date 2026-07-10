# BENCH-005 native GaussianImage++ matched-axis proxy

- Date: 2026-07-10
- StructSplat commit: `9c13583d68acdb75751f81d458ae9b576171e21d` plus dirty-state fingerprint in `config.json`
- GaussianImage++ commit: `549cfaab2b400248f685c12782a180f3cfc038b0`
- Protocol: COCO4, max-side 160, cap 640, initial 320, 500 requested steps, seeds 0/1, LPIPS enabled
- Status: 8/8 native cells passed adapter-v2 commit/extension/axis/history/reconstruction validation
- Paired result, native gain over StructSplat default: PSNR -5.0678 dB, proxy MS-SSIM -0.05142, LPIPS -0.1886, AUC -7.1638, fit time +0.4284 s
- Interpretation: short-horizon time-quality tradeoff; not native-authentic, multi-budget, codec-RD, or global ranking evidence
- Selection caveat: upstream restores the best training-PSNR checkpoint; all eight cells selected step 500/500 in this slice
- Validation: real adapter-v2 CUDA smoke passed; changed-file Ruff passed; full project suite passed 326 tests with the local libstdc++ preload

Primary artifacts are `summary.md`, `metrics.csv`,
`paired_native_vs_structsplat_summary.csv`, and `config.json`.
