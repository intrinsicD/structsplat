# HIER-027 cold pure-additive capacity confirmation

## Evidence class

Prospectively frozen, dirty-source, producer-reviewed confirmation diagnostic on eight previously
unopened official DIV2K validation images. Archive identity, selected names and member hashes,
seven arms, counts, schedules, two seeds, metrics, work units, and killing gates were bound before
Python decoded a selected pixel. The report remains provisional without distinct protocol/outcome
review and cannot authorize a semantic, default, codec, rate, full-resolution, or novelty claim.

## Executed protocol

- Sources: official `DIV2K_valid_HR.zip`, archive SHA-256
  `20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc`; prospectively ranked
  `0859.png`, `0833.png`, `0874.png`, `0880.png`, `0802.png`, `0808.png`, `0815.png`, and
  `0889.png` after excluding HIER-026.
- Raster/seeds: deterministic LANCZOS max-side 160, seeds 0/1, required LPIPS.
- Controls: normalized N=640, additive N=640 before/after the unchanged safe RGB projection, and
  non-selectable projected cold additive N=1024.
- Candidate/fallback: ordinary cold full-target additive N=1088 before/after safe projection and
  projected cold additive N=1152, all fitted for exactly 500 L1 + 0.3 SSIM updates.
- Endpoint: one four-array, opacity/mass/denominator/level-free additive `GaussianField`, rendered
  in one pass. Counts and Gaussian-row updates are explicit.

Command:

```bash
PYTHONPATH=src python scripts/experiments/hier027_cold_additive_capacity.py \
  /tmp/structsplat-hier027-div2k-valid-20260811/DIV2K_valid_HR \
  results/hier027_div2kvalid8_s160_capacity_s01_confirmation_2026-08-11 \
  --max-side 160 --seeds 0 1 --device cuda --lpips
```

## Result

| arm | N | mean PSNR | mean MS-SSIM | mean LPIPS | pixel max | 7x7 max |
|---|---:|---:|---:|---:|---:|---:|
| normalized plain | 640 | 28.0910 | 0.980676 | 0.095559 | 0.39061 | 0.13700 |
| additive plain | 640 | 27.2779 | 0.978779 | 0.139636 | 0.37489 | 0.12945 |
| additive projected safe | 640 | 27.3862 | 0.979112 | 0.138375 | 0.36135 | 0.12735 |
| cold additive projected | 1024 | 29.6538 | 0.988351 | 0.078282 | 0.29980 | 0.10629 |
| cold additive plain | 1088 | 29.8393 | 0.988789 | 0.074478 | 0.31293 | 0.10692 |
| cold additive projected | 1088 | 29.9398 | 0.988973 | 0.073287 | 0.30575 | 0.10550 |
| cold additive projected | 1152 | 30.2865 | 0.990050 | 0.067346 | 0.28959 | 0.10470 |

All 112 cells and all integrity gates pass. N=1088/N=1152 beat normalized by
`+1.84883/+2.19555 dB` on mean and `+1.34241/+1.55374 dB` in their worst paired PSNR cells. Both
improve every aggregate structural, perceptual, and local metric; every PSNR and LPIPS cell guard
also passes. The exact frozen local gate still rejects both: N=1088 has pixel-maximum regressions
of `+0.06168` on `0833/s1` and `+0.02780` on `0874/s1`; N=1152 has `+0.03087` on `0859/s0` and
`+0.04570` on `0833/s1`. Each paired 7x7 maximum passes. N=1024 fails the same per-cell local
clause and remains non-selectable. Same-count projected additive is `-0.70478 dB` on mean.

Native sheets show the higher-count fields are generally clean and sharper than N=640 additive,
with no material frame-scale lattice, ringing, hole, wash, color lobe, or blur. That does not
override the predeclared isolated-pixel failures. Neither selectable rung passes and this bank is
not tuned again.

## Results audit

The maintained report checker passes with the dirty-source provenance override. A separate
read-only recomputation over all persisted `analysis.npz` arrays found every reconstruction and
error finite; recomputed raw MSE differs from the ledger by at most `3.40e-11` and recomputed PSNR
by at most `2.91e-8 dB`. The manifest binds 2,037 files. Exact counts/work, shared pre-projection
digests, four-array payloads, projection rollback, coefficient bounds, and internal/cold/repeated
render parity all pass; maximum maintained-render parity is `3.07e-6`, below `2e-5`.

Repository verification passes Ruff, all 83 focused HIER-022--028 tests, and every structural
checker. The complete portable suite has `1,952 passed, 26 skipped, 9 failed`; those nine are the
unchanged inherited affine-condition, external-package subprocess-import, Torch-2.7 CUDA-property,
and descriptor-race failures, with no HIER-027/HIER-028 failure.

## Decision

Retain HIER-027 as a strict negative for ordinary cold capacity under the worst-pixel gate.
Normalization is not shown to be mathematically necessary—the larger additive fields dominate all
aggregate metrics—but merely spending 1.70x or 1.80x rows does not robustly control sparse extrema.
The failure pattern motivates a prospectively frozen residual-allocation test, not another count
sweep or threshold relaxation.

## Receipts

- Report: `results/hier027_div2kvalid8_s160_capacity_s01_confirmation_2026-08-11/index.html`
- Report checker: pass with `--allow-dirty`.
- Manifest SHA-256: `5e808e14599b18cd7369b377ad1818f23106099f86deb419b0ce2e47bf6c21f1`
- Metrics SHA-256: `cc30dd1c3ddf9ebc8d6161b1b0832f854c4ea6646edabf343d0f120d95d5c004`
- Decision SHA-256: `2cbd9eb30acb3a791a2dfbfa178bac6b6845daacf98ccb9b35089df73f27d843`
- Bundle inventory: 2,037 manifest-bound files, 85 MiB.

## Limitations

Eight downscaled validation images; two seeds; one RTX 4090; dirty executed sources; producer-only
review; no distinct prospective reviewer; unequal rows/equations/work; no complete bytes,
full-resolution, downstream, or rate result. The isolated maxima are deliberately not averaged
away.
