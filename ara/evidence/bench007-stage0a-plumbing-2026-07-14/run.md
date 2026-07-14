# BENCH-007 Stage-0a plumbing validation

Date: 2026-07-14

## Scope and claim limit

This run validates the actual-rate benchmark substrate on all four pinned COCO fixtures. Its
scientific scope is frozen as `plumbing_only`; no delta, interval, ranking, or apparent gate
condition from these four development fixtures is research evidence. Stage 0a cannot promote a
method or justify a paper claim.

## Frozen run

- Repository commit: `3b274eb55ddcbeb0857c9c06de0d4619c365bee7`
- Branch: `bench/007-actual-rate-phase-diagram`
- Tracked tree at freeze: clean
- Manifest identity hash: `99d6de0a605558bf15ade43a847315be373082fb606b62edfa0d6c3eb1fae841`
- Frozen manifest file SHA-256: `c0f0e956cdb9056522a67af768f8e391c834604266d1a59091e1d13fa893e41e`
- Images: pinned IDs `0009`, `0025`, `0030`, `0034`, with source-file and decoded-RGB hashes
  embedded in the manifest
- Arms: tensor-metric WSE, quadtree WSE, local SLIC/Sobel control, local gradient control,
  uniform Euclidean WSE, and seeded random
- Candidate search: two target rates (`0.5`, `1.0` bpp), six resolution-normalized counts per
  image, and four frozen codec bit mixes
- Fit horizon: two iterations, seed 0, normalized renderer; this deliberately tests plumbing, not
  convergence
- Runtime: Python 3.11.15, PyTorch 2.7.0+cu126, CUDA 12.6, NVIDIA GeForce RTX 4090

The frozen run was executed/resumed with:

```bash
PYTHONPATH=src python benchmarks/actual_rate_phase_diagram.py run \
  --manifest results/bench007_stage0a_20260714/manifest.json \
  --data-root . --outdir results/bench007_stage0a_20260714 --device cuda

PYTHONPATH=src python benchmarks/actual_rate_phase_diagram.py conventional \
  --manifest results/bench007_stage0a_20260714/manifest.json \
  --data-root . --outdir results/bench007_stage0a_20260714

PYTHONPATH=src python benchmarks/actual_rate_phase_diagram.py analyze \
  --manifest results/bench007_stage0a_20260714/manifest.json \
  --data-root . --outdir results/bench007_stage0a_20260714 --device cuda
```

## Completion and checks

| Item | Expected | Successful | Failed/missing |
|---|---:|---:|---:|
| Independent fitted fields | 144 | 144 | 0 |
| Complete cold-encoded SSPL1 candidates | 576 | 576 | 0 |
| Exact integer-cap target selections | 48 | 48 | 0 |
| Conventional context streams | 92 | 92 | 0 |

The conventional rows comprise four lossless PNG streams, 44 JPEG-444 streams, and 44 AVIF-444
streams. They are visually separated as context and never enter the method gate.

Every generated candidate passed complete-stream parsing, cold decode, and parity against its
in-memory decoded field under the frozen `1e-6` tolerance. F5--F9 were regenerated from the durable
journals, inspected at full resolution, and corrected for the only layout defects found: lossless
PNG is represented off-axis instead of stretching the method RD scale, labels and footers no
longer overlap, and failure/median/success case labels are visible in F9.

Validation after the final figure changes:

```text
ruff check benchmarks/actual_rate_phase_diagram.py src/structsplat tests
All checks passed!

PYTHONPATH=src python -m pytest -q
466 passed, 1 warning in 11.36s
```

## Artifact integrity

The large result bundle remains in `results/bench007_stage0a_20260714/` and is intentionally not
committed. These hashes bind the audited outputs:

| Artifact | SHA-256 |
|---|---|
| `analysis/summary.json` | `425663a0b3d7d21cede6b6ff917c04cb5be368f96fd0e9115951bc4ceb50f8c4` |
| `analysis/selected.csv` | `e6f64d8fe8be28014cbd1eac7ae79870258168d17f3260c101340cd049028dd5` |
| `analysis/conventional.csv` | `1b301437976364f6b3c1d8cf4021eed4740afc0a578eb5363761a865a432304a` |
| `index.html` | `386453e4da1990c68bcf24364550ca445b94d5a0e9f613ed066382c729b5a398` |
| `figures/f5_causal_allocation.png` | `fc510a0dc9830da2920fcb7da6a63656a88f1196532a2d52b648fb2a541930f2` |
| `figures/f6_actual_rate_phase_diagram.png` | `fb432898eabd6e5dc1ac93b2e4a412439091a185ea40c23aac5e0bc735fe5971` |
| `figures/f7_mechanism.png` | `05011d9f372f72f06af59d8de8ffa40bdb34faff6714b6c3416542e671f8c35b` |
| `figures/f8_resources.png` | `658286d476bdb96c12e4ba86959c5330b24ddd9500f09c98541f4502355064c7` |
| `figures/f9_qualitative_quantiles.png` | `e276c0a069f705a1a01c2c6b27b8a84826f1c3759cd9e5c7ccb8beeb7a1415e4` |

## Decision

Stage-0a plumbing is complete. The next admissible operation is Stage-0b rate calibration on the
preregistered DIV2K training IDs `0002`, `0268`, `0534`, and `0800`. That calibration may determine
the frozen bytes-per-Gaussian count ladder, but it may not compare method quality. Stage 1 must be
frozen from that calibration before any Stage-1 metric inspection, and no Stage 2 run is allowed
unless the preregistered Stage-1 gate passes.
