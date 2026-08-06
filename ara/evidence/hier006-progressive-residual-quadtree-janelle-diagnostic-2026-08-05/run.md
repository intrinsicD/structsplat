# HIER-006 progressive residual quadtree — exposed Janelle diagnostic

## Scope

Dirty-source, single-image mechanism diagnostic for the default-off HIER-006 parent-preserving
direct-additive hierarchy. This is not a clean prospective comparison, a semantic/default
selection, or a complete-codec result. The exposed C0001 image and provisional artifact gate were
already used throughout HIER-005.

Authoritative corrected bundle:
`results/hier006_janelle_progressive_residual_quadtree_corrected_2026-08-05/index.html`.
It passes:

```bash
python scripts/check_report_bundle.py \
  results/hier006_janelle_progressive_residual_quadtree_corrected_2026-08-05 \
  --allow-dirty
```

## Frozen method and protocol

- Native source/mask SHA-256:
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b` /
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`.
- Deterministic LANCZOS/nearest evaluation raster: 512x443, 15,929 active pixels.
- Start at all 12 mask-present level-6 cells. A split retains its parent and appends every
  mask-present child with fixed mask-moment geometry and signed mean-residual RGB initialization.
- Rank frontier parents by mask-aware sigma-1.5-smoothed residual energy per appended child row;
  add at most 256 rows/stage and never exceed 8,192 total rows.
- Optimize the 12-row base for 400 Adam steps and only each new child RGB block for 50 steps at
  LR 0.05. The objective is masked MSE plus four times the mean worst 1% pixel MSE.
- Check every five optimizer steps. The unchanged prefix and candidates are ordered by raw
  normalized pixel-max/7x7-patch violation, then foreground SSE. Cold joint rendering is the commit
  authority; older arrays must remain bit-exact.
- Displayed gate: foreground pixel RGB-RMSE maximum <=0.02 and maximum complete black-matted 7x7
  patch RMSE <=0.01. Preserve base, last <=4,096, first passing, and terminal prefixes.
- Context only: persisted HIER-005 hard-3-sigma touched rows at 4,096 and 8,192; they were not rerun
  as equal-work controls.

Executed module/driver SHA-256 in the corrected bundle:

- `src/structsplat/progressive_residual_quadtree.py`:
  `fd693b24055f293f59d8fbb7859865aa02e0728a6d4e29a3cf02fba4ff83ef37`
- `scripts/experiments/hier006_progressive_residual_quadtree.py`:
  `eacac5294c5ec0057bb7e643fb541cb53b9be48797a48b655c5c7c61ff9a3370`

## Corrected outcomes

| prefix N | PSNR dB | SSIM | MS-SSIM | LPIPS | pixel max | 7x7 max | gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 12 | 15.6191 | 0.951175 | 0.938690 | 0.109243 | 0.801520 | 0.462759 | fail |
| 3,986 | 27.8050 | 0.976880 | 0.995514 | 0.021590 | 0.222276 | 0.085998 | fail |
| 8,192 | 32.8816 | 0.986448 | 0.998979 | 0.014489 | 0.107301 | 0.037518 | fail |

No prefix passes. The 3,986-row killing rule rejects the literal fixed-prefix mechanism for the
exposed 4k goal, and the 8,192 cap also fails without weakening the gate.

Contextual HIER-005 deltas:

- Versus 4,096 touched rows, the 3,986 hierarchy is -2.676 dB PSNR; pixel maximum is 1.080x and
  7x7 maximum 1.216x worse. LPIPS is 0.00216 lower, illustrating that a global perceptual metric
  can improve while the declared local-artifact boundary worsens.
- At exactly 8,192 rows/290,496 canonical bytes, the hierarchy is -19.474 dB PSNR; pixel and 7x7
  maxima are 7.227x and 7.059x worse. HIER-005 passes at 0.014847/0.005315; HIER-006 fails at
  0.107301/0.037518.

## Structural diagnosis

The corrected trajectory accepts all 35 proposed stages, preserves every prior array prefix, and
maintains accumulated/cold render parity within `3.6e-7` max absolute error. The failure is not a
rollback or serialization artifact.

At N=8,192:

- retained non-leaf ancestors: 5,106 rows (62.3%);
- level-0 pixel-scale leaves: 3,086 rows (37.7%);
- accepted parent splits: 2,263;
- complete hierarchy build: 7.535 s on the recorded RTX 3050.

The terminal worst displayed pixel is `(173,265)` on the thin foreground boundary. Its path was
refined only through level 1; the level-1 cell remained unsplit at the cap. Summed smoothed error
per child favors spatially broad residual regions over isolated boundary maxima, while immutable
ancestors consume most of the fixed row budget. Visuals retain a quadtree/grid imprint at 3,986
and a weaker but still measurable imprint at 8,192.

## Implementation-defect correction and repeat

The first frozen output remains at
`results/hier006_janelle_progressive_residual_quadtree_2026-08-05`. At attempt 9, an unchanged
worst-local violation was `18.835389288` in the NumPy prefix and `18.835390091` in the float32 Torch
checkpoint. Strict tuple comparison incorrectly rejected a candidate whose SSE fell from 217.20
to as low as 191.59, then blocked all selected parents. This is a numerical-domain defect, not a
parameter outcome.

The corrected comparator evaluates optimizer keys in one Torch domain and defines a 32-float32-
epsilon equivalence band before using the SSE tie-break. No frozen hyperparameter changed. An exact
parameter repeat at
`results/hier006_janelle_progressive_residual_quadtree_precisionfix_repeat_2026-08-05` reproduces
all displayed pixel/patch gate metrics exactly. Terminal PSNR differs by about `1.7e-5` dB between
the two precision-fixed pre-telemetry executions; canonical field hashes differ, consistent with
the declared CUDA atomic-gradient limitation.

## Byte accounting

| N | full/canonical bytes | lossless reference NPZ | structural proxy bytes | native JPEG/full | eval PNG/full | eval PNG/proxy |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 28,736 | 31,828 | 28,498 | 496.53x | 1.018x | 1.027x |
| 3,986 | 155,904 | 159,004 | 76,683 | 91.52x | 0.188x | 0.382x |
| 8,192 | 290,496 | 293,596 | 127,680 | 49.12x | 0.101x | 0.229x |

The 14,268,226-byte native-JPEG ratio compares a native high-resolution file with a resized field
and cannot establish compression. Even the structural proxy is 4.36x the exact 29,263-byte
evaluation PNG at 8,192 rows. That proxy assumes shared mask/tree-derived geometry and only prices
float32 RGB plus one nominal bit/node; it omits a self-contained tree grammar, header, entropy
model, and cold decoder. Full field/reference bytes, not the proxy, describe the implementation.

## Verdict and next boundary

Retain this literal parent-preserving, prefix-frozen Gaussian quadtree as a negative HIER-006
control. The experiment refutes the narrow belief that stacking immutable Gaussian ancestors and
high-error children automatically improves both compression and artifact-safe quality at a fixed
row cap.

A successor remains untested. It should separate the quadtree's useful allocation/index role from
the literal final basis; include artifact-first maxima/components in split priority; reconcile old
and new coefficients locally or jointly under explicit drift constraints; consider nested
zero-moment/lifting detail atoms; and charge ancestor, coefficient-delta, tree, alpha, header, and
decoder bytes under matched work. It must not retune this exposed C0001 outcome into selection
evidence.

## Verification and limits

- Corrected and repeat bundles: `check_report_bundle.py --allow-dirty` both pass.
- Focused hierarchy/report suite: 20 tests pass; field/contraction/observation regression slice:
  84 tests pass.
- Repository gate after code/docs/task synchronization: 1,618 passed, 4 skipped, 514 deselected;
  lint, docs sync, ARA, task policy, script layout, and agent workflow all pass.
- Dirty exposed-image diagnostic, no prospective independent protocol review, no distinct
  numerical/scientific result review, no disjoint images, no equal-work control, and no actual
  codec. No formal claim or default changes.
