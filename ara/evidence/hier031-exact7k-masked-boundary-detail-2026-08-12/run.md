# HIER-031 exact-7k masked boundary and thin-detail allocation

## Evidence class

Frozen sequential, exposed-source, dirty-worktree, producer-reviewed diagnostic on canonical
Janelle C0001. It tests exact topology and fine-detail allocation at a fixed 7,000-row budget. It
is not held-out confirmation, native-camera or density-parity evidence, an equal-rate comparison,
or support for a maintained default.

## Bound source and protocol

- RGB: native 5328x4608 `C0001.jpg`, SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b`.
- Mask: native 5328x4608 `mask_C0001.png`, SHA-256
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`.
- Raster: deterministic max-side-1200 decode, 1200x1038, with 87,639 active mask pixels.
- Device/seed: RTX 3050 8 GiB, CUDA, seed 0, required LPIPS, 256-row render chunks.
- Every scored endpoint: exactly 7,000 rows; only means, log-scales, rotations, and RGB; all
  centres inside the raw mask; unit support and reconstruction outside `<=1e-7`.
- Ordinary geometry: 0.35-pixel minimum scale, margin 0.75, three-sigma C0 compact support,
  certified anisotropic containment. Micro geometry: 0.08 pixels, independently certified and
  frozen during recovery.
- The task records the initial arms and every mechanistically motivated sequential amendment
  before its outcome. Stage 9 is terminal; no further HIER-031 method development is allowed.

Command:

```bash
PYTHONPATH=src python scripts/experiments/hier031_exact7k_masked_boundary_detail.py \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png \
  results/hier031_janelle_c0001_s1200_exact7k_boundary_detail_s0_diagnostic_2026-08-12 \
  --max-side 1200 --seed 0 --device cuda --render-chunk 256 --lpips
```

## Feasibility result

With the ordinary scale floor, a centre requires SDF at least 1.80 pixels and its isotropic
support radius is 1.05 pixels. The mask has 980 pixels outside isotropic reach from an admissible
ordinary centre. This is an upper bound because a certified tangent ellipse can reach some sites.
The hard lower bound is ten pixels in three connected components with no admissible ordinary
centre at all. No increase in ordinary-row count can cover those components under unchanged
geometry.

At 0.08-pixel scale, the centre certificate radius is 0.99 pixels, so every active mask pixel is a
legal micro site. This justifies a representational micro cohort; it does not justify changing the
global scale floor.

## Scored endpoints

| arm | holes | `<0.05` | PSNR | boundary | interior | high-pass MSE | LPIPS | micro rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HIER-030 cold control | 869 | 1,649 | 21.5745 | 12.4807 | 35.3131 | 0.000113735 | 0.13494 | 0 |
| direct micro exchange | 0 | 826 | 23.2033 | 14.7928 | 30.1651 | 0.000346051 | 0.17705 | 874 |
| detail exchange | 869 | 1,649 | 21.6154 | 12.5216 | 35.3538 | 0.000115184 | 0.13324 | 0 |
| combined exchange | 0 | 821 | 23.1919 | 14.7919 | 30.1051 | 0.000369177 | 0.18129 | 874 |
| corrected merge-funded micro | 0 | 783 | 23.5599 | 14.7693 | 32.9184 | 0.000201788 | 0.15807 | 873 |
| ordinary geometry recovery | 221 | 733 | 24.3996 | 15.4117 | 36.0373 | 0.000113329 | 0.11971 | 873 |
| recovery + terminal closure | 0 | 513 | 25.0938 | 16.3255 | 34.2602 | 0.000150771 | 0.13043 | 1,095 |
| coverage-hinge recovery | 185 | 500 | 24.4194 | 15.6244 | 33.8159 | 0.000136133 | 0.13005 | 873 |
| deep-only recovery | 37 | 780 | 23.8282 | 14.7733 | 36.6672 | 0.000094334 | 0.11787 | 873 |
| **deep-only + terminal closure** | **0** | **743** | **23.8589** | **14.8367** | **36.0677** | **0.000106562** | **0.12085** | **910** |
| current fixed pipeline | 933 | 1,221 | 25.2175 | 16.0939 | 39.8058 | 0.000088250 | 0.07828 | 0 |
| pipeline + boundary recycle | 955 | 1,278 | 25.1792 | 16.0572 | 39.7181 | 0.000088776 | 0.07935 | 0 |

The table omits the preserved erroneous merge-recertification endpoint from interpretation; its
row and error history remain visible in the report. Fourteen attempts produce thirteen scored
rows and one explicit error (`only 761 SDF>2 ordinary pairs for 856 sites`). The corrected replay
exempts already certified micro rows and is separately labeled.

## Selected endpoint and guard

The frozen selection requires exact count/containment, zero raw holes, interior PSNR no more than
0.05 dB below the HIER-030 control, then lowest deep high-pass MSE. Only
`deep_only_terminal_closure_n7000` satisfies the complete guard.

Relative to HIER-030 cold additive it changes:

- overall PSNR `+2.28440 dB`;
- four-pixel boundary PSNR `+2.35597 dB`;
- greater-than-four-pixel interior PSNR `+0.75458 dB`;
- deep high-pass MSE `-6.31%`, Sobel MSE `-11.40%`, LPIPS `-10.45%`;
- Laplacian MSE `+4.56%` (a real counter-metric, so detail is not uniformly better);
- raw holes `869 -> 0`, weak coverage `1,649 -> 743`, thin-ridge raw-hole fraction
  `65.96% -> 0%`;
- maximum outside unit support and reconstruction remain exactly `0.0`.

The selected field has 910 micro and 6,090 ordinary rows. Its maximum/q99 absolute additive
coefficient is 15.998/6.266 versus 10.051/4.299 for the control, a conditioning caveat.

## Pipeline killing controls and visual audit

The untouched current pipelines are substantially sharper and have the best PSNR, high-pass MSE,
and LPIPS. They nevertheless leave 933 and 955 raw holes, including broken hair strands and a
zero-support silhouette fringe. Capacity-time boundary recycling does not repair the failure.

Native report-size inspection finds that the selected endpoint removes the red raw-hole pixels and
keeps the long hair strands connected. It remains visibly soft and is not equivalent to the source
or to the sharper pipeline. Hole maps still show 743 orange weak-coverage pixels; zero raw holes
does not mean uniform or high coverage.

## Allocation/scaling interpretation

Raw error should not be spatially equalized. The optimal fixed-budget condition is approximately
equal marginal reduction in the selected loss per additional row, subject to representability and
topology constraints. HIER-031's deterministic next-row proxy CV changes from 0.7922 in the control
to 0.8216 in the selected endpoint, so marginal equalization is not established. The defensible
result is narrower: establishing a fixed topology reserve prevents count-independent holes and
leaves an ordinary cohort that can benefit from later scaling.

## Presentation-only finalization

The first HTML serializer inherited HIER-029's full-source display and therefore showed background
difference in its primary error panel even though every HIER-031 metric uses the black-matted
objective. No field, metric, decision, feasibility record, or attempt was recomputed. The explicit
finalizer changed only the HTML and six display crops per arm to use the already persisted
objective images. `presentation_finalization.json` binds the old/new index hashes and identical
before/after hashes for every field and measurement table.

## Integrity and validation

The finalized 378-file, 153,700,053-byte report passes:

```bash
python scripts/check_report_bundle.py \
  results/hier031_janelle_c0001_s1200_exact7k_boundary_detail_s0_diagnostic_2026-08-12 \
  --allow-dirty --allow-error-cells
```

Focused HIER-031 and fixed-certified-row tests pass. The complete repository gate passes 1,992
tests with 26 skips and 514 deselections, Ruff, docs sync, ARA, task-policy, script-layout, and
agent-workflow checks. The fit hook rejects dynamic topology, active-prefix storage, missing freeze
masks, and any overlap between exempt and trainable rows.

## Decision

Accept the exposed mechanism result, not a method/default promotion. A count-independent
representability defect exists at the ordinary scale floor. A certified micro reserve plus
deep-only recovery and unchanged terminal closure eliminates raw holes and beats the frozen
control's interior/detail guard at the same 7,000 rows. The endpoint remains softer than the
ordinary pipeline and has mixed detail/conditioning evidence.

The next scientific test should use disjoint masks/images and preserve topology as a hard guard.
The separately deferred approximately-57.6k study should scale ordinary capacity while retaining a
certified topology reserve; it must not assume that adding rows automatically reaches starved
structures.

## Receipts

- Report:
  `results/hier031_janelle_c0001_s1200_exact7k_boundary_detail_s0_diagnostic_2026-08-12/index.html`
- Manifest SHA-256: `34afcdcf29b56adcb457e5838e2f2cc40efff0398725dbedee5b8b1ac6ea0d98`
- Metrics SHA-256: `93f5e88f05d093822b6a41d242eaffc61806e2a96c5ea377742d81f6ef95031f`
- Decision SHA-256: `52016532a23290b12c45b2b9a75c2fc7e3fb0d3001cd19924f30a1a52eb8e2a8`
- Index SHA-256: `777fc14942b3f8bb35d73bbe18c712071307d051818216d3e0d5630e00b6ef76`
- Config SHA-256: `9587b99266f24351f5f8a32f51bb4dba0a5e5abffd2f4152b6ed23ff9e503e81`
- Presentation-finalization SHA-256:
  `c8de52cb9f967cc519f168e478c1b4be225bc8dd63445d7ed4094fa9f60e66a6`

## Limitations

One exposed 1200x1038 image, one seed, one RTX 3050, dirty executed sources, sequential
development on the same raster, producer-only review, no distinct prospective protocol review,
no held-out confirmation, no multiple masks/topologies, no native 5328x4608 or approximately
57.6k run, no equal bytes/rate, no codec/downstream evidence, no proof that silhouette topology
labels every internal hair, and no claim that the deterministic next-row proxy is a true marginal
oracle.
