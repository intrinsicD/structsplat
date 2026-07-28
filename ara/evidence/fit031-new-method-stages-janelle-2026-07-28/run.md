# FIT-031 versus newly added fine-detail stages — evidence audit

## Outcome

Only FIT-032 through FIT-038 were rerun. Their task notes survived, but their original
machine-readable result directories did not. FIT-039 through FIT-041 already had source-bound
production, perceptual, spatial, exact-same-base, and adversarial-audit artifacts, so those results
were hash-checked and reused instead of being recomputed.

The new common-base replication preserves the original stage decisions:

- FIT-032's gauge-lifted dipoles never produce an accepted protected recovery.
- FIT-033 validates high-pass placement plus an exact partial color solve as the useful local
  mechanism.
- FIT-034 through FIT-036 do not justify spectral mixing, affine appearance, or residual-ridge
  anisotropy as the missing mechanism.
- FIT-037 confirms that one-shot site ranking saturates.
- FIT-038 confirms that iterative residual reselection is materially better, but its 5x5
  cross-wave exclusion remains too restrictive.
- FIT-039's exact-site exclusion is the first stage to clear both frozen fine-detail targets;
  FIT-040 is the default-off production integration of that mechanism.

The method judgment remains objective-specific. On FIT-041's exact-same-base full-frame control,
orthogonal pursuit is the fine-detail winner, while FIT-031 is the global foreground-PSNR winner.
Neither result authorizes a default, generality, efficiency, or equal-rate claim.

Open the portable comparison report at [`index.html`](index.html).

## Evidence inventory

| item | durable evidence before this audit | action |
|---|---|---|
| FIT-031 original crop | audit, executed-source patch, run note | reused as historical context only |
| FIT-032–038 | task/result prose but no surviving raw result JSON | rerun on one shared target/base and archived |
| FIT-039 | prototype result, cold audit, perceptual and spatial artifacts | reused after validation |
| FIT-040 | production replay and acceptance artifacts | reused after validation |
| FIT-041 | exact-same-base FIT-031 control and 17-check comparison audit | reused after validation |

The copied result JSONs and base metadata are under `raw/`. `comparison.json` is the independent
18-check aggregation. `artifact.json` is the validated report manifest and bounded snapshot.

## Protocol and comparability

The requested full-frame target is masked Janelle `frame_00008/C0001`, fit at `1200x1038`, seed
`0`. The source PNG used for the replication hashes to
`b11b3a3b063e5630581f6a15ee09527216522b19bee1002a641ddbcc39443db3`, exactly matching the
published FIT-039/040 target file. The mask hashes to
`94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`.

Reproducing the target required Pillow 12.3. A first Pillow 11 Lanczos resize produced different
pixels and is quarantined at `runs/fit031_new_methods_comparison_20260728/base`; no scientific
result uses it.

The fresh RTX-4090 base ended at 10,816 rows and hashes to
`b4dde2c30d82dcd84457609770c1b85d064971f29af0d1bdc38bfc1e191d26be`. All FIT-032–038
replications use that exact field. The published RTX-3050 control ended at 11,000 rows. This is
ordinary CUDA trajectory variation, so:

- comparisons within the FIT-032–038 replication tier are common-base;
- FIT-031 versus pursuit uses only FIT-041's published exact-same-base control;
- endpoint values across the two tiers are descriptive, not exact-base treatment effects.

The primary detail metric is relative reduction in sigma-1.5 high-pass RGB residual MSE on pixels
deeper than `mask margin + 6`. The orthogonal metric is relative reduction in Laplacian residual
MSE on the same region. All selected endpoints must also pass the protected foreground, boundary,
tail-risk, hole, finiteness, and outside-mask gates.

## Stage results

Percentages below are independently recomputed from the archived before/after MSE values. The
“original” columns are the surviving task-note values; the “replication” columns are the new
source-bound RTX-4090 result.

| stage | tested change | rows | original HP / Lap | replication HP / Lap | judgment |
|---|---|---:|---:|---:|---|
| FIT-032 | gauge-lifted residual dipoles | 0 accepted | negative | `0.00% / 0.00%` | reject; protected recovery never accepts |
| FIT-033 | high-pass births + partial color solve | 128 | `6.473% / —` | `7.622% / 8.073%` | keep mechanism; confirmation only |
| FIT-034 | spectral/raw mixed solve | 128 | `6.548% / 7.515%` | `7.705% / 8.076%` | reject; selected raw weight is zero |
| FIT-035 | sparse affine colors | 128 | `8.895% / 9.723%` | `9.686% / 9.392%` | reject; target missed |
| FIT-036 | residual-ridge anisotropy | 128 | `6.565% / 7.506%` | `8.085% / 8.457%` | reject; target missed |
| FIT-037 | static nested high-pass rows | 2,048 | `15.01% / 12.04%` | `15.728% / 12.555%` | reject; stale ranking saturates |
| FIT-038 | iterative pursuit, radius 2 | 2,048 | `20.22% / 16.21%` | `20.902% / 16.240%` | keep mechanism; exclusion too strong |
| FIT-039/040 | iterative pursuit, exact-site exclusion | 768 | `25.926% / 27.316%` | published result reused | fine-detail target reached |

The numerical endpoints move modestly with the base/GPU trajectory, but every qualitative stage
disposition reproduces. The evidence supports the mechanism chain rather than a claim that the
RTX-4090 and RTX-3050 endpoints are interchangeable.

## Static versus iterative pursuit

FIT-037 and FIT-038 coincide at 128 rows because they start with the same ranking. Thereafter
FIT-038 remeasures the residual after each accepted 128-row solve:

| added rows | FIT-037 static HP | FIT-038 iterative HP |
|---:|---:|---:|
| 128 | 7.622% | 7.622% |
| 256 | 10.295% | 11.510% |
| 512 | 12.856% | 15.322% |
| 768 | 13.951% | 17.402% |
| 1,024 | 14.511% | 18.644% |
| 1,536 | 15.231% | 20.057% |
| 2,048 | 15.728% | 20.902% |

Iterative reselection is therefore a real contributor. FIT-039's result shows that the remaining
bottleneck was prior-site exclusion: exact-site deduplication reaches the frozen `25%/20%`
high-pass/Laplacian target at 768 rows.

## FIT-031 comparison

FIT-031's original committed evidence is a `1200x437` crop and is not a direct control for the
full-frame stages. Its within-run tail adds 4,608 rows and improves foreground/boundary PSNR by
`+0.522239/+0.582752 dB`. That result remains valid for its own broad foreground/boundary
objective.

FIT-041 supplies the valid full-frame exact-same-base comparison:

| method | added rows | deep rows | HP reduction | Lap reduction | LPIPS reduction | FG PSNR gain | tail seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIT-031 MAE effective-support tail | 2,777 | 0 | 0.000042% | 0.0000% | 3.2114% | **+0.320788 dB** | 18.002 |
| orthogonal pursuit | **768** | **768** | **25.926%** | **27.316%** | **10.460%** | +0.034326 dB | **3.360** |

The pursuit uses 3.616x fewer rows and wins every declared fine-detail/perceptual objective.
FIT-031 wins the global foreground metric. A hybrid may be useful, but that is a new experiment,
not a conclusion from these data.

## FIT-035 audit correction

The first FIT-035 rerun stopped at its renderer A/A check. The affine zero-gradient renderer
matched the constant renderer within `2.384e-7` maximum absolute error and had zero foreground-MSE
delta, but the strict protected gate labeled identity as `no_material_gain`. The harness
incorrectly treated that label as a parity failure.

The correction permits either a full gate pass or exactly `["no_material_gain"]` for this A/A
identity check. Any protected regression still fails closed. A focused regression test covers the
accepted identity and rejected-regression cases. The final source-bound rerun passes A/A and keeps
FIT-035's scientific disposition negative.

## Independent audit

`comparison.json` passes all 18 checks:

- schemas, target bytes, mask, base-field binding, and common baseline;
- captured source hashes and recomputed detail reductions;
- the FIT-032–038 stage dispositions;
- the corrected FIT-035 renderer A/A;
- FIT-031's original audit validity;
- all 17 checks in the published FIT-040/FIT-041 exact-same-base audit.

Reproduce the aggregation and report artifact with:

```bash
PYTHONPATH=src python scripts/experiments/audit_fit031_new_method_stages.py
PYTHONPATH=src python scripts/experiments/build_fit031_new_method_report.py
node /home/alex/.codex/plugins/cache/openai-curated-remote/data-analytics/\
0.2.8-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs \
  --input ara/evidence/fit031-new-method-stages-janelle-2026-07-28/artifact.json \
  --output ara/evidence/fit031-new-method-stages-janelle-2026-07-28/index.html
```

The Data Analytics manifest validator passes. Portable packaging passes, and structural HTML
verification passes. Browser interaction and viewport QA were unavailable because no local
Chromium headless-shell executable is installed; the portable tooling did not download one.

## Verification

Focused new-method tests:

```text
57 passed, 4 deselected
```

The FIT-035 regression subset:

```text
3 passed
```

The first plain `./scripts/verify.sh` invocation collected an unrelated editable checkout from
`/home/alex/Documents/Deeplearning3/external/structsplat`; it was discarded as an environment-path
collision. With this checkout first on the import path,
`PYTHONPATH=src ./scripts/verify.sh` produced:

```text
ruff: passed
pytest: 1,486 passed, 4 skipped, 3 failed, 514 deselected
```

The three failures are the same unrelated broad-suite failures already documented by FIT-031's
scientist pass: a rank-deficient affine diagnostic expects a finite rather than infinite condition
number, this Torch build omits CUDA PCI properties, and a filesystem descriptor-mutation timing
test does not raise. None touches the changed FIT-035 harness, its regression test, the stage audit,
or the report builder.

Because `verify.sh` exits at pytest failure, its four remaining gates were run directly and all
passed:

```text
docs_sync: OK
check_ara: OK (60 claims)
check_task_policy: OK
check_script_layout: OK
```

## Fine-detail visual companion

The published RTX-3050 FIT-040 field was no longer present under `runs/`, so its exact rendered
pixels could not be reconstructed from the surviving metric JSON alone. To supply the missing
visual evidence without repeating already-supported arms, only the FIT-040 production pursuit
path was replayed on this bundle's surviving RTX-4090 common base. The target and mask are the
same exposed Janelle `frame_00008/C0001` inputs used above, with hashes
`b11b3a3b063e5630581f6a15ee09527216522b19bee1002a641ddbcc39443db3` and
`94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`.

On this trajectory, pursuit crosses the frozen targets after five 128-row waves and 640 added
rows: sigma-1.5 high-pass MSE falls `25.1422%`, Laplacian MSE falls `25.4642%`, foreground PSNR
gains `0.037602 dB`, every accepted wave is protected-safe, and outside-mask color and coverage
remain exactly zero. This is a visual replay, not a replacement for the published 768-row
RTX-3050 endpoint.

The crop center `(y=634, x=420)`, 192-pixel crop size, and common error-map scales are reused from
FIT-033's base-only high-pass selection. They were fixed before this winner replay; no crop or
per-panel scale was selected from the after image. The fixed crop contains 15,329 deep pixels and
shows a `25.6763%` high-pass MSE reduction.

- [`full_frame_comparison.png`](visuals/full_frame_comparison.png): target, base, and winner with
  the fixed crop marked.
- [`detail_crop_comparison.png`](visuals/detail_crop_comparison.png): exact target/base/winner
  crop pixels at 3x nearest-neighbor magnification.
- [`detail_diagnostics.png`](visuals/detail_diagnostics.png): absolute and high-pass errors on
  shared base-derived scales, plus signed correction maps.
- [`site_allocation.png`](visuals/site_allocation.png): all 640 added centers colored by pursuit
  wave.
- [`result.json`](visuals/result.json): source hashes, complete schedule, metrics, crop rule,
  output hashes, and environment.

Reproduce this visual-only winner replay with:

```bash
PYTHONPATH=src python scripts/experiments/render_fit040_fine_detail_winner.py \
  --out runs/fit040_janelle_visual_replay_reproduction \
  --quiet
```

The pursuit schedule regression test passes (`2 passed`), Ruff passes, and a 16-check visual
audit verifies source, target, mask, field, and image hashes; recomputed reductions; protected
acceptance; exact outside-mask zeros; base-render display parity; and the explicit FIT-042 scope
boundary. All four structural gates pass after adding this companion.

FIT-042 explicitly excludes `frame_00008/C0001` and all FIT-023--041 development sources. These
images therefore do not start, satisfy, or otherwise count toward FIT-042 independent
confirmation, and they do not widen C59/C60.

## Disposition

For stage judgment:

1. Close FIT-032 and FIT-034–037 as negative/diagnostic branches.
2. Retain FIT-033's partial solve and FIT-038's iterative reselection as components of the winning
   mechanism, not separate production choices.
3. Retain FIT-039 exact-site exclusion and FIT-040 integration as the sole current fine-detail
   candidate.
4. Keep FIT-031 available for broad foreground/boundary cleanup.
5. Keep both tails default-off until a preregistered multi-image, multi-seed, equal-rate or
   equal-count comparison is complete.

This is a replication and evidence-completeness companion to claims C59/C60; it does not expand
their one-image scope.
