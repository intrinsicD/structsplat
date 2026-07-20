# Native SAD frontier screen: audited development result

## Decision

The frozen BENCH-016 v6 action is **abandon SAD reuse**. The result is narrower than that label:
native SAD is exceptionally strong at the 0.5-bpp nominal operating point and decisively faster
than the matched StructSplat control, but it does not clear the preregistered quality gate at the
2.0-bpp point. This closes the proposed SAD reach/temperature/top-8 transplant; it does not show
that SAD is generally worse, that its primitives have no value, or that StructSplat dominates it.

The direct sources are the [SAD paper](https://arxiv.org/abs/2604.21984) and
[official repository](https://github.com/LuckyIYI/SAD), pinned here at commit
`0eeb7e1e72b81e550f90db8ebbf432cdc57383ed`. The experiment ran the unmodified upstream CUDA
system and a frozen equal-terminal-coordinate StructSplat control on eight resized DIV2K training
images. It is development evidence on one RTX 3050, not a paper-table reproduction or SOTA claim.

## Frozen comparison

Each image and requested SAD nominal rate `{0.5,2.0}` had three fresh native repetitions. For each
native repetition, the control used `ceil(10N/8)` constant-color StructSplat Gaussians, matching or
slightly exceeding SAD's ten terminal learned/exported scalars per site with StructSplat's eight
scalars per Gaussian. Both methods ran 4,000 updates. This matches terminal coordinate count, not
initial state, transient capacity, optimizer state, candidate-map work, FLOPs, or information.

Native quality came exclusively from the role-fixed first fresh official replay of each persisted
SAD TXT. The training-process PNG and 53 prepared timing replays were diagnostics. StructSplat
quality came from its persisted and cold-decoded SSPL1 control. Image-level inference averages
the three quality/convergence repetitions and takes medians for time and memory.

## Result

Positive PSNR and AUC favor SAD; negative LPIPS favors SAD. Time and memory ratios are
`SAD/StructSplat`.

| Requested nominal rate | Median PSNR gain | Worst gain | Positive images | Mean-gain bootstrap 95% CI | Median LPIPS delta | Median normalized-iteration AUC gain | Training ratio | Prepared-render ratio | VRAM ratio | RSS ratio | Frozen result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.5 | +5.473608 dB | +0.452240 | 8/8 | [+3.195462,+7.465275] | -0.066571 | +10.715291 dB | 0.212252 | 0.123949 | 0.669492 | 0.073046 | all gates pass |
| 2.0 | +0.428143 dB | -1.587153 | 7/8 | [+0.025393,+3.442337] | +0.000540 | +6.135503 dB | 0.173206 | 0.059747 | 0.635779 | 0.073025 | **fail** |

The 2.0-bpp cell fails three exact requirements:

- median PSNR gain is `+0.4281428655 dB`, below the required `+1.0 dB`;
- worst-image gain is `-1.5871531169 dB`, below the `-0.25 dB` floor; and
- median LPIPS delta is `+0.0005401423`, above the non-worse threshold of zero.

The LPIPS miss is too small to call a substantive perceptual regression. The median PSNR gate is
independently decisive. The 0.5-bpp advantage is nonetheless large and consistent, and SAD also
has a strong internal normalized-update convergence and native GPU performance advantage at both
rates. The honest outcome is a rate-dependent frontier, not a uniform winner.

## Outlier and replay audit

The high-rate rejection is not an artifact of the worst `0513` native repeat. An unauthorized
sensitivity that replaces only that repetition with the mean of its other two changes the cell
effect from `-1.587153` to `-0.143083 dB`, so the worst-image gate would pass. The high-rate median
remains exactly `+0.428143 dB` and median LPIPS remains `+0.000540`; both still fail. This is a
sensitivity analysis only and does not change the official estimator or result.

SAD's recipient replay is stochastic on this GPU because colliding sites can race while seeding
the jump-flood candidate map. Terminal counts agree in all 16 image/rate cells, but site and
primary-pixel hashes agree across training repeats in none. Native within-cell PSNR spans have
median `0.115206 dB` and maximum `4.372990 dB` at `0513/2.0`.

Across 2,544 prepared replays, `981` (`38.56%`) exactly match their role-fixed primary replay.
Sample-minus-primary PSNR lies in `[-0.060337,+0.083530] dB`; this timing-replay variability is far
too small to rescue the failed median gate. The training PNG differs from the primary replay in
all 48 native rows and can be as much as `4.541775 dB` better. Substituting that diagnostic would
flip the high-rate result, but it is explicitly prohibited: recipient-replay quality was the
frozen comparison. The rejection should therefore be read partly as a failure of the current
saved-TXT recipient path, not proof that SAD's transient training representation cannot attain the
quality.

## What each research axis says

- **Quality:** demonstrated SAD advantage at low nominal rate; heterogeneous and below the frozen
  minimum effect at high nominal rate.
- **Convergence:** demonstrated SAD advantage in internal training-state PSNR AUC versus normalized
  iteration. It is not wall-time convergence and cannot be attributed to one SAD primitive.
- **Performance:** demonstrated native SAD advantage in training and prepared render time, plus
  lower measured peak memory, on this single RTX 3050 software/hardware stack.
- **Compression:** not tested for SAD. Its nominal `16 bytes/site`, raw float state, TXT, and
  gzip-TXT diagnostics are not a self-contained entropy-coded stream. StructSplat's SSPL1 rate is
  descriptive control evidence, not a matched SAD compression comparison.
- **Expressiveness:** not isolated. The systems differ in initialization, optimizer, transient
  128k-site state, pruning, candidate map, parameterization, and renderer. Equal terminal scalar
  counts do not prove a representation theorem.

## Artifact and independent audit

The final v6 artifact is `results/bench016_native_sad_frontier_v6_2026-07-16`. It contains 48
native and 96 control rows, 2,928 command records, and a 6,766-file / 1,578,954,906-byte manifest.
Independent quantitative reconstruction found zero discrepancies in all raw rows, effects,
bootstrap intervals, AUCs, timing/resource aggregation, and gates. Independent artifact review
found no unsafe paths, source drift, predecessor-row import, aliasing, or missing manifest links.

Canonical hashes:

```text
binding.json             323b41979ca115cf89f5a16579561690ecd4535768f972cb6506d02320cae42b
targets.json             94ff750b365c7c23e73ae820fbee842fdbf1f406f18cb48777895337802c565e
sad_rows.jsonl           1e3528c539987667152addfc7e3aa2001ac72ac3071b10c207d018d117374646
structsplat_rows.jsonl   ff560b6224d334e5d63b6e94b2567a53d7b855d2e5b6bd942a389bdfd6df5d21
analysis.json            ec4aacf2a7c56e76c9e3f0e1ca97e49a43edeeba04e179fe5b61e852ee2e331c
artifact_manifest.json   639578bef4e2a5aa94d83a59461ecd4aa0fa3ec435f26af596ab3541175ed38e
replay.json              06231b926b495be57138094aa99e76381c43425e0efc27734c8d3f6ba4765211
completion.json          558f52655b1043778698d57b3a5c13f073fc3dafecf3af49cfe02a9c6e4fad5d
```

The artifact replay passes both frozen exact checks. The scientific action is therefore decision-
ready even though the claim remains bounded by the exposed development set, outcome-responsive
v4/v5 integrity repairs, one GPU, and the system-level rather than mechanism-level comparison.

## Next decision

Do not run the preregistered SAD reach/temperature/top-8 bridge. BENCH-016 required both rates to
pass and forbids retuning. The selected disjoint branch is COMP-008: first run a cheap,
reconstruction-invariant SGI-inspired conditional-entropy lower bound. If even that optimistic
bound cannot clear a complete-stream margin, abandon the fixed mean-conditioned coder before
implementing arithmetic coding.
