# FIT-020 ranked deduplication perturb--recover assay

Date: 2026-07-15
Decision: `stop`
Provenance: AI-executed benchmark and replay under user-directed continuation

## Bound artifacts

- Task: `tasks/FIT-020-perturb-recover-spectroscopy.md`
- Benchmark: `benchmarks/perturb_recover_spectroscopy.py`
- Focused tests: `tests/test_perturb_recover_spectroscopy.py`
- Detailed report: `docs/research/2026-07-15-perturb-recover-spectroscopy.md`
- Decision: `docs/adr/0015-keep-response-bend-selector-benchmark-only.md`
- Primary: `results/fit020_response_spectroscopy_v1_2026-07-15/`
- Measurement-equivalent post-writer-fix replay:
  `results/fit020_response_spectroscopy_v1_replay_2026-07-15/`

Target-manifest SHA-256:
`4e51518b9c2b1816311f1b2856c19ea77acaea913b534987fa776cf160740ebe`.
Primary/replay source combined SHA-256:
`e1d64469d858c65d74aa6083f2845a1a421946eab646ede5bd0be6857ba9b383` /
`75fbe6ac96af2d3b250e81c6157a0645402ebe2ed70e5fa29dd5ca2cff3b99ab`.

## Frozen protocol and integrity

Six procedural families x six variants were split into variants `{0,2,3,5}` for training and
`{1,4}` for within-assay held-out procedural-variant evaluation. This is not confirmation or
natural-image evidence. Seeds `{0,1,2}` were averaged before model fitting. Four arms
C5--C8 each applied eight sequential moment-preserving births to the same N=32 checkpoint, ended
at N=40, and recovered for 200 fresh-Adam steps. The response model added only the preregistered
bend `((d10-d2)/8)-((d2-d0)/2)` to the early/static controls.

The primary contains 432/432 complete trajectories and 108/108 seed-averaged paired rows (72
training, 36 held out). Histories, counts, finiteness, actions, ranking/source/target hashes,
terminal-inclusive AUC, grouped CV, and held-out isolation all pass. The signal guard passes with
held-out `SD(y)=3.2529236365 dB` and 35/36 cells at `|y| >= 0.10 dB`.

## Frozen result

| Metric | Early | Response |
|---|---:|---:|
| Held-out RMSE (dB) | 2.9616239375 | 2.9640810597 |
| Response/early RMSE ratio | -- | 1.0008296537 |
| Sign accuracy | 25/36 | 25/36 |
| Bias, mean prediction minus observation (dB) | -1.0515786970 | -1.0455236330 |
| Spearman | 0.6036036036 | 0.5951093951 |
| Response family RMSE wins | -- | 2/6 |

The response/early ratio bootstrap 95% interval is `[0.9985823449, 1.0018898483]`; response bias
interval is `[-1.9078539804, -0.2196336150]`. Grouped train-only CV is also worse for response:
`1.6628939230` versus early `1.6615323434`; every model selects ridge alpha 100.

Early and response select C5 for all 12 held-out targets and both have late regret
`1.1115683450 dB`. Observed-step-10 selection has regret `0.7668961419 dB`; always C8 has
`2.9873440001 dB`. The prediction and screening gates fail.

## Descriptive boundary

Held-out step-200 deltas versus C8 are C5 `+1.8757756551`, C6 `+2.2486516635`, and C7
`+1.6424255371 dB`. These were not preregistered fixed-policy claims and are dominated by sinusoid
and chirp targets. Excluding those two families gives C5 `-0.280813`, C6 `+0.429437`, and C7
`+0.408748 dB`. Best step-10 and step-200 arms differ for 9/12 held-out targets. This is bounded
descriptive evidence for the ranked ticket-deduplication path, not pure coverage, natural-image
quality, or a C6 promotion.

All arms have identical N=40 and atom schema and no stream was encoded. There is no compression or
expressiveness evidence. Dense CPU instrumentation is not a performance result. Fresh Adam means
there is no production optimizer-state-continuation claim.

## Writer correction and replay

The primary completed every measurement before `paired_rows.csv` failed because the first training
row lacked held-out-only prediction fields. Finalization reloaded immutable `rows.json`, aggregated
with the primary source snapshot, proved the aggregate unchanged, and repaired only the writer to
use the sorted union schema. `finalization.json` binds the corrected aggregate-with-paired-rows by
SHA-256 `b8848767ae746c19576bbbc546b66f5f809e318a745f0cf164b812c21687b2bf`.

The replay therefore differs in source only by that serializer repair and its regression test. The
normalized comparator excludes three timing fields and `source_combined_sha256`; all compared
non-timing measurement fields across 432 keyed cells, all paired rows, the normalized aggregate,
target manifest, and STOP decision match exactly. It is measurement-equivalent, not literally an
untouched-source replay.

Independent protocol, numerical/reproducibility, and literature audits found no residual blocker
after the scope corrections. Focused FIT-020/gauge tests pass 52/52; the full suite passes 540/540
with the documented system `libstdc++` preload; Ruff, source-snapshot hashes, local documentation
links, and diff hygiene pass.
