# COMP-007 gauge-free covariance codec: valid negative

**Decision:** kill the frozen log-SPD codec branch; do not open confirmation
**Scope:** complete-stream covariance quantization and coding on even Kodak development IDs
**Artifact:** `results/comp007_gauge_free_covariance_dev_v4`

## Research question

Rotation--scale covariance coordinates contain an axis-swap and angular gauge. COMP-007 tested a
simple exact alternative: encode the three unique entries of the log covariance. The experiment
compared this `log_spd` chart with both the current rotation--scale chart and a causal canonicalized
rotation--scale control. It exhausted 84 integer covariance-bit allocations, two complete-stream
coders, and both legal predictors on 12 images at two Gaussian counts.

This is deliberately not a novelty claim for log-Euclidean SPD coordinates. The useful question
was narrower: does removing the coordinate gauge improve actual StructSplat rate--distortion once
all syntax, predictor search, quantization reconstruction, and rendered distortion are counted?

## Audit disposition

Two earlier artifacts were rejected before scoring:

- v2 used symbol recompression instead of the required decoded-field ordinary re-encode check;
  the real identity test failed all 12,096 streams.
- v3 omitted three transitive initializer modules from its executed-source closure and was stopped
  after 20/24 fields and 9,871/12,096 streams.

V4 corrected those implementation defects without changing images, fields, bit budgets, gates,
or minimum effects. The full decision audit passed. It validated 24 fields, 12,096 candidate
streams, exact byte accounting, predictor selection, complete source/data/config bindings, and
true decode--re-encode identity. It then independently rerendered all 12,096 decoded fields; all
render hashes matched and all numerical quality metrics replayed within the frozen tolerances.
An additional read-only audit checked the exact `12 x 2 x 3 x 2 x 84` row matrix, all stream
hashes and `152,846,906` counted complete-stream bytes, independently re-encoded every stream,
and rerendered a stratified 216-cell sample. Its raw-row reduction exactly reproduced the stored
analysis (`9e500c476fff98fccf39200991ead901b221e56ec02ec667b03573a2d180fd84`).

## Result

Seven of eight conjunctive gates failed:

| Gate | zlib-9 | zstd-9 | Result |
|---|---:|---:|---|
| median whole-container reduction | `-0.4053%` | `+0.3426%` | fail |
| bootstrap upper rate-ratio bound | `1.007615` | `1.001999` | fail |
| image wins | `5/12` | `7/12` | fail |
| attributed covariance-stream reduction | `-1.6077%` | `+1.8145%` | fail |
| worst envelope PSNR shortfall | `5.2945 dB` | `4.6074 dB` | fail |
| encode-time ratio | `1.0432x` | `1.0485x` | pass |
| cold-decode ratio | `1.1028x` | `1.1030x` | pass |
| direction stable across counts | no | yes | fail |
| canonical mechanism guard | no | yes | fail |

The chart therefore does not provide a robust compression benefit. The only clean positive is
that its implementation overhead stayed within the permissive `1.25x` timing budget; this does not
matter after the compression gates fail. The odd Kodak IDs remain sealed.

The gate-5 values are worst-case distances to piecewise Pareto envelopes around near-vertical,
few-byte frontier jumps; they are not a claim that a typical reconstruction became five decibels
worse. The other six failed gates independently kill the branch even if this brittle worst-case
diagnostic is ignored.

## What the experiment says—and does not say

| Axis | Evidence |
|---|---|
| Quality | Not tested unquantized; matched-distortion codec behavior failed. |
| Convergence | Not tested; all charts received the same already-fitted fields. |
| Performance | Codec encode/decode overhead stayed within budget; renderer/training performance was not tested. |
| Compression | Valid negative: no qualifying complete-stream RD gain. |
| Expressiveness | Not tested; the charts have the same covariance degrees of freedom before quantization. |

Canonicalizing rotation--scale is descriptively a little smaller than the current chart, but
COMP-007 did not freeze a standalone promotion rule for it. It is a useful implementation clue,
not a result to promote post hoc.

## Decision and next branch

Close this exact gauge-free chart without retuning and preserve v2/v3 as unavailable audit
artifacts. The next experiment changes the representation mechanism rather than polishing the
failed codec: test whether the existing normalized Gaussian compositor can be raised from a local
constant to a local affine reproducing operator with zero extra per-Gaussian state. Its analytic
conditioning/ringing stage must pass before any natural-image fitting or held-out data are opened.

## Canonical hashes

- config: `fbf1846492a930faf556c3ce9b8c98927c32bb1a20055ac91cc30ed9f52113f9`
- fields: `a8d0c60a5cb5dd8b5bb1027f284386bb6261bb1f722460638d8f88a694f7f40a`
- candidates: `c16be1e17b6a67a87077c3bb2169c4dd730eb593b192a8844ada628c85e83164`
- audit: `ad1ec6c889e818e4b1af4cc63a4f99959453534951f480554471bf14fc621aa5`
- analysis: `115c2e272a406b1d85313496a94c76e6a4f47c59e41b79e13f951f0ff464ea27`
- executed source: `cb6d0d8eb77b6b98328dd66290d8a514a746e00de07c10a2d23cc62959eb9753`
