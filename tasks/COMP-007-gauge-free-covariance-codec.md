# COMP-007: Gauge-free log-Euclidean covariance codec assay

## Status

Completed negative (2026-07-16). Protocol-v3 development v4 is available and decision-ready, but
the gauge-free `log_spd` chart fails seven of eight frozen gates. The odd-ID confirmation split
remains unopened. Benchmark-only: no SSPL1 syntax, production codec, renderer, fitter, default,
configuration, or CLI change is authorized by this task.

The first complete development artifact,
`results/comp007_gauge_free_covariance_dev_v2`, is **pre-scoring unavailable**. Its audit substituted
symbol decompression/recompression for the frozen `cold decode -> ordinary encode` identity check;
the real check failed all `12,096/12,096` streams. No chart outcomes were reduced or used. V2 stays
immutable and cannot authorize confirmation. Protocol v2 below is a source-bound implementation
correction made solely from that correctness failure: float32 range endpoints are used by both
encoder and decoder, row order is derived from reconstructed 12-bit means, and geometry is mapped
to a deterministic decoder fixed-point/cycle representative. It does not change images, fitted
fields, bit budgets, candidate allocations, gates, or minimum effects, and it was frozen before
any chart comparison. This is a new binding/outdir, not an in-place repair or a scientific rescue.

The subsequent `results/comp007_gauge_free_covariance_dev_v3` launch proved the codec correction
but is also **pre-scoring unavailable**: its source archive omitted three transitive modules
(`structure_tensor.py`, `density.py`, and `sampling.py`) that materially determine the frozen
initializer/fitter. It was interrupted at 20/24 fields and 9,871/12,096 streams as soon as the
omission was found; no outcome reduction was run. Protocol v3 archives every Python/C++/CUDA
source under `src/structsplat` plus the benchmark package dependencies, and a test enforces that
closure. V4 again uses a new binding/outdir; v2 and v3 remain immutable negative audit artifacts.

## Question and narrow claim

SSPL1 stores a 2-D Gaussian covariance as two log standard deviations and one angle. The chart is
not identifiable:

```text
(u, v, theta) == (v, u, theta + pi/2),
theta == theta + pi,
theta is undefined when u == v.
```

COMP-007 asks whether replacing those redundant coordinates with the three unique entries of the
log covariance improves **actual complete-stream rate--distortion**:

```text
ell = u + v
a   = (u - v) cos(2 theta)
b   = (u - v) sin(2 theta)

log Sigma = [[ell + a, b],
             [b, ell - a]].
```

The map is exact before quantization, has the same three degrees of freedom, is invariant to axis
swap and angle wrap, and collapses all orientations to one point at isotropy. The claim under test
is recipient-specific: this known log-Euclidean SPD chart can reduce rate at matched rendered
distortion in StructSplat. The chart itself is not claimed as novel.

This assay can establish compression/quantization evidence only. It cannot establish better
unquantized quality, optimizer convergence, renderer performance, or expressiveness. A pass would
authorize a separate direct-log-covariance training assay for those questions.

## Why this branch, and why not the apparent alternatives

The post-BENCH-012 adversarial review ranked this ahead of two alternatives:

1. a decoded-neighbor attribute predictor was only `0.72%` smaller than the current transformed
   zstd stream on the representative `N=20,000` field and remained `0.53%` larger than raw zstd;
2. moment-exact maximum-entropy normalized coordinates are more expressive, but compact-support
   convex-hull failures and a per-pixel nonlinear solve make them the higher-risk follow-up.

Discovery-only probes, consumed before this freeze, found about `10.5%` lower zlib geometry payload
on the 36 lattice-biased COMP-006 parents and `2.5--8.2%` at 4--8 bits on one natural `N=20,000`
field. Those fields are ineligible for the decision below. They justify the experiment, not its
claim.

## Literature boundary

- Arsigny et al., [*Geometric Means in a Novel Vector Space Structure on Symmetric
  Positive-Definite Matrices*](https://epubs.siam.org/doi/10.1137/050637996), establish that the
  matrix logarithm maps SPD matrices into a Euclidean vector space.
- [GaussianImage](https://arxiv.org/abs/2403.08551) already reports that Cholesky covariance
  coordinates are more robust than rotation--scale under identical quantization, and
  [GaussianImage++](https://arxiv.org/abs/2512.19108) directly quantizes the three entries of
  `Sigma` with an invalid-SPD policy. Generic decomposition-free or gauge-free covariance-coding
  novelty is therefore unavailable; Cholesky/direct-`Sigma` are required future threats.
- [ContextGS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5c20ca4b0b20b0bd2f1d839dc605e70f-Abstract-Conference.html)
  and [EntropyGS](https://arxiv.org/abs/2508.10227) show that context and fitted attribute
  distributions are strong Gaussian-stream entropy baselines; a generic entropy-coding novelty
  claim is therefore unavailable.
- [Structure-Guided Allocation](https://arxiv.org/abs/2512.24018) reports large gains from
  adaptive covariance bit allocation, so equal scalar bits are not an adequate control.
- The July 2026 [CGVQ](https://arxiv.org/abs/2607.05667) work groups 2-D Gaussian parameters before
  codebook quantization and reports about 20% lower bpp, making vector quantization a direct
  state-of-the-art threat rather than novelty available to this task.

The potentially new evidence is much narrower: a gauge-removal ablation against canonical
scale--angle, with exhaustive integer bit allocation and self-contained cold streams in a learned
2-D Gaussian image codec.

## Frozen arms

Every field uses one frozen Morton row order and the same complete binary container, fixed-size
header, range slots, stream framing, dense bit packing, mean/color symbols, and decoder. Only the
three covariance coordinates differ.

1. `current_rs`: unmodified `(u, v, theta)`.
2. `canonical_rs`: store major log scale first, minor second, add `pi/2` on axis swap, wrap angle
   to `[0,pi)`, and set exact-isotropic angles to zero.
3. `log_spd`: store `(ell, a, b)` above and decode to the canonical RS representative.

`canonical_rs` is the causal challenger. A win over `current_rs` alone does not support the
log-SPD mechanism.

For all arms:

- means use 12 bits per coordinate and Morton modular deltas;
- colors use 8 bits per channel and Morton modular deltas;
- each covariance coordinate uses an integer allocation in `3..10` bits;
- allocations exhaust every ordered triple summing to total covariance budgets `{12,18,24}`;
- covariance coordinates share one stream in every arm;
- `absolute` and causal `modular_delta` are the same legal predictor menu in every arm; stable
  minimum complete bytes selects the predictor and its tag is counted; encoder time charges all
  five timed encodes of both predictor candidates, while selected-config time remains diagnostic;
- zlib level 9 and zstd level 9 are separate, fully decoded coding strata; and
- all header bytes, float32 ranges, tags, framing, and compressed payload bytes count.

The protocol-v3 container additionally fixes decoder synchronization before scientific scoring:

- min/max endpoints are rounded to their transmitted float32 values before quantization;
- the shared Morton order is computed from reconstructed 12-bit means, so the decoder-visible row
  order is idempotent;
- geometry quantization iterates only the actual float32 decode/re-encode map and emits the
  lexicographically canonical member of its fixed point or finite cycle; this is deterministic
  canonicalization, not an outcome-tuned range sweep. Constant-memory Brent detection uses a
  fail-closed `2^20` projection budget, and the test suite includes an adversarial 852-state
  transient that defeated the earlier 128-step cap; and
- the covariance-only byte diagnostic charges its three framed coordinate planes plus six float32
  range endpoints and the chart, predictor, and three bit-depth tags (`29` syntax bytes). The
  primary rate metric remains the complete container.

Selection and inference use rendered rate--distortion, never covariance error alone. Per
arm/coder/cell, retain the nondominated complete-stream byte/PSNR points. On their common PSNR
interval, interpolate log bytes and average the log-rate ratio over 25 fixed points. No
extrapolation is legal. Empty or less-than-`0.10 dB` overlap is an unavailable cell, not a win.

## Frozen data and split

The mechanism-discovery COMP-006 and `N=20,000` COCO fields are descriptive only.

Primary development fields are built after this freeze from the even Kodak IDs
`{02,04,06,08,10,12,14,16,18,20,22,24}` at max side 160, with counts `{640,2560}`, seed 0,
shipped `quadtree_wse` initialization, 500 ordinary fit steps, normalized owned-exact-CUDA
rendering, constant RGB, and no opacity. The same fitted float field feeds all arms. Source PNG,
decoded target, initial field, fitted field, configuration, and source hashes are persisted.

Odd Kodak IDs `{01,03,...,23}` form a chart-untouched confirmation split under the identical
protocol. They may be fit or scored only if development passes every gate. Kodak is broadly
repository-exposed, so even a confirmed result is internal method evidence, not a paper-level
dataset-generalization claim.

Images, not counts, coders, allocations, or Gaussian rows, are the independent units. Count-level
effects are paired within image. Report paired image bootstrap intervals (20,000 fixed resamples)
and coder-specific results; do not treat the 24 development fields as 24 independent images.

## Correctness and availability gates

Before any scientific decision:

- analytic axis-swap, angle-wrap, and isotropy controls pass;
- float64 covariance round-trip max relative error is at most `1e-12`;
- unquantized current/canonical/log-SPD rendered max-absolute disagreement is at most `2e-5`;
- bit-pack, predictor, compressor, complete-container, and cold-decoder round trips are exact;
- decoded stream re-encode is byte-identical;
- complete bytes equal fixed header plus every framed payload exactly;
- source/data/config bindings and the executed-source archive validate; and
- each primary cell has a legal common PSNR interval at least `0.10 dB` wide.

The decision audit must independently recompute the config/environment binding, archive member
hashes, source and target hashes, fitted-field hashes, row grid and semantics, predictor selection,
ordinary parent-field encoding, decoded-field re-encoding, component accounting, covariance
diagnostics, and every rendered quality metric. CUDA render hashes are diagnostic because atomic
accumulation is not a bitwise oracle; numerical metric tolerances are frozen in the archived
runner. Analysis must invoke the full rerender audit. A confirmation token is valid only when bound
to the development config, artifact audit, fields, candidates, executed source, current source,
task, environment, and passing decision.

The initial field itself (not only a hash) is persisted for every image/count cell. The decision
audit reloads it and independently rebuilds the frozen initializer from the bound target and seed.
If development passes, the confirmation science binding must embed the verified development
analysis, audit, fields, candidates, source archive, and authorization hashes; confirmation audit
recomputes the development reduction from raw candidates rather than trusting editable decision
flags.

Failure before scientific scoring is recorded as unavailable and cannot be repaired by changing
ranges, bit budgets, row order, field count, fit steps, coder level, or target split.

## Development pass/kill gates

All gates are conjunctive and are evaluated against `canonical_rs`:

1. median paired whole-container area-rate reduction is at least `1.0%` for **each** coder;
2. for each coder, the paired image-bootstrap 95% upper confidence bound on the geometric mean
   log-SPD/canonical rate ratio is below `0.995`;
3. at least `9/12` images have a ratio below one after averaging their two count strata, for each
   coder;
4. median covariance-stream reduction is at least `3.0%` for each coder;
5. no matched operating-point log-SPD reconstruction loses more than `0.05 dB` to the canonical
   envelope;
6. median encode and cold-decode time are each at most `1.25x` canonical RS;
7. the direction is stable across `N=640` and `N=2560`; and
8. canonical RS captures less than `75%` of log-SPD's whole-container gain over current RS.

Any failed gate closes this formulation without tuning or confirmation access. Passing all gates
authorizes the frozen odd-ID confirmation. Confirmation must independently pass gates 1--7; gate
8 remains a development mechanism guard.

## Required outputs

- immutable config/data/source manifests and executed-source archive;
- fitted-field and target hashes;
- one row per complete candidate stream with exact component bytes and timing;
- all five raw encode and cold-decode timing repetitions, not only their medians;
- cold-decoded PSNR, SSIM, MS-SSIM, covariance error diagnostics, and parity controls;
- per-cell Pareto envelopes, common-overlap area-rate ratios, paired bootstrap samples/intervals,
  gate ledger, and replay comparison; and
- an explicit outcome table for quality, convergence, performance, compression, and
  expressiveness with unauthorized axes marked unavailable.

## Stop rules

- No QAT, learned quantizer, vector quantizer, entropy model, direct-log-covariance training, or
  range retuning may rescue this task.
- Do not promote an SSPL2 syntax or production default from development evidence.
- If the chart passes, freeze a separate confirmation/production task before making code changes.
- If it fails, retain canonical RS as a separately attributable engineering candidate only if it
  independently beats current RS; do not relabel that as evidence for log-SPD.

## Outcome

The immutable decision artifact is
`results/comp007_gauge_free_covariance_dev_v4/analysis.json`. Its full audit rebuilt all 24 initial
fields, validated all 24 fitted fields and 12,096 complete streams, performed true decoded-field
ordinary re-encoding, and numerically replayed all 12,096 renders. It reports
`decision_ready=true`, no errors, and exact agreement for every replayed render hash in the frozen
environment.

`log_spd` failed gates 1--5, 7, and 8. Relative to `canonical_rs`:

| Metric | zlib-9 | zstd-9 | Frozen requirement |
|---|---:|---:|---:|
| Median whole-container reduction | `-0.4053%` | `+0.3426%` | at least `+1.0%` each |
| Bootstrap 95% upper rate-ratio bound | `1.007615` | `1.001999` | below `0.995` each |
| Image wins | `5/12` | `7/12` | at least `9/12` each |
| Median attributed covariance-stream reduction | `-1.6077%` | `+1.8145%` | at least `+3.0%` each |
| Worst canonical-envelope PSNR shortfall | `5.2945 dB` | `4.6074 dB` | at most `0.05 dB` each |
| Median encode-time ratio | `1.0432x` | `1.0485x` | at most `1.25x` each |
| Median cold-decode ratio | `1.1028x` | `1.1030x` | at most `1.25x` each |

Only the timing gate passed. The small zstd rate movement is below the minimum effect, its
bootstrap interval includes no qualifying benefit, and it is not stable enough to authorize
confirmation. Canonical RS itself is descriptively smaller than current RS by geometric-mean
area-rate ratios `0.99128` (zlib) and `0.99308` (zstd), but this task did not preregister a
standalone promotion gate for that engineering ablation; it remains benchmark evidence only.

The conclusion is narrow: gauge removal through this log-Euclidean chart does not improve
StructSplat's complete-stream rate--distortion under the frozen codec. Unquantized quality,
training convergence, renderer performance, and representation expressiveness remain untested.
Do not retune the chart or expose the odd Kodak IDs as a rescue.

Canonical artifact SHA-256 values are:

- `config.json`: `fbf1846492a930faf556c3ce9b8c98927c32bb1a20055ac91cc30ed9f52113f9`
- `fields.jsonl`: `a8d0c60a5cb5dd8b5bb1027f284386bb6261bb1f722460638d8f88a694f7f40a`
- `candidates.jsonl`: `c16be1e17b6a67a87077c3bb2169c4dd730eb593b192a8844ada628c85e83164`
- `artifact_audit.json`: `ad1ec6c889e818e4b1af4cc63a4f99959453534951f480554471bf14fc621aa5`
- `analysis.json`: `115c2e272a406b1d85313496a94c76e6a4f47c59e41b79e13f951f0ff464ea27`
- `executed_sources_v3.tar`: `cb6d0d8eb77b6b98328dd66290d8a514a746e00de07c10a2d23cc62959eb9753`
