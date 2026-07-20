# SGI entropy contingency for StructSplat

## Status: contingency only

This is a durable design note, **not an authorized task, implementation, preregistration, or
experiment**. It must not start before BENCH-016 reaches an audited outcome and this direction is
then explicitly selected. No BENCH-016 evidence is used here. Failure of any frozen gate below
closes this fixed-grid formulation without retuning; success authorizes only the disjoint
multirate confirmation described below, not integration into SSPL1 or a production SSPL2 format.

## Source mechanism and transfer boundary

The source is [SGI, arXiv:2603.07789v2](https://arxiv.org/abs/2603.07789v2), with its
[official repository pinned at commit
`1aa6e1f99026323f73a90f0a0d5c0af7080d51bb`](https://github.com/zx-pan/SGI/tree/1aa6e1f99026323f73a90f0a0d5c0af7080d51bb).
SGI partitions a large image into seed-defined local spaces, uses lightweight MLPs to generate
groups of structured Gaussians, fits the seed representation coarse-to-fine, and uses binary
hash-grid context to predict entropy distributions for seed attributes. Its compression result is
therefore a representation/generator/context system, not evidence that an entropy coder alone
will improve StructSplat.

The contingency transfers one narrow mechanism: **decoder-visible position conditions the
lossless probability model for already-quantized attributes**. It does not transfer SGI's seeds,
Gaussian generator, multiscale optimizer, or representation. The candidate is deliberately
reconstruction-invariant so rate can be isolated from quality and allocation.

## Exact benchmark-only SSPL2 candidate

Name: **reconstruction-invariant mean-conditioned binary-grid entropy coder**.

For each frozen QAT field, first produce exactly the SSPL1 quantized arrays and Morton order. The
candidate preserves the SSPL1 header, Morton mean stream, rotation symbols, quantization arrays,
and normalized renderer. It changes only the containers for the two absolute log-scale symbol
streams and three RGB symbol streams:

1. Decode means first and use their reconstructed indices to address a fixed `16 x 16` spatial
   grid with an eight-bit binary feature at each grid entry (`16 x 16 x 8` transmitted bits).
2. Feed that binary feature to a transmitted `int8` linear head. Integer-only `int32` inference
   emits integer location and log-scale indices for five discretized-logistic distributions, one
   for each of the two scale and three color streams.
3. Build range-coder probabilities from a versioned, fixed 15-bit integer CDF lookup table. The
   decoder must not depend on floating-point probability evaluation.
4. Count the range stream, binary grid, head, frequency/CDF metadata, format header, framing,
   termination, and every other decoder-required byte. Rotation remains on the unchanged SSPL1
   zlib path.
5. Train only the binary grid and `int8` head through a straight-through estimator. The fitted
   field, QAT ranges, quantized values, Morton order, and renderer remain frozen. No model state is
   shared for free across images.

Thus the candidate must decode the exact same five symbol arrays as SSPL1. It has zero authority
to trade distortion for rate or to alter the field.

## Controls and attribution

All controls start from the identical frozen arrays and use the same framing and complete-byte
accounting.

1. **SSPL1 control:** the unchanged per-stream zlib codec.
2. **Factorized range control:** the same range coder, but with a transmitted factorized empirical
   model per attribute channel and no spatial conditioning.
3. **Shuffled-position control:** the candidate's identical grid, head, optimizer, and byte
   accounting, but symbols are assigned deterministic pseudo-random mean positions. This retains
   model capacity while destroying the proposed spatial correspondence.

For each image and bit configuration, the primary baseline is the smaller complete stream from
controls 1 and 2, including a one-bit mode signal. The shuffled control is the causal attribution
test: an advantage over zlib or factorized coding alone is insufficient evidence for mean
conditioning.

## Frozen evidence program

### Data separation

Use only the CLIC 2019 Professional validation images. Normalize each basename to Unicode NFC,
compute

```text
SHA256(b"structsplat-comp008-v1\0" || UTF8(NFC_basename))
```

and rank basenames by the digest in ascending byte order. The first eight images are development;
the next eight remain sealed confirmation. Persist the resulting names and source hashes before
loading any pixels. Resize each image once with Lanczos to maximum side `768`, preserving aspect
ratio. Development results may select only pass/fail; they may not change the split, coder,
training, or thresholds. The sealed eight are opened only after a development pass authorizes
confirmation.

### Shared field and rates

For every image, fit one normalized-coordinate, constant-RGB, opacity-free `quadtree_wse` field
with `N=8192`, seed `0`, and `4000` steps. Apply the existing matched QAT separately at bit tuples
`(12,6,6,8)` and `(16,8,8,8)`, ordered as `(means, scales, rotation, colors)`. Freeze the resulting
quantization ranges and integer arrays before training or evaluating any entropy model. Candidate
and controls consume byte-identical frozen inputs.

### Integrity requirements

Every cell must reparse its complete byte stream and use a fresh, stream-only decoder. Candidate
and controls must agree bit-for-bit on all decoded symbol arrays and decoded tensors, and their
CPU reconstruction hashes must be identical. Decode timing begins from bytes alone and includes
model parsing, integer probability inference, range decoding, tensor construction, and the same
CPU render. Integrity is conjunctive and precedes every scientific gate; a mismatch is invalid/no
decision, not a quality result.

### Gates

The following gates apply independently to each bit tuple across the eight development images.
The candidate advances only if both tuples pass every gate:

- Complete-byte geometric-mean ratio to the per-image primary baseline is `<= 0.90`, at least
  `7/8` images win, the worst image ratio is `<= 1.02`, and the paired-bootstrap 95% upper bound
  on the geometric-mean ratio is `< 0.95`.
- Candidate scale-plus-color bytes, including all model and syntax bytes, are at least 10% smaller
  on average than the shuffled-position control and win on at least `6/8` images.
- Median fresh decode-plus-render time is `<= 1.5x` the primary baseline and the worst-image
  median is `<= 2.0x`.
- Peak process RSS is `<= 1.5x` the primary baseline.

The implementation must freeze the paired-bootstrap seed, resample count, timing repetitions,
machine binding, and tie handling before reading development pixels. Those procedural constants
are not selected in this contingency note and therefore cannot be chosen from observed outcomes.

If either bit tuple fails, close the fixed `16 x 16 x 8` grid and do not tune grid size, head
width, CDF precision, optimizer, or thresholds. A two-tuple development pass permits only the
same frozen evaluation on the sealed eight; it does not authorize a new model family.

## Novelty and prior-art audit

The expected novelty is **N1, at most N2-T**. SGI directly establishes binary hash-grid
contextual entropy models for structured Gaussian attributes. [HAC](https://eccv.ecva.net/virtual/2024/poster/1306)
directly establishes hash-grid-assisted context prediction for Gaussian compression. StructSplat's
open COMP-003 ladder already names Morton-context range/arithmetic coding as a planned entropy
rung. The defensible contribution can therefore be only a careful transfer and causal benchmark:
whether a tiny, fully transmitted, integer, mean-conditioned model reduces complete SSPL bytes
while provably preserving reconstruction. It is not a new entropy-model class.

Rejected alternatives for this contingency are:

- full SGI replication, seed-generated Gaussians, and coarse-to-fine seed fitting;
- adaptive `K`, learned allocation, pruning, or any change in Gaussian count;
- affine or frequency children and any richer decoded primitive;
- learned quantization steps or joint rate-distortion refitting;
- overlap-graph context, already unsupported by the local graph-predictor evidence;
- plain range coding without factorized and shuffled controls.

These exclusions keep the experiment interpretable: the only tested causal variable is useful
spatial conditioning of five existing SSPL symbol streams.

## Claim boundary

A pass would establish only complete-byte savings and bounded CPU decode overhead for this fixed
per-image model on the frozen CLIC subset, two bit tuples, `N=8192`, and max-side-768 fields. Since
all decoded tensors are identical, it cannot improve or make claims about image quality,
expressiveness, fitting convergence, allocation, renderer speed, or robustness to a different
representation. It would not reproduce SGI, establish large-image behavior, beat conventional or
learned image codecs, justify uncounted shared priors, or make SSPL2 production-ready.

A failure would reject only this fixed-grid, mean-conditioned implementation under complete-byte
and resource accounting. It would not show that contextual entropy models, SGI's structured
generator, or other decoder-visible contexts cannot compress StructSplat.
