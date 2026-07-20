# COMP-008: Mean-conditioned entropy-oracle killing test

## Status

**Protocol v1 frozen before CLIC pixels are decoded (2026-07-16).** This is the
predeclared SGI contingency selected by BENCH-016's audited `abandon SAD reuse` decision. It is a
benchmark-only necessary-condition screen. It does not implement a range coder, change SSPL1,
or authorize production code. The task, acquisition/benchmark sources, tests, transitive source
snapshot, environment, and remote-object identity must be hash-bound in a fresh output directory
before a selected PNG is decoded.

If either frozen bit tuple fails any gate, the decision is **kill this fixed-grid formulation**.
Do not tune its grid, overhead, model, field, QAT, split, or thresholds. If both tuples survive,
the result is only inconclusive and authorizes implementation of the separately specified SSP2E
coder on the same development fields. Confirmation remains sealed until an actual coder passes a
new, source-bound development protocol.

## Question and claim boundary

[SGI](https://arxiv.org/abs/2603.07789) predicts structured-Gaussian attribute distributions from
decoder-visible spatial context. COMP-008 transfers only the weakest corresponding hypothesis:

> Could exact reconstructed mean coordinates contain enough information about StructSplat's
> already-quantized scales and RGB values to save at least 10% of a complete SSPL1 stream after a
> small per-image conditional model is fully counted?

The experiment computes an optimistic lower bound. It gives every spatial cell its exact
empirical distribution for each modeled channel and charges only a fixed 656-byte candidate
syntax/model overhead. It does not charge finite-CDF approximation, arithmetic-coder redundancy,
learned-model suboptimality, optimizer failure, or decode compute. Therefore failure is decisive
for this candidate: a real self-contained coder cannot beat its lower bound. Survival is not a
compression result and does not imply that a coder can attain the bound.

The fitted fields and decoded tensors are unchanged. This task can address compression potential
only. It cannot improve or support claims about quality, convergence, renderer performance,
allocation, robustness, or representation expressiveness.

## Prior art and novelty boundary

- [SGI](https://arxiv.org/abs/2603.07789) uses seed-local structure and binary hash-grid context
  for structured-Gaussian entropy prediction.
- [HAC](https://arxiv.org/abs/2403.14530) already uses hash-grid-assisted context models for
  Gaussian compression.
- StructSplat COMP-003 already names Morton-context arithmetic/range coding as an entropy rung.

Accordingly this is `N0` method novelty and, at most, `N2-T` recipient evidence. Neither spatial
context coding nor the oracle is claimed as a new method. The useful contribution is a complete-
byte impossibility screen before implementing a fragile per-image coder.

## Frozen data and separation

Use the official CLIC 2020 Professional validation archive only:

```text
https://storage.googleapis.com/clic_datasets/clic2020_professional_valid.zip
expected Content-Length: 134862753
expected GCS generation: 1755735394217541
remotezip: 0.12.3
```

The acquisition helper must bind the remote object before and after ranged access, enumerate the
archive, reject unsafe/ambiguous members, and stream only the eight selected development PNG
members. It
must normalize archive stems to Unicode NFC, require exact case-sensitive uniqueness and `.png`,
reject traversal, absolute paths, backslashes, encryption, symlinks, duplicates, negative sizes,
and CRC/hash mismatches, and persist encoded-byte SHA-256 values. It never decodes pixels.

Development basenames, in this exact order, are:

```text
nomao-saeki-33553
martyn-seddon-220
zugr-108
jason-briscoe-149782
martin-wessely-211
stefan-kunze-26931
vita-vilcina-3055
philippe-wuyts-45997
```

Confirmation basenames, in this exact order, are:

```text
lobostudio-hamburg-75377
gian-reto-tarnutzer-45212
roberto-nickson-48063
wojciech-szaturski-3611
philipp-reiner-207
sergey-zolkin-21232
todd-quackenbush-222
alexander-shustov-73
```

The acquisition stage binds confirmation names and complete central-directory metadata, but it
must not open an archive member, extract, hash the uncompressed payload, or store a local PNG for
confirmation. No command in this oracle task may open, decode, resize, inspect dimensions of,
render, fit, or score a confirmation PNG. The oracle accepts development names only and fails
closed on every other basename.

Development preparation converts the encoded image to strict RGB and resizes once with Pillow
Lanczos to maximum side 768 while preserving aspect ratio. Use
`scale=768/max(H,W)` and `round()` for both output dimensions. Persist source PNG hash, decoded
native RGB `uint8` pixel hash/dimensions, derived PNG hash, and derived RGB pixel hash/dimensions
before fitting. The eight images are the only independent units.

## Shared fitted fields and two QAT copies

For each development image:

1. Seed NumPy, Torch CPU, and Torch CUDA with zero. Use one exact normalized-CUDA renderer after
   prebuilding its extension.
2. Build one `quadtree_wse` field with `N=8192`, seed `0`, `color_basis=constant`,
   `opacity_mode=none`, no covariance filter, and every split/growth/prune/relocate/structural-edit
   schedule disabled. Early stopping and checkpoint selection are disabled.
3. Run exactly 4,000 ordinary fitting updates. Persist the terminal field and its state hash.
4. Create two independent detached copies of that same terminal field. On each copy run exactly
   150 STE-QAT updates using the existing `codec.qat_finetune` path with frozen scale/color ranges
   and bit tuple `(means,scales,rotation,colors)` equal to `(12,6,6,8)` or `(16,8,8,8)`.
5. Encode the resulting copy as one complete, ordinary zlib-9 SSPL1 stream using the returned
   frozen codec configuration. Persist, hash, cold-parse, and cold-decode the stream.

The existing QAT helper does **not** freeze an SSPL1 mean range: its mean fake-quantizer uses the
image box, whereas the encoder later derives `_means_extent` from the final field and image box.
This task intentionally measures the shipped path and must not call it exact-lattice or fully
matched QAT. Persist each final field's off-image mean count, extrema, encoded integer mean-range
endpoints, and whether those endpoints differ from the image box. The oracle itself uses only the
final cold-parsed SSPL1 absolute symbols, so this implementation mismatch cannot invalidate its
rate lower bound, but it remains a QAT diagnostic and claim limitation.

Every stream must have `n=8192`, no opacity, Morton reorder, mean byte planes, circular rotation,
color deltas, normalized exact renderer semantics, the requested bits, and exact complete-byte
component accounting. Direct decode and an independently fresh parse must agree exactly on all
eight **ordered** absolute symbol arrays. They must also produce the same decoded-field boundary
state hash over named shapes/dtypes/contiguous float32 tensor bytes, with per-tensor maximum
absolute difference `<=1e-6`. A CPU reference render may be persisted as a diagnostic, but no
CUDA-render-to-CUDA-render equality is a validity oracle. Any symbol, order, tensor, or accounting
mismatch is invalid/no-decision, not a failed compression result.

The eight **absolute** symbol arrays are:

```text
mean_x, mean_y, scale_x, scale_y, rotation, R, G, B
```

SSPL1's modular mean and color transforms must be inverted before the oracle is evaluated.
The eventual SSP2E decoder must reproduce these exact arrays in this exact row order before
dequantization. Equality only after float dequantization is insufficient: float aliases,
canonicalization, or row permutation may not be used to reduce entropy.

## Oracle and complete-byte lower bound

For a reconstructed absolute mean symbol `(qx,qy)` at `bits_means=b`, define its decoder-visible
cell without floating point:

```text
cx = min(15, (qx * 16) >> b)
cy = min(15, (qy * 16) >> b)
cell = 16 * cy + cx
```

For each of the five modeled absolute symbol channels
`{scale_x,scale_y,R,G,B}`, let `n_c` be the count in cell `c` and `n_c,s` the count of symbol `s`.
The conditional empirical entropy in bits is

```text
H = sum_c [n_c log2(n_c) - sum_s n_c,s log2(n_c,s)]
```

with `0 log2 0 = 0`. No floating logarithm may determine a row or gate. For all five channels and
cells form the exact positive integers

```text
Num = product_c n_c ^ n_c              # repeated independently for each channel
Den = product_(c,s:n_c,s>0) n_c,s ^ n_c,s
```

Let `k` be the unique nonnegative integer satisfying

```text
Den * 2^(8k) <= Num < Den * 2^(8(k+1)).
```

Then `H_down=8k`, `H_down<=H<H_down+8`, and `floor(H_down/8)=k` are certified by integer
comparisons. Persist the numerator/denominator bit lengths and canonical integer hashes as the
certificate. Floating entropy values may be emitted only as diagnostics.

The counterfactual SSP2E container is fixed as follows. All bytes are mandatory,
noncompressible, nonelidable, per-image bytes:

- 20-byte outer framing `<5sBBBQI>`;
- 100-byte fixed header `<III4B4i12f5I>`;
- nine 20-byte stream-directory entries `<IQQ>`;
- one raw 256-byte binary grid;
- one raw 100-byte integer head.

Total fixed overhead is exactly `656` bytes. The format also reuses the **payload bytes only** of
the ordinary SSPL1 zlib means and rotation streams. Framing for all nine SSP2E streams is already
in the fixed directory. The optimistic complete-container lower bound is

```text
L = 656
    + SSPL1_means_zlib_payload_bytes
    + SSPL1_rotation_zlib_payload_bytes
    + k
```

The complete SSPL1 control is `S=len(sspl1_blob)`. The primary per-image ratio is `L/S`.
No factorized coder is required: the eventual primary actual baseline can be
`P=min(S,factorized_actual)<=S`, hence `actual_candidate/P >= L/S`. Failing against `S` is already
a valid impossibility result against the stronger future baseline.
The candidate itself has no per-image fallback/mode that selects SSPL1, factorized coding, another
grid, or another context. Such a selector would be a different format with additional syntax and
a different claim.

## Future SSP2E syntax bound by this oracle

These rules are not implemented or measured here. They prevent a surviving oracle from later
changing the candidate whose compulsory bytes justify `L`. All integer fields are little-endian.
The 20-byte outer fields are magic `SSP2E`, version `1`, flags `0`, stream count `9`, total
container bytes, and one CRC-32 over every following byte. The 100-byte header fields, in order,
are `N,H,W`; the four bit depths; four exact signed integer SSPL1 mean-range endpoints; twelve
float32 values `(scale_lo[2],scale_hi[2],color_lo[3],color_hi[3],aa_dilation,sigma_cutoff)`; and
five uint32 values `(render_chunk,renderer_id,support_fade,cdf_table_id,zlib_level)`. This is
sufficient for the frozen opacity-free normalized renderer because SSPL1's fitted scale/color
ranges originate as float32 values, its mean extents are integral, and the remaining transform
flags are fixed by SSP2E version 1.

The nine directory records have immutable IDs and order:
`means_zlib,scale_x_arith,scale_y_arith,rotation_zlib,R_arith,G_arith,B_arith,grid_raw,head_raw`.
Each `<IQQ>` record contains its ID, absolute offset, and byte length. Payloads are contiguous in
that order with no aliases, gaps, overlap, or trailing bytes. Mean and rotation payloads are copied
byte-for-byte from SSPL1; grid/head lengths are exactly 256/100. Arithmetic payload lengths may be
zero only for an empty field, which this task never has. Directory validation plus the single
outer CRC supplies container integrity; no uncounted per-stream checksum exists.

The binary grid stores one 8-bit feature per spatial cell. The head has five fixed channel records;
each stores `w_mu[8]:int8`, `b_mu:int16`, `w_s[8]:int8`, and `b_s:int16` (20 bytes/channel).
Feature components are mapped to `2*bit-1`; signed `int32` dot products emit a location clamped to
`0..63` and a scale clamped to `0..15`, with no learned shift. Model IDs, stream order, offsets,
lengths, the container CRC, and end position are fixed and validated. A static versioned 15-bit CDF table is
bound before a target symbol is encoded; all frequencies are positive and total exactly 32768.
For each of the five arithmetic streams, every symbol CDF is exclusively a deterministic function
of `(frozen bit tuple, channel ID, absolute-mean cell, transmitted grid, transmitted head, static
CDF table)`. It may not depend on Gaussian index/order beyond the paired mean cell, a previous
symbol, another attribute channel, a decoded prefix, adaptive counts, or any other context/state.
The five streams are coded independently. This exclusivity is what makes the sum of marginal
cell-conditioned empirical entropies a valid lower bound; a joint/autoregressive coder is a new
candidate and is not killed by this task.
Each modeled stream decodes exactly `N=8192` symbols from its own arithmetic payload. Header,
directory, grid, and head bytes may influence reconstruction only through the declared static
per-symbol CDF lookup. They may not carry escape symbols, exceptions, literal residuals,
post-decode corrections, run lengths, a shared/joint stream, an alternate row permutation, or any
other side channel. Every decoded symbol must come from the declared channel alphabet and must
match the bound ordered SSPL1 array exactly.

The future arithmetic coder uses inclusive 32-bit `low/high`, `T=32768`, normalization while the
range is below `2^30`, and

```text
unit = range // T
child_low  = low + unit * cumulative
child_high = child_low + unit * frequency - 1
```

Unused tail states and decoded scaled values `>=T` are invalid. Standard E1/E2/E3 renormalization,
a contained-dyadic canonical terminator, byte padding, EOF-only zero extension, and byte-identical
decode/re-encode verification are compulsory. The coder must enforce
`2^(8B) * product(frequency) >= T^N`. Exhaustive tiny-alphabet tests are required before any real
pixel is decoded. These details may be revised only as a new named protocol after an oracle
survival; revisions cannot retroactively change this task's decision.

## Frozen paired bootstrap

Use exactly 100,000 paired resamples. For row `r in 0..99999`, compute

```text
d = SHA256(b"structsplat-comp008-bootstrap-v1\0" || uint64_le(r))
index[r,j] = d[j] & 7, j=0..7
```

Materialize the `100000 x 8` matrix as C-order `uint8` and bind its SHA-256 before any development
pixel is decoded. Its frozen byte SHA-256 is
`cca3d7403c0c41ea5b5cd8604e566114939ffe90177f9a2ec2a71490ba7232b9`. For each tuple and
resample, compute the geometric mean of the eight selected
`L/S` ratios. Sort all 100,000 values ascending and take zero-based element `97499`; do not
interpolate. The implementation may use the exactly equivalent rank count, but the gate must be
decided by rational cross-products, never floating point. Compare geometric-mean products against
`(9/10)^8`, worst ratios against `51/50`, and bootstrap products against `(19/20)^8` with exact
integers. Equality passes the first/worst gates and fails the strict bootstrap gate.

## Development gates and action

Apply the following conjunctively and independently to each bit tuple:

1. geometric mean of the eight `L/S` ratios is `<=0.90` (equality passes);
2. at least seven images have `L<S` (strict wins);
3. every image has `L/S<=1.02` (equality passes); and
4. the fixed paired-bootstrap 97.5th-percentile geometric mean is `<0.95` (strict).

For each tuple, report either `reject tuple under frozen margin` or `survive necessary-condition
screen`. A failed tuple establishes impossibility only for that tuple's fixed QAT field, exact
ordered symbols, 16x16 cell address, five independent marginal streams, and 656-byte SSP2E under
these gates. It says nothing negative about a tuple that survives.

Because COMP-008 requires one unchanged format to work at both tuples, if **any** gate fails for
**either** tuple, write the policy decision
`abandon jointly-required two-tuple COMP-008 candidate`, stop, and do not decode confirmation. Do not add a
shuffled oracle or timing/RSS gate: oracle entropy already uses exact cell distributions, so a
shuffle would test a different question; no coder exists to time. Report factorized entropy and
shuffled-position entropy only if they were frozen before pixel access and only as diagnostics;
they cannot enter or rescue the decision.

If both tuples survive, write `oracle inconclusive; implement coder`. That result is not a pass,
compression win, or authorization to inspect confirmation. The next source-bound task must build
and validate the exact coder, count every byte, compare against complete SSPL1 and a complete
factorized range stream, and only then define any causal shuffled control or resource gates.

## Lifecycle, provenance, and replay

Legal stages are `preflight`, `acquire`, `prepare-dev`, `run-dev`, `analyze`, and `replay`.

- `preflight` runs synthetic/no-pixel correctness tests, prebuilds CUDA, writes the source and
  environment binding, frozen bootstrap matrix/hash, commands, and a complete source archive.
- `acquire` streams encoded selected PNG members and writes its self-hashed manifest.
- `prepare-dev` alone may decode the eight development PNGs and writes the frozen target manifest.
- `run-dev` is append-only and resumable by image/tuple, never by silently replacing an existing
  artifact. It persists initial/base/QAT fields, SSPL1 bytes, symbol arrays, hashes, configs,
  timings, and integrity checks.
- `analyze` requires all 16 unique image/tuple rows and revalidates every referenced artifact.
- `replay` independently reparses the persisted SSPL1 streams and recomputes all symbols, entropy
  bounds, rows, aggregates, gates, and decision from captured source without fitting again.

Fail closed on source/environment/remote binding drift, a nonempty output directory with a
different binding, missing/duplicate cells, confirmation access, unexpected file/hash/schema,
nonfinite values, a changed bootstrap matrix, illegal stage order, or incomplete source closure.
Record process and CUDA versions, device, deterministic/thread settings, all resolved configs,
commands, start/end times, and separate ordinary-fit/QAT/encode/decode timings. Timings are
diagnostic only.

## Acceptance criteria

- [ ] Task, implementation, tests, environment, bootstrap matrix, remote object, and selected
      archive members are bound before development pixel decode.
- [ ] Focused tests cover unsafe ZIP entries, exact symbol inversion, cell addresses, entropy
      direction, full byte accounting, threshold ties, bootstrap order/hash, confirmation sealing,
      malformed/trailing SSPL1, and replay drift.
- [ ] Eight base fields, sixteen independent terminal-QAT fields, sixteen complete SSPL1 streams,
      and all integrity records are fresh and complete.
- [ ] Analysis and independent persisted-stream replay agree exactly on every non-timing value and
      the decision.
- [ ] A negative result stops without retuning or confirmation access; survival authorizes only
      the actual-coder task described above.
- [ ] Task/index, benchmark docs, dated research report, and ARA agree; focused/full tests, Ruff,
      source-snapshot verification, artifact audit, and diff hygiene pass.

## Interfaces allowed

New acquisition/benchmark modules, tests, task/research documentation, ignored result evidence,
and ARA records only. No production codec, fitter, renderer, configuration, CLI, default, or
existing frozen benchmark source may change.

## Depends on

COMP-001/002/003/004, COMP-006/007, BENCH-016, SGI, and the 2026-07-16 hard-frontier research
prompt/evidence program.
