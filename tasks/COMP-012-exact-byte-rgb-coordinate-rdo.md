# COMP-012: Exact-byte RGB coordinate RDO with unchanged SSP2F decoding

## Status

**`NO_GO_PRE_DATA`; repaired protocol draft, with all COMP-012 payload access forbidden.** The
first hostile review required data separation, direct-fine, source-only, equal-rate, convergence,
transfer, and accounting controls. A later hostile implementation review found additional
objective-ordering, transcript-growth, materialization, numerical-terminal, component-schema, and
split-brain defects. This revision repairs those protocol defects, but it does not freeze new data
or authorize a run. The exact repeated-CDF primitive and objective/search/size-oracle/source-bound-
operator/paired-adapter cores are implemented mechanics only. End-to-end lifecycle orchestration,
poisoned-commit integration, runtime and transcript caps, frozen gate binding, source-bound replay,
and empty-directory preflight remain unfinished.
Do not extract, decode, resize, fit, render, or score any COMP-012 development or confirmation
member until every blocker is frozen, the task digest is bound, and empty-directory preflight
succeeds.

This is a new experiment, not a repair, continuation, or confirmation of COMP-011. COMP-011 is
terminal invalid/no-decision and supplies no method-failure authorization. The only inherited
evidence is non-source-bound calibration on already exposed COMP-011 development cells. It
justifies infrastructure and a new frozen experiment; it is not COMP-012 development evidence.

### Already-exposed calibration justification only

An earlier source-teacher mechanics probe found that exact-byte-aware RGB coordinate moves reduced
CPU source-teacher SSE by 50.10% at +51 bytes, while an entropy-monotone variant reduced it by
31.65% and saved 103 bytes. Two later target-guided calibration probes used only the previously
exposed COMP-011 `(12,6,6,8)` development cells and matched each cell's incumbent complete-byte
cap:

| Already-exposed cell | Conservative PSNR gain | Conservative MS-SSIM gain | Conservative LPIPS decrease |
|---|---:|---:|---:|
| `jason-briscoe-149782/m12_s6_r6_c8` | `+0.931 dB` | `+0.000196` | about `0.000449` |
| `nomao-saeki-33553/m12_s6_r6_c8` | `+0.861 dB` | `+0.000171` | about `0.002742` |

Both are single-cell calibration signals from unsealed mechanics runs. Neither search was terminal
after ten fine sweeps. They therefore establish neither convergence nor a frozen method, threshold,
runtime, compression, performance, generalization, or promotion claim. No confirmation payload and
no COMP-012 roster member was opened; these two identities remain excluded from COMP-012 and cannot
be reused as its development evidence. The numbers may justify adapter and protocol work only and
must not set a COMP-012 cap, gate, exclusion band, roster choice, or stopping rule.

## Question and claim boundary

For a cold-decoded quantized StructSplat field, can an encoder-only search over absolute RGB symbols
improve conservative target PSNR under the strongest incumbent's exact complete-stream byte cap,
while:

- keeping the decoder and canonical SSP2F wire grammar unchanged;
- preserving every header/range bit, row, geometry symbol, and non-RGB symbol;
- objective-scoring every label and pricing every rate-relevant label by exact complete SSP2F
  bytes under frozen dominance rules;
- reaching a reconciled, explicitly epsilon-bounded coordinatewise terminal state;
- transferring from a deterministic clamped CPU objective to repeated persisted-CUDA rendering;
- beating direct-fine, target-denied, uniform-lattice, and exact-equal-rate controls; and
- satisfying frozen encoder and bytes-to-boundary decode resource gates?

A pass would establish development evidence only for this fixed encoder search. A search-core
`core_terminal` is only an arm-local prerequisite; it is not a lifecycle pass or promotion. Even a
full lifecycle pass cannot establish global optimality, bit-exact real-arithmetic stationarity,
field-fitting convergence, renderer acceleration, new decoded expressiveness, generic
coordinate-descent or RDO novelty, state of the art, or confirmation.

Nearest prior art already covers operational codec RDO, hard target-rate allocation,
coordinate-descent quantization, decoder-compatible index optimization, perceptual RDO,
test-time encoder optimization, and renderer-aware Gaussian rate-distortion training. The narrow
recipient-specific contribution is evidence for exact complete-stream byte accounting per accepted
RGB edit with checkpoint-reconciled ordinary streams, unchanged SSP2F decoding, fixed
normalized-splat rendering, causal source/target/equal-rate controls, and cold replay. Do not claim
a new generic RDO method.

## Frozen archive, disjointness, and roster

Use only the already metadata-bound official CLIC 2020 Professional validation archive:

```text
URL: https://storage.googleapis.com/clic_datasets/clic2020_professional_valid.zip
Content-Length: 134862753
GCS generation: 1755735394217541
central-directory payload SHA-256:
634d5e278665a32199fcaa7b161e128426e761f81a09c5b129621907984b45a2
```

Exclude all sixteen COMP-008/009/011 development and confirmation identities. Without opening any
member payload, rank the remaining NFC-normalized `.png` archive paths by unsigned lexicographic
order of:

```text
SHA256(
  UTF8("COMP-012-exact-byte-rgb-rdo-v1") || 0x00 ||
  ASCII("634d5e278665a32199fcaa7b161e128426e761f81a09c5b129621907984b45a2") ||
  0x00 || UTF8(archive_path_nfc)
)
```

The first eight eligible identities are development, in this order:

```text
todd-quackenbush-27493
kazuend-28556
michael-durana-82941
davide-ragusa-716
paul-itkin-46101
juskteez-vu-1041
jared-erondu-21325
felix-russell-saw-140699
```

The next eight are the unopened confirmation roster:

```text
amy-zhang-15940
dogancan-ozturan-395
casey-fyfe-999
thong-vo-428
jeremy-cai-1174
schicka-307
alberto-montalesi-176097
ales-krivec-15949
```

Preflight must recompute the full eligible ordering and prove disjointness from every prior
development/confirmation roster and exposed RDO target. Acquisition may bind confirmation names and
central-directory metadata only. COMP-012 may never range-request, extract, hash member payload,
decode, inspect dimensions, resize, fit, render, or score a confirmation member.

Development preparation follows COMP-008 exactly: strict RGB, one Pillow-Lanczos resize to maximum
side 768 preserving aspect ratio, `scale=768/max(H,W)`, and `round()` for both dimensions. Persist
encoded source, native decoded RGB, derived PNG, and derived RGB pixel hashes and dimensions.
Images, not tuples, fields, sweeps, edits, or metric repetitions, are the independent units.

Before fitting, compare each opened development member's encoded-byte hash, native strict-RGB
sample hash, derived-PNG hash, and derived strict-RGB sample hash against every corresponding known
hash from a prior development/confirmation/RDO authority and against the other seven selected
development members. An exact collision is invalid/no-decision. Do not replace it with the next
ranked archive member, change the roster, or open a confirmation payload. A prior confirmation
whose payload hash was intentionally never computed is excluded by its frozen identity only; this
rule never authorizes opening it to obtain another hash.

## Frozen base-field producer

For each development image:

1. Seed NumPy, Torch CPU, and Torch CUDA with zero.
2. Build one `quadtree_wse` field with `N=8192`, seed zero, constant color,
   `opacity_mode=none`, no covariance filter, and all split/growth/prune/relocate/structural-edit
   schedules disabled.
3. Run exactly 4,000 ordinary updates with the persisted exact normalized CUDA renderer. Disable
   early stopping and checkpoint selection; persist the terminal field.
4. Create two independent detached copies and run exactly 150 STE-QAT updates with frozen
   scale/color ranges at tuples `(12,6,6,8)` and `(16,8,8,8)`.
5. Encode each terminal copy as canonical zlib-9 SSPL1; strict-parse, cold-decode, canonically
   re-encode, and persist the complete stream and absolute symbols.

Development preparation and base-field fitting are explicitly target-authorized producer stages.
After the two base streams are sealed, every source-arm worker receives only those streams and is
filesystem-denied all source/derived target pixels. Target access is not released to any RDO worker
until all source arms are lifecycle-validated, unpoisoned, terminal, and sealed; a search-core
status alone is insufficient. Captured COMP-012 replay starts from the sealed base stream
authorities and does not claim or attempt bit-exact CUDA refitting.

Before the task digest is frozen, preflight must serialize one canonical, target-free producer
configuration record containing every resolved initializer, fit, optimizer, renderer, QAT, codec,
range, thread, dtype, and seed field plus the source/callable hashes that consume it. Referring to
"COMP-008 defaults" or whatever values the installed CLI currently resolves is not an authority.
The record SHA-256 must be inserted into this task/source binding before any development payload is
opened; a missing field, mutable default, or runtime-resolved value is invalid preflight.

Analyze the two tuples separately. Failure of one tuple cannot be hidden by pooling it with the
other.

## Immutable decoded state and strongest incumbent

For each cell let `x in uint8[8192,3]` be the cold-decoded absolute RGB symbols. Every arm keeps
bitwise fixed:

- the full header, flags, dimensions, count, bit tuple, color-range float bits, and row order;
- absolute means, scales, and rotation symbols;
- geometry, means, scale, and rotation decoded fields;
- the SSP2F 280-byte prefix and ordinary SSP2F grammar/decoder;
- arithmetic termination, empirical normalization, zlib level/version, and canonicality rules.

Reproduce all six exact incumbent formats `SSPL1`, `SSP2Z`, `SSP2E`, `SSP2S`, `SSP2L`, and
`SSP2F`. Define `Q_i` as the smallest complete byte length using the COMP-011 tie order:

```text
SSPL1, SSP2Z, SSP2E, SSP2S, SSP2L, SSP2F
```

Every mandatory format must strict-parse, cold-decode to the same field boundary, and canonically
re-encode byte-identically. A missing or invalid format makes the cell invalid; it cannot be
dropped from the minimum. `Q_i` is the common hard byte cap and its decoded render is the incumbent
quality reference.

## Exact SSP2F byte oracle

Every RGB label that reaches the exact-rate stage is priced as:

```text
280
+ means_zlib
+ scale_x_arith + scale_y_arith
+ rotation_zlib
+ R_arith + G_arith + B_arith
+ canonical_zlib9(full five-channel empirical-frequency model)
```

Every initial state, exact-price response, paired commit receipt, sweep reconciliation, terminal
certificate, and ordinary encoded blob uses the exact numeric mapping
`structsplat.comp012.ssp2f-component-bytes.v1`. Its keys are neither extensible nor aliases:

```text
prefix
means
scale_x_arithmetic
scale_y_arithmetic
rotation
red_arithmetic
green_arithmetic
blue_arithmetic
model_zlib9
total
```

`prefix` is exactly `280`; the next eight values are the physical payload lengths in canonical
factorized directory order; and `total` equals both the sum of those nine non-total fields and the
physical complete-stream length. All values are nonnegative integers, every required payload is
nonempty, and missing, extra, differently named, reordered-before-hashing, or inconsistent fields
are invalid/no-decision. The ordinary parser's directory lengths are the reconciliation authority;
an arbitrary mapping such as `{"payload": ..., "total": ...}` is never accepted as production
evidence.

For a changed RGB channel, recompute its entire histogram normalization, size the full
order-sensitive arithmetic sequence, and recompress the complete five-channel model. Old/new-bin
deltas, ideal entropy, fractional bits, and cached compressed model segments are forbidden as rate.

The native allocation-free repeated-CDF size function and the physical encoder must share one
arithmetic state machine, termination rule, pending-bit behavior, and byte padding. Required proofs
include:

- every physical bit-length residue modulo eight and long E3 pending chains;
- empty input and direct-C ABI pointer/overflow failures;
- A64 and A256 random sequences, including `N=8192`;
- a third-bin normalization change such as counts `[8,8,1]`, edit `8 -> 9`;
- a histogram-preserving position swap whose arithmetic length changes;
- baseline, random edit, sequential commit, accepted-sweep, and terminal equality:
  `oracle complete bytes == len(ordinary SSP2F encode)`;
- strict final parse, cold decode, exact proposed RGB, exact non-RGB identity, and canonical
  byte-identical re-encode.

Cache only state-independent components and explicitly validated current-state channel/model sizes.
Every proposal carries the current epoch and state digest; stale proposal commits fail closed.
Record compile-time and runtime zlib versions.

## Deterministic CPU color, target, renderer, and objectives

### Decoded colors and target

For a proposed absolute RGB symbol matrix `y` and `levels=2^bits_color-1=255`, construct colors in
the exact cold-boundary operation order:

```python
lo = np.asarray(header.color_lo, dtype=np.float64)
hi = np.asarray(header.color_hi, dtype=np.float64)
y64 = np.asarray(y, dtype=np.float64)
C_f32 = np.ascontiguousarray(
    (lo + y64 / 255 * (hi - lo)).astype(np.float32)
)
C = np.ascontiguousarray(C_f32, dtype=np.float64)
```

The header endpoints are their parsed float32 values promoted to float64; no endpoint, symbol, or
result is refit or clipped during dequantization. Require C-contiguous finite shapes
`y:uint8[8192,3]`, `C_f32:float32[8192,3]`, and `C:float64[8192,3]`. `C(x)` uses the identical
path. This is the existing SSP2 decoded-boundary rule: float64 arithmetic followed by exactly one
cast to float32, then promotion for the benchmark objective.

Let `T_u8` be the copied C-contiguous strict-RGB `uint8[H,W,3]` target. Construct it exactly as the
quality authority does:

```python
T_hwc_f32 = torch.from_numpy(T_u8.copy(order="C")).to(
    dtype=torch.float32, device="cpu"
) / 255.0
T_hwc = np.ascontiguousarray(T_hwc_f32.numpy(), dtype=np.float32).astype(np.float64)
```

The division is one CPU IEEE-float32 elementwise operation. Do not divide `uint8` directly in
float64, apply ICC/gamma/color management, normalize again, or change HWC order.

### Frozen normalized color operator

Construct the target-independent sparse operator `A` once per cell from the exact cold-decoded
float32 geometry. Build
`GaussianField(means,decoded_log_scales,rotations,colors,opacities=None)` and use its
`conics(header.aa_dilation)`, `radii(header.sigma_cutoff,header.aa_dilation)`, and the header's
dimensions. Reproduce the normalized reference support rectangles, rounded centers, integer pixel
coordinates, `_support_weight`, `support_fade=false`, header `sigma_cutoff`, opacity absence, and
renderer epsilon `1e-8`.

Compute quadratic forms, raw weights, the per-pixel denominator `float32 index_add_`, and
`raw_weight/(denominator+1e-8)` in CPU torch.float32 in frozen Gaussian/support order. Drop only
exact float32 zero normalized weights, then promote each retained weight once to float64. Preserve
each column's generated pixel order. Define `A C` by initializing C-contiguous
`float64[H*W,3]` zeros and, for Gaussian columns `i=0..8191`, adding
`weight * C[i,:]` to that column's pixels in the stored order. Reshape only after accumulation.
No BLAS-dependent dense multiplication, reordered sparse reduction, CUDA value, or target may
construct the objective authority. The builder and objective worker set Torch intra/inter-op
threads to one, enable deterministic algorithms, and require
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.

For any state `y`, define in this exact order:

```python
Y = np.clip(apply_A_in_frozen_order(C(y)), 0.0, 1.0)  # float64 HWC
delta_source = Y - Y_source                           # Y_source from x, same path
delta_target = Y - T_hwc
```

For either residual, compute the scalar objective with
`float64-squared-residual-block-reduction-v1`: square into a C-contiguous float64 HWC array, flatten
in C order, split into consecutive blocks of exactly 4,096 scalar elements, sum each block in
ascending order with `np.sum(..., dtype=np.float64)`, then sum the block vector in ascending order
with the same operation. The last block may be shorter and no empty block exists. The arithmetic
schema string, block size, accumulation order, and all reconciliation constants are part of the
objective configuration hash. A direct whole-array `sum`, BLAS reduction, scalar-delta accumulator,
or different block size is a different objective implementation.

All arrays, squared residuals, blocks, and objectives must be finite and nonnegative. Clipping
happens after normalized accumulation and before subtraction; target values are already in
`[0,1]`. Persist the exact source-render, target, sparse operator, denominator, support, reduction
configuration, and objective-state hashes.

Synthetic preflight must prove the sparse operator against the deterministic CPU renderer and
prove its column-action and clamped-objective deltas on dense/sparse fixtures. After each base
stream is sealed and before source search, a target-denied per-cell gate must compare this operator
with the persisted CUDA renderer on `x` and frozen hash-derived legal RGB perturbations. Its exact
max-absolute/objective-delta tolerances require exposed-data calibration and remain a freeze
blocker; a failed parity gate is invalid/no-decision, not a method failure.

The incremental objective cache contains the unclamped render, clamped render, residual,
squared-residual array, 4,096-element block sums, scalar objective, epoch, and RGB state digest.
One scalar proposal replaces only the affected residual entries, re-sums every touched complete
block, and then re-sums the complete candidate block vector. It must not form
`current_scalar + local_delta`. Commit repeats that cache update and requires `float.hex()` equality
with the proposal before advancing the objective epoch. Proposal scoring and exact-byte pricing do
not mutate either engine.

At initial state, the end of every local sweep including a zero-accept sweep, every accepted
full-audit jump, and the terminal boundary, independently rebuild the complete unclamped render,
clamped render, residual, squared residuals, block sums, and scalar objective from the current RGB
state. Unclamped/clamped/residual arrays use `atol=1e-12, rtol=1e-13`;
squared-residuals, block sums, residual-derived scalar SSE, and cached scalar SSE use
`atol=1e-13, rtol=1e-13`, always as
`atol + rtol*max(abs(first),abs(second))`. Compare each cache, persist both values and responsible
maxima, and rebase only after every check passes. The same checkpoint performs one
ordinary SSP2F encode, strict parse, cold decode, exact RGB/non-RGB comparison, canonical
byte-identical re-encode, exact component-schema comparison, and state-digest reconciliation.
These are the only routine full materializations; an accepted scalar edit is not full-rendered or
ordinary-encoded merely to select or commit it. Any nonfinite/negative cache, tolerance failure,
ordinary/oracle disagreement, or failed cold-canonical check poisons the arm as
invalid/no-decision and permanently latches the paired coordinator's poison state, including all
accepts since the previous checkpoint.

Accept only if:

```text
candidate_D < current_D - 1e-12 * max(1, abs(current_D))
```

The tolerance is numerical hygiene, not a scientific effect-size gate. Because candidate and
current values are incremental float64 reductions, conservatively bound either scalar value `v` by

```text
epsilon(v) = (1e-13 + 1e-13 * abs(v)) / (1 - 1e-13)
```

After scoring every label at one coordinate, set
`epsilon_coordinate=max_label epsilon(candidate_D)`. Let
`tau(v)=1e-12*max(1,abs(v))`,
`margin=(current_D-tau(current_D))-candidate_D`, and use the **same**
`margin_error=epsilon_coordinate+(1+1e-12)*epsilon(current_D)` for every label at that coordinate.
`margin>margin_error` is `definitely_improving`;
`margin < -margin_error` is `definitely_nonimproving`; every other value, including either
equality, is `ambiguous_improvement`. Only a definitely improving winner may commit. An ambiguous
label is still exact-priced when objective ordering reaches it, but it cannot be silently coerced
to an accept or ordinary reject. Persist its interval and margin. If a later checkpoint exceeds
the assumed bound, the intervening search is invalid rather than retroactively reclassified. A
terminal may therefore claim only that no feasible scalar move improves beyond the frozen
numerical margin—an epsilon-terminal claim—not bit-exact stationarity of independently rebuilt
real arithmetic.

## Frozen alphabets, neighborhoods, and starts

The coarse lattice is:

```text
L3 = {0,3,6,...,255}
```

The fine alphabet is all integers `L1={0,...,255}`. Independent step-3 initialization maps every
source scalar to its nearest `L3` value, breaking any exact tie toward the lower value.

For a scalar at value `v`, the efficient local coarse neighborhood is the in-range immediate
predecessor and successor in `L3`. The local fine neighborhood is the union of in-range
`{v-1,v+1,v-3,v+3}` so direct fine search inherits every coarse displacement. A search may not
omit objective scoring for a legal local neighbor. It may suppress an exact-byte query only through
the frozen objective-first classification and dominance rule below.

Local zero-acceptance is not terminal. After any local zero-accept sweep, perform one exhaustive
full-label audit using the first-definitely-improving-coordinate procedure under Coordinate search.
A `core_terminal` certificate requires an entire exhaustive full-label audit with zero definitely
improving feasible accepts, after scoring every alternate label and resolving exact-rate queries
under the objective-first rule. Feasible ambiguous labels and dominance-pruned labels remain
explicit in its bounded summary. This earns only epsilon-bounded coordinate terminality over the
frozen scalar alphabet, order, cap, and arithmetic; without it an output is incomplete. Nested and
direct paths still test order-dependent basins, but neither receives a larger terminal move set.

The target-blind uniform-lattice menu is provisionally:

```text
step in {1,2,3,4,5,6,8,10,12,16}
L_step = sorted(unique({0, step, 2*step, ...} union {255}))
```

For every positive integer step `s`, the ellipsis means all `k*s<=255` for integer `k>=0`, followed
by an explicit endpoint `255` and unsigned numeric sort/deduplication. Independently map every
source scalar `q` by minimizing `(abs(q-a),a)` over `a in L_step`; hence an exact midpoint chooses
the lower reconstruction value. Apply this independently in original row/channel order. The
complete SSP2F blob is then encoded without search. The step-3 initializer is exactly this rule
with `s=3`.

The menu itself must be performance-calibrated on synthetic or already exposed data and frozen
before COMP-012 development payload access. Calibration may remove or retain steps only before the
task digest is frozen; it may not alter the rounding, endpoint, eligibility, or tie rules.

## Required arms

| Arm | Start | Search/objective | Byte cap | Role |
|---|---|---|---:|---|
| `incumbent_Q` | cold exact `Q_i` | none | `Q_i` | strongest exact incumbent |
| `uniform_source` | every frozen lattice | no search; source objective chooses the feasible lattice | `Q_i` | strongest target-blind lattice |
| `coarse_source` | independent step 3 | `L3`, source objective | `Q_i` | target-denied coarse terminal |
| `nested_fine_source` | terminal `coarse_source` | `L1`, source objective | `Q_i` | target-denied nested control |
| `direct_fine_source` | independent step 3 | `L1`, source objective | `Q_i` | target-denied path control |
| `coarse_target` | independent step 3 | `L3`, target objective | `Q_i` | target-guided coarse terminal |
| `nested_fine_target_Q` | terminal `coarse_target` | `L1`, target objective | `Q_i` | sole primary arm |
| `direct_fine_target_Q` | independent step 3 | `L1`, target objective | `Q_i` | direct-fine control |
| `nested_fine_target_equal` | terminal `coarse_target` | `L1`, target objective | exact terminal `coarse_target` bytes | equal-rate alphabet-release control |

`uniform_source` is chosen by lowest source objective among feasible predeclared lattices, then
complete bytes, numeric step, and blob SHA-256 interpreted as 32 unsigned bytes. Target pixels and
target metrics may not choose it. If no frozen lattice produces a complete SSP2F blob within
`Q_i`, the required uniform arm has a valid fixed-method infeasibility; it is not removed from the
control set.

All source arms run in fresh target-denied workers and are sealed before target copies may be
released to or opened by any RDO worker. Independent step-3 infeasibility under `Q_i` is a valid
fixed-method failure; another start, step, or lattice cannot rescue the cell.

### Guaranteed facts and comparison semantics

Both nested fine arms start from a feasible terminal coarse state and accept only strict decreases
under a cap that contains that start. Consequently

```text
D_source(nested_fine_source) <= D_source(coarse_source)
D_target(nested_fine_target_Q) <= D_target(coarse_target)
D_target(nested_fine_target_equal) <= D_target(coarse_target)
```

by construction. These CPU inequalities are implementation invariants, not evidence, promotion
statistics, or quality wins. The equal-rate arm additionally guarantees only
`bytes(nested_fine_target_equal)<=bytes(coarse_target)`; it does not guarantee a repeated-CUDA
gain.

Comparisons with `incumbent_Q`, `direct_fine_target_Q`, and the source-only arms are **common-cap**
comparisons because every arm is independently constrained by `Q_i`; differing terminal byte
counts are not called equal rate. The primary-versus-direct path control is stronger: promotion
requires `bytes(nested_fine_target_Q)<=bytes(direct_fine_target_Q)` in every cell in addition to
its frozen conservative quality gate. The sole alphabet-release attribution is
`nested_fine_target_equal` versus `coarse_target`, where the former is constrained by the latter's
exact terminal byte count. The source-only envelope tests the operational value of legal target
guidance under a common budget; it does not isolate a same-byte objective effect.

## Coordinate search and terminal certificate

For each local sweep, visit all 8,192 rows in original order and channels `R,G,B`. For each scalar:

1. enumerate every legal local neighbor in ascending value order;
2. score **every** label with the nonmutating incremental objective proposal before invoking the
   byte oracle, persist its `float.hex()` value, numerical margin/bound, and one of
   `definitely_improving`, `ambiguous_improvement`, or `definitely_nonimproving`;
3. transcript every definitely nonimproving label as
   `not_strictly_improving_unpriced` and never invoke exact rate for it;
4. group the definitely or ambiguously improving labels by exact objective `float.hex()` equality,
   order levels by numeric objective ascending, and order members of a level by replacement value;
5. exact-price every member of the best remaining level in replacement order, requiring the exact
   component schema; if the whole level is infeasible continue to the next objective level;
6. stop pricing after the first level containing any feasible candidate, because that objective
   level dominates every later level; transcript the unqueried later labels as
   `dominated_by_feasible_better_objective_unpriced`;
7. if the first feasible level is ambiguous, record its feasible ambiguity and make no commit; if
   it is definitely improving, choose lowest complete bytes and then replacement value among its
   feasible members and send its paired objective/rate proposals to the commit coordinator.

The common coordinate error bound is mandatory for step 6: objective levels are nonnegative and
their conservative lower bounds preserve numeric objective order. Therefore, after a feasible
ambiguous level, every later level is also noncommittable. Preflight must prove this ordering from
the frozen epsilon formula; a per-proposal bound that can reorder lower bounds disables this
dominance rule and is invalid rather than silently pruned.

There is no candidate-blob hash tie: replacement value is unique within a coordinate and is the
final tie break. Ordinary blob materialization occurs only at the frozen checkpoints. The core
advances its state only after one valid paired receipt; the byte cap may never be crossed even
transiently.

In a full-label audit, visit coordinates in the same order and score all other labels in the active
alphabet. At the first coordinate with a feasible definitely improving winner, commit it, abort the
audit, perform the full `audit_jump` reconciliation, and resume local sweeps. Feasible ambiguous
levels do not commit and do not terminate traversal. Only a complete scan followed by a passing
terminal reconciliation can return `core_terminal`.

### Bounded transcript

Feed every scored label disposition, including the accepted winner, in canonical traversal order
into a domain-separated, length-prefixed rolling SHA-256 record containing:

```text
pass_index, pass_kind, epoch, origin_state_digest,
row, channel, old_value, new_value,
objective_float_hex, margin_float_hex, margin_error_float_hex, objective_class,
rate_query_class, complete_bytes_or_null, component_record_sha256_or_null,
feasibility_class, disposition
```

The preflight-frozen transcript chunk capacity bounds only the in-memory canonical-record buffer.
When full, hash and discard that buffer, then feed `(chunk_index,record_count,chunk_sha256)` into a
second rolling accumulator. Persist only aggregate record/hash counts, the aggregate chunk
hash/count, per-pass and global disposition counts, and the single closest objective and byte
margin with its coordinate identity. Do not persist raw rejected records, a list of per-chunk
hashes, a visited-state list, or any other object whose storage grows with rejected proposals.
Accepted-edit records are bounded by the frozen accepted-edit cap and contain the paired
pre/post-epoch and state digests, objective proposal identity, exact component bytes, and commit
receipt; checkpoint records, not individual accepts, carry ordinary blob hashes. Replay must
regenerate every rolling hash, count, summary, accepted record, and checkpoint exactly.

### Paired objective/byte coordinator

One adapter owns both incremental engines. Direct objective or byte commits are forbidden. A
candidate is committable only when its objective proposal and exact-price proposal have identical
origin epoch/digest and candidate digest. The coordinator issues one single-use paired token,
prevalidates both payloads without mutation, consumes both proposals inside one commit boundary,
and returns the two engines' pre/post epochs and digests plus objective and exact component bytes.
The search core mutates its own state only after that receipt proves both engines advanced exactly
once to the same candidate.

Any stale response, token reuse, exception with unknown mutation state, one-sided or partial
mutation, missing receipt, post-commit mismatch, engine/core epoch or digest divergence,
objective/rate state disagreement, or process death during the boundary permanently poisons the
coordinator. Poison is process-terminal: all later price, commit, materialize, reconcile, resume,
and terminal calls fail; the lifecycle records `INVALID_NO_DECISION`. There is no rollback,
best-so-far recovery, reconstruction from one engine, or resume across an incomplete paired commit.
The journal must make an unmatched commit-start durable so captured replay also detects this
split-brain state.

A deterministic frozen sweep/query cap exceeded is valid search nonconvergence. A completed
terminal exceeding a frozen encoder wall/RSS/VRAM gate is a valid operational failure. A cycle,
invalid proposal, external kill, OOM, scheduler interruption, host-contention-invalid measurement,
poisoned coordinator, or failure to complete the terminal full-label audit/reconciliation is
invalid/no-decision. A core terminal record has status `core_terminal`, always carries
`promotable=false`, and certifies only epsilon-bounded coordinate terminality for the frozen CPU
objective, scalar alphabet, order, cap, and numerical rule. It is one lifecycle input; only the
final reducer may decide development promotion after every arm, CUDA, resource, provenance, and
replay gate passes.

## Persisted-CUDA transfer and exact quality contrasts

### Render and metric schedule

After all target terminals are immutable, score exactly these arms in order:

```text
incumbent_Q
uniform_source
coarse_source
nested_fine_source
direct_fine_source
coarse_target
nested_fine_target_Q
direct_fine_target_Q
nested_fine_target_equal
```

Render each arm exactly three times through the copied hash/ABI-bound CUDA binary:

```text
repetition 0: arms in the order above
repetition 1: arms in reverse order
repetition 2: arms in the order above
```

Each render starts in a fresh worker from an independently cold-decoded complete blob and newly
constructed float32 boundary tensors; no numerator, denominator, field tensor, or renderer
workspace is shared. Construct
`GaussianField(means,decoded_log_scales,rotations,colors,opacities=None)`, then call the unchanged
production adapter with `field.conics(aa_dilation)`,
`field.radii(sigma_cutoff,aa_dilation)`, `field.effective_scales(aa_dilation)`, header
`render_chunk`, opacity absence, `support_fade=false`, header `sigma_cutoff`, and `mode="cuda"`.
Synchronize immediately before and after rendering. Clamp the finite float32 HWC result to `[0,1]`
on its device, synchronize, copy to a new C-contiguous CPU float32 tensor, then form
`C=NCHW[1,3,H,W]` without another normalization.

Construct `T` from the previously defined `T_hwc_f32` as
`T_hwc_f32.permute(2,0,1).unsqueeze(0).contiguous()`. For each
`(cell,arm,repetition)`, one fresh single-thread deterministic CPU metric worker executes PSNR,
MS-SSIM, and LPIPS once in that order. It uses the exact COMP-011 calls and state:

```python
delta = C.to(torch.float64) - T.to(torch.float64)
mse = delta.square().sum(dtype=torch.float64) / (3 * H * W)
psnr = 10.0 * math.log10(1.0 / float(mse))

ms_ssim = pytorch_msssim.ms_ssim(
    C, T, data_range=1.0, size_average=True, win_size=11, win_sigma=1.5,
    win=None, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
    K=(0.01, 0.03),
).reshape(()).item()

lpips_model = lpips.LPIPS(
    pretrained=True, net="alex", version="0.1", lpips=True, spatial=False,
    pnet_rand=False, pnet_tune=False, use_dropout=True, model_path=None,
    eval_mode=True, verbose=False,
).to(device="cpu", dtype=torch.float32).eval().requires_grad_(False)
lpips_value = lpips_model(
    2.0 * C - 1.0, 2.0 * T - 1.0,
    retPerLayer=False, normalize=False,
).reshape(()).item()
```

Require finite strictly positive MSE and finite metrics. Bind Python/Torch/Pillow, MS-SSIM source
and explicit weights, LPIPS source/version, and the canonical LPIPS state-dict hash. The worker
sets both Torch thread counts to one, deterministic algorithms true, manual seed zero, inference
mode, and
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`. Preflight must prove
repeated-call scalar IEEE-byte identity on identical inputs.

For every arm, all three max-minus-min spreads must pass:

```text
PSNR <= 0.001 dB
MS-SSIM <= 0.00001
LPIPS <= 0.00001
```

Exceeding a spread cap is invalid/no-decision. CUDA atomic renders are not expected to be
bit-identical. Captured replay may differ from the original minimum or maximum endpoint by at most
`0.001 dB`, `0.00001`, and `0.00001`, respectively, for PSNR, MS-SSIM, and LPIPS, while
independently passing all spread caps and reproducing every decision class.

### Conservative comparisons

For candidate arm `A` and comparator `B`, using their three metric repetitions, define:

```text
psnr_gain(A,B)       = min(A_psnr)    - max(B_psnr)
ms_ssim_loss(A,B)    = max(B_msssim)  - min(A_msssim)
lpips_increase(A,B)  = max(A_lpips)   - min(B_lpips)
```

Higher PSNR gain is better; lower MS-SSIM loss and LPIPS increase are better. Persist every scalar,
endpoint, responsible repetition, spread, and conservative contrast.

The source-only control set is frozen as:

```text
incumbent_Q, uniform_source, coarse_source,
nested_fine_source, direct_fine_source
```

It is a post-scoring oracle envelope, not a selectable encoder. Define its hardest PSNR endpoint as
the maximum `max(arm_psnr)` over that exact set; exact endpoint ties choose lower complete bytes,
then the listed arm order, then unsigned complete blob bytes. Thus

```text
psnr_gain_vs_source = min(nested_fine_target_Q_psnr)
                      - max_over_source_arms_and_repetitions(source_psnr)
```

Target values may construct this conservative comparator only after all source streams are sealed;
they never select or modify a source stream.

For a minimum-gain threshold `G` and exclusion half-width `b`, a PSNR contrast definitely passes
iff `gain>=G+b`, definitely fails iff `gain<G-b`, and is ambiguous otherwise. For an upper loss
threshold `L` and half-width `b`, a guard definitely passes iff `loss<=L-b`, definitely fails iff
`loss>L+b`, and is ambiguous otherwise. No ambiguous decision-relevant cell can promote or be
coerced to one side.

The numerical `G`, `L`, half-widths, maximum epsilon-terminal ambiguity, image win/effect-size
requirements, worst-case rule, and image-bootstrap matrix/statistic remain calibration-dependent
freeze blockers. The two already-exposed calibration cells above may test mechanics but may not
alone set these values. They must be inserted as exact values and predicates before development
access. At minimum, promotion must require separately for both tuples:

- every required terminal is canonical SSP2F and respects its exact cap;
- every required search has a `promotable=false` `core_terminal` with a complete full-label
  traversal, passing objective/ordinary-stream reconciliation, and terminal ambiguity no greater
  than the separately frozen epsilon-terminal gate;
- primary `nested_fine_target_Q` definitely passes the conservative PSNR-gain and MS-SSIM/LPIPS
  guards versus `incumbent_Q`;
- the primary definitely passes PSNR versus `direct_fine_target_Q` and has complete bytes no
  greater than that direct arm in every cell;
- `nested_fine_target_equal` has bytes no greater than `coarse_target` and definitely passes its
  conservative PSNR and perceptual guards versus that coarse comparator;
- `psnr_gain_vs_source` definitely passes the frozen common-cap source-envelope requirement;
- every encoder and decode resource gate passes; and
- image-level paired inference passes without treating tuples, edits, or repeats as independent.

The direct and source-envelope comparisons are common-budget evidence except for the explicit
primary-byte dominance condition; only the coarse/equal contrast supports alphabet-release
attribution. A common-cap quality win is not a compression win. Any compression wording
additionally requires a separately frozen complete-byte geometric-mean, win-count, worst-case, and
image-bootstrap gate.

## Resource accounting

Encoder timing begins before footprint/design construction and includes proposal construction,
every exact byte query, every objective evaluation, all sweeps, every initial/sweep/audit-jump/
terminal full reconciliation, terminal certification, final full encode, parse, cold decode, and
canonical re-encode. Record wall/CPU time, peak RSS, peak VRAM, labels objective-scored, labels
rate-priced, rate queries avoided by nonimprovement and dominance separately, accepted edits,
paired commits, reconciliations, full renders, and ordinary encodes. Only exceeding the frozen
deterministic sweep/query cap is search nonconvergence. An external kill, OOM, scheduler
interruption, poison event, or invalid resource measurement is invalid/no-decision; it is never
relabeled as a converged or operational result.

The exposed native `size_repeated` benchmark is approximately `0.28--0.33 ms` per 8,192-symbol
query. A two-neighbor coarse sweep scores roughly 49,000 labels and a four-neighbor fine sweep
roughly 98,000, but the repaired objective-first algorithm does **not** issue one rate query per
label. The earlier `13.8--16.2 s` coarse-sweep, `29--34 min` fine-audit, `10--11.5 min`
coarse-audit, and `44--52 h` serialized-grid arithmetic figures apply only to the hypothetical
unpruned case in which every scored label reaches exact rate; they are not lower bounds for this
protocol. Exact-price counts are data-dependent and must be measured during synthetic/already
exposed calibration before caps freeze. Model-zlib work, incremental objective scoring, checkpoint
full renders/encodes, coordinator overhead, and replay remain additional costs. Failure to fit
frozen resource limits requires implementation optimization or a newly named protocol before data
access; it does not permit weakening the terminal audit after seeing COMP-012 cells.

Measure the primary SSP2F and exact `Q_i` with the contemporaneous bytes-to-boundary protocol.
Import modules, load verified native assets, read the artifact-relative blob into immutable bytes,
and verify its prebound hash before timing. Pin one thread to the lowest available CPU. Run one
untimed fresh-worker warm-up in `Q_i` then primary order, followed by nine measured fresh
subprocesses per blob. For frozen image index `i=0..7`, tuple index `t=0,1`, and repetition
`r=0..8`, run the primary first iff `(i+t+r) mod 2 == 0`; otherwise run `Q_i` first. Complete both
arms before advancing `r`.

The timer begins immediately before strict complete-container parse and ends after exact
decoded-boundary tensor construction. It includes CRC/model/zlib/arithmetic decode and mandatory
canonical re-encode, but excludes imports, blob I/O/hash verification, target access, rendering,
and metrics. For each arm sort `(nanoseconds,repetition_id)` ascending and use zero-based index
`4`; exact nanosecond ties use lower repetition ID. Peak RSS is the maximum measured child
`ru_maxrss*1024`, with lower repetition ID only as a provenance tie. Sort the eight exact rational
time ratios `primary_ns/Q_ns` by integer cross-product with frozen image index ties; the
across-image upper median is zero-based index `4`. All denominators must be positive. Persist every
launch order, PID, affinity, environment, format/hash, raw sample, selected median/RSS, rational
order, and predicate. Candidate/Q decode gates are provisionally inherited:

```text
upper image-median time ratio <= 1.5
every image time ratio <= 2.0
every image peak-RSS ratio <= 1.5
```

Rendering is outside decode timing. Absolute encoder wall/RSS/VRAM caps and the retained relative
decode gates remain blockers until the exact oracle/search implementation is calibrated on
synthetic or already exposed data.

## Lifecycle and replay

Legal stages are:

```text
preflight-and-roster
acquire-development
prepare-development
fit-and-seal-base-streams          # target-authorized producer
reproduce-incumbents-and-Q
run-source-arms                    # target-denied workers
release-target-copies-to-rdo-workers
run-target-arms
cuda-quality
resource-benchmark
analyze
captured-source-replay
```

Preflight runs in an empty output directory and binds the task, source closure/archive, remote
object and central directory, roster/exclusions, base-field config, native size coder, SSP2F
grammar and exact component schema, objective arithmetic/block/reconciliation configuration,
paired-coordinator implementation and poison proofs, bounded-transcript configuration, persisted
renderer, metric packages/weights, compiler/ABI/CPU/GPU/driver/thread environment, all thresholds,
bootstrap, resource limits, and synthetic proofs.

As in COMP-011, preflight must copy the actually loaded CUDA extension into the artifact and bind
its relative path, size/SHA-256, ELF/build/dynamic identity, complete loader dependencies,
compiler/source command and hashes, CUDA toolkit/runtime, Torch ABI, driver, GPU name/UUID/compute
capability, required system `libstdc++` preload, and thread environment. Every later worker and
captured replay loads only this verified copy; cache rediscovery or rebuilding is forbidden.

Every stage is append-only, journaled, content-addressed, and enabled only by the exact predecessor
seal. Base fitting is target-authorized, but no source-arm RDO worker receives target authority.
`core_terminal` never advances the lifecycle by itself. Target copies are released to target-guided
RDO workers only after the source-stage reducer verifies every required source arm's unpoisoned
paired coordinator, epsilon-terminal certificate, bounded transcript, checkpoint reconciliation,
resource-valid completion, and immutable stage seal. Confirmation access is impossible in every
stage.

Captured replay extracts the source archive to a randomized root, verifies the sealed base-stream
authorities without refitting, and must reproduce exact incumbents, `Q_i`, source-arm terminals in
target-denied workers before releasing target copies to RDO workers, target-arm terminal blobs
byte-for-byte, all transcript aggregates, paired receipts, oracle/full-encode/reconciliation
checks, CUDA classes within frozen tolerances, resource booleans, and the final decision. A search
core that emits `promotable=true`, resumes after poison, or bypasses a paired receipt is invalid.
Reuse the COMP-011 fresh-root plan/broker/reducer architecture. Original analysis must reject any
decision-relevant metric exclusion-band or terminal-epsilon ambiguity before it can authorize a
promotable replay; replay independently fails if an endpoint enters a band or a class/decision
changes. Failed attempts remain immutable invalid/no-decisions.

## Decisions and no-rescue rule

No decision branch below is executable while this task remains `NO_GO_PRE_DATA`.

```text
both tuples pass every validity, primary, direct-fine, source-control,
equal-rate, CUDA-transfer, terminal-convergence, and resource gate
    -> AUTHORIZE_SEPARATELY_BOUND_COMP012_CONFIRMATION
otherwise, for a valid fixed-method failure
    -> ABANDON_FIXED_COMP012_EXACT_BYTE_RGB_RDO_V1
any source/provenance/parity/replay failure
    -> INVALID_NO_DECISION
```

COMP-012 never opens confirmation. Development failure cannot change the archive, roster, field
producer, bit tuples, incumbent set, lattice menu, alphabet, neighborhood, receiver set, start,
objective, cap, proposal order, tie rule, metric, threshold, resource limit, or primary arm.

## Remaining freeze blockers

- Source-bound integration of the exact SSP2F size oracle with the ordinary encoder/cold decoder
  and adversarial proof of the exact component schema on baseline, proposal, commit, sweep,
  audit-jump, and terminal states.
- One paired objective/byte adapter plus hostile tests for atomic token consumption, stale
  evidence, one-sided mutation, incomplete-journal commit, permanent poison, and split-brain
  invalid/no-decision behavior.
- Repair of the coordinate core to objective-score before rate, price exact objective levels only
  until feasible, retain no raw rejection/chunk/visited-state lists, materialize only at
  initial/sweep/audit-jump/terminal checkpoints, and emit only `promotable=false`
  `core_terminal`.
- Performance calibration and freeze of the provisional lattice menu, deterministic
  sweep/objective/rate/accept/reconciliation caps, bounded transcript chunk capacity, and all
  required arms using only synthetic or already exposed data.
- Source-bound integration of the block-reduced incremental CPU objective, ambiguity classifier,
  and independent full reconciliation, plus frozen synthetic and target-denied per-cell CPU/CUDA
  parity tolerances.
- Exact maximum epsilon-terminal ambiguity; primary/direct/source/equal-rate PSNR effect sizes and
  exclusion bands; image-win and worst-case/bootstrap predicates; and MS-SSIM/LPIPS guard
  thresholds/bands. The two exposed calibration cells above do not freeze them.
- Absolute encoder wall/RSS/VRAM limits and final decode-resource gates.
- Canonical resolved base-producer configuration record/hash and complete source binding.
- Captured-source lifecycle implementation, hostile review, and successful empty-directory
  preflight.

Until every blocker is removed and the task is rehashed, no COMP-012 member payload may be opened.

## Interfaces allowed

New COMP-012 acquisition/benchmark modules, exact SSP2F oracle helpers, tests, this task,
research documentation, ignored result evidence, and ARA records only. Do not edit production
codec, fitter, renderer, CLI, defaults, prior frozen scientific sources/tasks/artifacts, or any
confirmation material.
