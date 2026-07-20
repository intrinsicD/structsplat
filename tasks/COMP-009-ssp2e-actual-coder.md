# COMP-009: Exact SSP2E actual-coder development test

## Status

**Protocol/implementation v1 frozen after the audited COMP-008 oracle survival, 151 passing
pre-data proofs, and an independent pre-data GO, before any SSP2E, SSP2L, SSP2F, or SSP2S stream
is encoded from a development field.** The source-bound preflight is pending. Until that preflight
succeeds, no command may open a COMP-008 `stream.sspl1`, `absolute_symbols.npz`, or field artifact.
Aggregate COMP-008 results and ordinary artifact metadata were visible while this task was
designed; no actual-coder rate was observed.

COMP-008 left the table construction, bit-level arithmetic code, model fitter, numeric stream IDs,
factorized comparator, and shuffled control intentionally revisable after oracle survival. This
task resolves those ambiguities without changing COMP-008's candidate syntax or lower-bound
decision.

If either bit tuple fails any frozen development gate, the decision is **abandon fixed SSP2E v1
jointly-required candidate**. Do not tune its CDF table, grid, head, fitter, starts, arithmetic
coder, controls, tuple, or thresholds. A two-tuple pass authorizes only a separately source-bound
confirmation task; COMP-009 never opens confirmation.

## Question and claim boundary

On the sixteen existing COMP-008 development SSPL1 streams only, can the exact fixed
mean-conditioned SSP2E v1 format realize at least a 10% complete-stream saving against the best of
SSPL1 and two complete factorized arithmetic formats, while showing causal spatial attribution
and bounded decode resources?

The candidate is reconstruction-invariant. It must reproduce all eight ordered absolute symbol
arrays and the decoded-field boundary state exactly. Thus COMP-009 can establish actual
compression, model-fitting convergence, and bounded decode resource evidence only. It cannot
improve or support claims about image quality, ordinary field-fitting convergence, renderer speed,
allocation, robustness, or mathematical expressiveness.

## Frozen inputs

Use only `results/comp008_sgi_entropy_oracle_dev_v3_2026-07-16`, with binding
`9c5dca682d490567a56ce4fc4043112a5bacf72b2fee908e95e86e691a1ea0f2`, analysis seal
`3368a3f96926b0eaf2e18119ae587d4bb3822c450c0f6f5a33e883c4e87e6792`, and replay seal
`95829d0c563842adcc0a7788d7ff02a8e6bd74596f80a3a3966d78c2da9b1397`.
Also require the independently audited `cells.jsonl` file SHA-256
`9f14774f091afb0b8c2e9c471f9e9bd21f2d8800f44da6e413a076acf33ccb8b` and
`analysis_record.json` seal
`1081a77e7d5d58c04ef8695e0b45d0508b6e0835e5db597094fc1d61597f0c56`
(file SHA-256 `c465bff62384d1219098076b11a9f571fb84352030b1b29307c1525d1352464a`),
including its ordered sixteen cell seals.

The eight development images and two tuples `(12,6,6,8)` and `(16,8,8,8)` are unchanged. After
preflight, parse each persisted `stream.sspl1` afresh and verify its byte and absolute-symbol hashes
against the sealed COMP-008 cell record. `absolute_symbols.npz` is an integrity comparator, not an
input authority. Do not fit a field, decode a PNG, render a CLIC target, alter QAT, or access any
confirmation payload.

## Static discretized-logistic table

Freeze `cdf_table_id=1`, total frequency `T=32768`, and alphabets `A in {64,256}`. For location
`loc=0..63`, scale index `j=0..15`, and exact Q8 scale list

```text
sigma_q8 = [64,93,134,194,281,406,588,851,
            1232,1783,2580,3734,5405,7822,11321,16384]
mu = loc * (A-1) / 63
sigma = sigma_q8[j] / 256
```

generate with Python `Decimal`, precision `160`, `ROUND_HALF_EVEN`:

```text
boundary[0] = 0
boundary[k] = 1/(1+exp(-(k-0.5-mu)/sigma)), k=1..A-1
boundary[A] = 1
mass[s] = boundary[s+1] - boundary[s]
R = 32768-A
quota[s] = R*mass[s]
alloc[s] = floor(quota[s])
```

Assign the remaining `R-sum(alloc)` units by descending fractional remainder, ties by ascending
symbol, then set `frequency[s]=1+alloc[s]`. Every frequency is positive and sums to 32768; the
CDF begins at zero, ends at 32768, and is strictly increasing.

Serialize one universal asset as:

```text
<8sIII> = (b"SSP2ECDF", version=1, T=32768, table_count=2)
for A in (64,256):
    <HH> = (A, reserved=0)
    uint16_le frequencies in loc-major, scale-major, symbol-major order
```

The frozen asset is exactly `655388` bytes with SHA-256
`89534aa31405e5129fc57727076dfc32829deb8090362e43869110db2a45c493`.
It is universal decoder source state, bound before development import and never transmitted per
image.

## Exact Q24 fitting cost

Generate an encoder-only integer cost asset with the same Decimal context:

```text
cost[f] = round_half_even(2^24 * log2(32768/f)), f=1..32768
<8sII> = (b"SSP2ECST", version=1, Q=2^24)
uint32_le cost[1],...,cost[32768]
```

Its frozen SHA-256 is
`b67d48caf7c35a79a78663fba83ce63d1df5d892efaf43c921d056d8fdf65c7e`.
Only this integer objective may choose a fitted model; actual stream bytes choose among converged
starts as specified below.

## Grid and head

Grid byte order is cell `0..255`; feature bit `j` is least-significant-bit first:

```text
phi[j] = 2*((grid[cell] >> j) & 1)-1
```

The five channel records are serialized in order `scale_x,scale_y,R,G,B`, each as `<8bh8bh>`:
eight `int8` location weights, one `int16` location bias, eight `int8` scale weights, and one
`int16` scale bias. Signed `int32` inference is exactly

```text
loc   = clamp(b_mu + sum_j w_mu[j]*phi[j], 0, 63)
scale = clamp(b_s  + sum_j w_s[j] *phi[j], 0, 15)
```

There is no rounding, shift, wraparound, hidden scale, channel transform, or learned offset.

## Deterministic model fitting

Aggregate exact ordered symbols into per-context histograms and minimize

```text
J = sum counts[cell,symbol] * cost[frequency(cell,symbol)].
```

Use exactly eight starts, with IDs `0..7`:

- ID 0: `grid[c]=c`;
- ID 1: `grid[c]=c XOR (c>>1)`;
- ID 2: `grid[c]=bitreverse8(c)`;
- IDs 3--7: the first digest byte of
  `SHA256(b"structsplat-comp009-grid-start-v1\0" || uint8(start_id) || uint16_le(cell))`.

For every start, initialize all weights to zero. Exhaustively choose the global `(loc,scale)` bias
for each channel, minimizing `J` with ties by ascending pair. Then repeat:

1. sweep channels in frozen order and parameters in serialized order;
2. for each `int8` weight test all values `-128..127`, accept only a strict objective decrease,
   and retain the current value on ties;
3. for each bias, form the complete distinct set
   `{-32768,32767,current} union {k-d_c | c=0..255,k=0..L}`, clip to `int16`, test in ascending
   order, accept only a strict decrease, and retain current on ties;
4. sweep cells `0..255`, test grid bytes `0..255` in ascending order, and accept only a strict
   decrease;
5. stop only when a complete head-plus-grid sweep has no strict improvement.

Here `d_c` is the current weight dot product and `L` is 63 for location or 15 for scale. Every
start must reach a fixed point within 64 outer sweeps or the cell is invalid/no-decision.

Encode all eight converged starts and choose by: (1) smallest sum of five arithmetic payload
lengths, (2) smallest `J`, then (3) lexicographically smallest `grid_raw || head_raw`. Persist each
start's trajectory, terminal state, payload lengths, and selection reason.

## Arithmetic coder

Use MSB-first bits and exact 32-bit inclusive state:

```text
FULL=2^32; MASK=FULL-1; HALF=2^31; Q1=2^30; Q3=3*2^30
low=0; high=MASK; pending=0
range=high-low+1                 # uint64 intermediate
unit=range//32768; require unit>=1
child_low=low+unit*cumulative
child_high=child_low+unit*frequency-1
```

The unused tail `unit*T..range-1` is never assigned. After every symbol apply E1/E2/E3 repeatedly:

```text
if high < HALF:
    emit 0 then pending 1 bits
elif low >= HALF:
    emit 1 then pending 0 bits; low-=HALF; high-=HALF
elif low >= Q1 and high < Q3:
    pending+=1; low-=Q1; high-=Q1
else:
    stop
low=(low<<1)&MASK; high=((high<<1)&MASK)|1
```

Canonical termination increments `pending`; if `low<Q1` emit zero followed by `pending` ones,
otherwise emit one followed by `pending` zeros; pad only with zero bits to the next byte.

The decoder initializes `code` from the first 32 MSB-first bits, supplies zeros only after physical
EOF, computes `scaled=(code-low)//unit`, rejects `scaled>=32768`, locates the unique positive CDF
interval, and mirrors E1/E2/E3. It decodes exactly `N=8192` symbols per modeled stream, rejects
nonzero padding, illegal rows, malformed termination, suffixes, and wrong counts, then re-encodes
and requires byte-identical equality. A raw payload mutation can legitimately be another canonical
codeword for a different symbol sequence; raw arithmetic canonicality is not an integrity code.
The complete container CRC rejects payload corruption, while frozen table/model identity prevents
silent decoder-model substitution.

For every stream enforce exactly

```text
2^(8*payload_bytes) * product(actual_symbol_frequencies) >= 32768^8192
```

and persist bit lengths and canonical integer hashes for both sides.

Before development import, require reduced-width exhaustive tests over all positive compositions
for alphabets two and three and sequences through length six; production-width representative
uniform/minimum/skew tests; Python/native differential tests; truncation/suffix/padding/wrong-count/
noncanonical rejection; representative raw mutations that are malformed; explicit documentation
of mutations that form a valid alternate codeword; complete-container single-bit CRC rejection;
and a brute-force reduced-width proof that the zero-padded terminator tag lies inside the final
interval. Any failure stops pre-data and requires a new named protocol.

## SSP2E v1 container

The COMP-008 outer/header/directory formats remain `<5sBBBQI>`, `<III4B4i12f5I>`, and `<IQQ>`.
Numeric IDs `1..9` are, in order:

```text
means_zlib, scale_x_arith, scale_y_arith, rotation_zlib,
R_arith, G_arith, B_arith, grid_raw, head_raw
```

Offsets begin at byte 300 and payloads are contiguous in that order. Grid/head lengths are exactly
256/100 bytes. Complete fixed overhead remains 656 bytes. Magic is `SSP2E`, version 1, flags 0,
stream count 9. Validate total length, CRC-32 over every byte after the outer record, directory
order, IDs, offsets, gaps, overlaps, aliases, suffixes, nonempty arithmetic streams, byte-identical
SSPL1 mean/rotation payloads, exact header semantics, and `cdf_table_id=1`.

The five trailing uint32 header fields are exactly `render_chunk`, `renderer_id=1` for the sole
accepted normalized CUDA renderer, `support_fade=0`, `cdf_table_id=1`, and `zlib_level=9`.

Fresh decoding must reproduce every ordered absolute symbol, the symbol hash, and decoded-field
boundary state. The Python proof arithmetic coder and native C++17 arithmetic core must emit and
decode byte-identical arithmetic payloads. The measured bytes-to-boundary implementation uses the
native arithmetic core inside the same strict Python container parsing, head/table inference, and
tensor-construction path used for both candidate and controls; this is not a claim that the entire
container decoder is native C++.

## Complete factorized controls

Define `F=min(SSP2L,SSP2F)` and the primary baseline
`P=min(SSPL1,F)` independently per image. Resolve exact complete-byte ties in the fixed order
`SSPL1`, then `SSP2L`, then `SSP2F`; the chosen stream supplies both the primary bytes and the
resource baseline. Each alternative is a complete self-identifying stream;
no selector or free mode byte is assumed.

Both factorized containers use a 20-byte outer, the same 100-byte header, eight 20-byte directory
entries, and therefore 280 pre-payload bytes. IDs are `1..7,10` for means, scale_x, scale_y,
rotation, R, G, B, model.

- **SSP2L:** magic `SSP2L`; same static table/coder; exhaustively choose one global `(loc,scale)`
  per channel by the exact Q24 objective; model payload is ten bytes `(loc:uint8,scale:uint8)` in
  channel order.
- **SSP2F:** magic `SSP2F`; form positive empirical frequencies with `R=32768-A`, floor
  `R*count/8192`, distribute the remainder by descending integer remainder and ascending symbol;
  concatenate five full `uint16_le` frequency arrays and store them as one exact canonical zlib-9
  member. Reject zlib suffixes and require local decompress/recompress equality.

Both copy the same SSPL1 mean/rotation payloads and use the same five independent arithmetic
streams. Both must independently cold-decode to exact symbols/state.

## Shuffled-position causal control

SSP2S has distinct magic `SSP2S`; every other syntax, table, fitter, start, selection, and byte rule
equals SSP2E. Generate

```text
key[i]=SHA256(b"structsplat-comp009-shuffle-v1\0" || uint32_le(i))
pi=indices 0..8191 sorted by (key[i],i)
shuffled_context[i]=true_mean_cell[pi[i]]
```

The serialized `uint16_le[8192]` permutation has SHA-256
`63691f3fdda3cbf83d1bc7b954785dc5467b11433bc4fbbe84fd5ec2d94c05f2`.
The decoder derives it statically after means are decoded. Attributes remain in original row order.
Row-index dependence exists only in this negative control, never in SSP2E.

## Frozen gates

Apply independently to both tuples. Let `E_i` be complete SSP2E bytes,
`P_i=min(SSPL1,SSP2L,SSP2F)`, and let `M_i`/`U_i` be `656` plus the five arithmetic payloads for
SSP2E/SSP2S.

Actual-rate gates are conjunctive:

1. exact geometric mean `E/P <=0.90`;
2. at least `7/8` strict `E<P` wins;
3. every `E/P<=1.02`;
4. the reused COMP-008 100,000-replicate bootstrap order statistic is strictly `<0.95`.

Every decision comparison uses exact integer products and the unchanged bootstrap matrix/hash.

Causal attribution is conjunctive:

1. `10*sum(M_i) <= 9*sum(U_i)`;
2. at least `6/8` strict `M_i<U_i` wins.

Resource protocol uses the native arithmetic core only after exact Python parity. Read/import before timing;
pin one thread to the lowest available CPU; run one untimed warm-up and nine measured fresh
subprocesses per blob. For image index `i=0..7` in the frozen image order, tuple index `t=0..1`
in the frozen tuple order, and measured repetition `r=0..8`, run the candidate first iff
`(i+t+r) mod 2 == 0`, otherwise run the selected primary first. Complete both arms of repetition
`r` before starting `r+1`; the untimed warm-up is primary then candidate. Persist this exact
18-measured-launch schedule (and both warm-ups) per cell. Time
container parse, model setup, zlib/arithmetic decode, and exact decoded-boundary tensor
construction; use the fifth sorted nanosecond value and Linux child peak RSS. This bytes-to-boundary
gate isolates work changed by the codec. For each arm use the maximum of its nine measured child
peak-RSS values; warm-up RSS is diagnostic only. Persist nongating combined wall diagnostics for
each eight-start fit plus its arithmetic-length selection encodes, factorized control-model
selection, and final complete encode plus dual Python/native validation decode; do not relabel
these combined scopes as pure fit or pure encode speed. After resource sampling, decode and run the
selected primary once and then SSP2E once through the unchanged renderer as a parity and
render-only timing diagnostic, never as part of the codec timing gate. Record finite-output and
`rtol=atol=5e-4` parity diagnostics without invalidating an otherwise exact cell on a finite CUDA
atomic-order delta; decoded symbol/boundary equality remains the validity authority.

Resource gates are the across-image **upper median** time ratio (fifth value, zero-based index 4,
after exact ratio sorting) `<=1.5`, worst-image time ratio `<=2.0`, and per-image peak RSS ratio
`<=1.5`. Those combined execution times, restart spread, and sweep count are mandatory
diagnostics.

Any gate failure in either tuple yields `ABANDON_FIXED_SSP2E_V1`. A two-tuple pass yields
`AUTHORIZE_SEPARATELY_BOUND_SSP2E_CONFIRMATION`. Report actual-rate, attribution, and resource
strata separately even though promotion is conjunctive.

## Lifecycle and replay

Stages are `preflight`, `import-dev`, `run-dev`, `benchmark-dev`, `analyze`, and `replay`.
Preflight in an empty output directory binds task, implementations, tests, native source/binary,
the COMP-008 source/provenance surface, complete execution-source-closure snapshot, environment, compiler/
ABI/CPU/affinity/thread state, CDF/cost assets, shuffle permutation, bootstrap matrix, and every
synthetic/exhaustive proof result. The frozen host requires
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` so the CUDA renderer extension resolves the
system `CXXABI_1.3.15`; bind and reuse that loader state in every stage. Only then may `import-dev`
open the sixteen SSPL1 blobs. Also bind and reuse
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1` in every stage.

`run-dev` is append-only/resumable by image/tuple and persists every start, chosen model, complete
blob, symbols, state hashes, coder certificates, and combined execution timing. `benchmark-dev`
persists the fresh-process decode timing and RSS. Replay from captured source must
reparse SSPL1, recompute both controls, rerun the fitter, reproduce all model and complete-stream
bytes, fresh-decode, and recompute exact gates/decision without accessing pixels or refitting the
underlying Gaussian field/QAT state.

Fail closed on source/environment/input drift, a malformed asset, preflight proof failure,
duplicate/missing cell, noncanonical stream, parity mismatch, illegal stage order, confirmation
access, or replay drift. Invalid/no-decision is not a compression failure.

## Acceptance criteria

- [ ] CDF/cost/permutation/bootstrap assets match their frozen hashes before development import.
- [ ] Exhaustive reduced-width and Python/native arithmetic proofs pass pre-data.
- [ ] All 16 SSPL1 inputs match the audited COMP-008 byte/symbol/state records.
- [ ] SSP2E, SSP2S, SSP2L, and SSP2F are complete, canonical, independently decodable streams.
- [ ] Every decoded symbol and boundary tensor is exact; re-encoding is byte-identical.
- [ ] All eight starts converge within 64 sweeps and selection is replay-exact.
- [ ] Resource measurements use the frozen native/fresh-process protocol.
- [ ] Analysis and captured-source replay agree on all non-timing values and the decision.
- [ ] Independent quantitative/artifact review passes before any confirmation action.

## Interfaces allowed

New benchmark modules/native source, generated static assets, tests, task/research documentation,
ignored result evidence, and ARA records only. Do not edit production codec/fitter/renderer/CLI,
COMP-008 code/tests/task, its valid artifact, or any confirmation material.

## Depends on

COMP-001/002/003/004/006/007/008, BENCH-016, SGI, HAC, and the audited COMP-008 v3 development
artifact.
