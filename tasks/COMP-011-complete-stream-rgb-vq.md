# COMP-011: Complete-stream RGB VQ and residual-VQ development test

## Status

**Terminal 2026-07-17: `INVALID_NO_DECISION`; no SSP2V or SSP2L confirmation is authorized.**
The ordinary v2r35 and v2r36 development lifecycles produced descriptive diagnostics, and v2r36
replay phase A reproduced the complete candidate streams and decoded states. The full replay
cannot satisfy the frozen quality contract: in
`vita-vilcina-3055/[16,8,8,8]/ssp2f`, decision-relevant prefix arm
`flat_s2_k128_l384` has LPIPS increase `0.00101422518491745`, inside the frozen
`(0.00098,0.00102]` exclusion band. The arm still definitely fails on PSNR and MS-SSIM, but
revision v2 separately requires every decision-relevant replay metric to remain outside every
exclusion band. The captured reducer therefore correctly fails closed. Under the frozen rule that
any replay failure is invalid/no-decision, the ordinary
`ABANDON_FIXED_SSP2V_V1_UNDER_COMP011_V2` payload is not a promotable scientific decision.
Changing replay to accept aggregate arm failure after observing this value would be
outcome-conditioned retuning. Stop replay reruns, preserve the ordinary measurements as
development diagnostics only, and use a separately named preregistered task with genuinely new
data for any successor. See
`ara/evidence/comp011-v2-invalid-replay-audit-2026-07-17/run.md`.

**Historical pre-data status.** Protocol revision v2 was specified with the audited COMP-010
dependency satisfied. The adversarial pre-data audit of
revision v1 returned **NO-GO** because the primary comparator omitted exact formats, targets could
be imported before the candidate set was irreversibly sealed, several metric/resource/replay
predicates were underspecified, and the proposed attribution exceeded the controls. Revision v1
was retired before any COMP-011 development stream, target, fit, encode, decode, render, rate, or
resource observation. This v2 repair is therefore protocol clarification, not outcome-conditioned
retuning.

### 2026-07-17 post-result replay-dispatch hardening

The ordinary source-bound development lifecycle through analysis completed before this addendum.
Two subsequent source-bound attempts reproduced all 224 scientific candidate streams
byte-for-byte, all sixteen selected-Q choices, candidate classifications, selected labels, selected
bytes, and the frozen analysis decision. Neither attempt produced a valid replay seal. The first
exposed an incomplete exact system-read policy grammar; after that grammar was repaired and the
focused lifecycle suite passed, the second reached replay quality execution and exposed a
Landlock identity error: an outer rule for lexical `/proc/self/*` is bound to the outer process,
so a stacked nested rule cannot grant a renderer child access to that child's distinct
`/proc/self/maps` or `/proc/self/task/<tid>/comm`. CUDA initialization therefore failed before
the first replay render. These are lifecycle failures, not scientific outcomes, and they authorize
no change to the method or frozen decision.

The replay dispatcher is now hardened without changing any stream grammar, candidate bytes,
candidate ordering, metric, repetition/order schedule, tolerance, exclusion band, resource
schedule, gate, bootstrap, or decision:

1. A captured-source fresh-root pass verifies the persisted target-hidden phase-A authorization
   and emits one canonical sealed logical quality plan. It launches no scientific worker and
   performs no metric reduction.
2. The trusted lifecycle host verifies that plan by independently reconstructing it from the
   frozen authorities, derives every command/path/environment/allowlist itself, and launches only
   the declared render and metric jobs. Every such job is a distinct Landlock `fresh_root` with a
   null parent-sandbox digest. The host records worker outputs and attestations but performs no
   selection or other scientific reduction.
3. A second captured-source fresh-root pass verifies all quality receipts, recomputes the frozen
   endpoint tolerances, classes, and selections, and emits the only resource authorization.
   Invalid quality authorizes zero resource jobs; otherwise secondary schedules are authorized,
   and primary schedules are authorized only when all sixteen replay cells are selected.
4. The host launches exactly the authorized resource samples as fresh roots and does not summarize
   them. A final captured-source fresh-root pass verifies and summarizes those samples, reconstructs
   rows and gates, and emits the phase-B decision record. The trusted host independently recomputes
   that complete sealed record from the persisted quality and resource intermediates and requires
   byte-exact equality both before publication and during later replay verification.

All intermediate plans, authorizations, results, pass launches, and worker receipts are sealed.
Attempt artifacts are content-addressed so a failed attempt cannot conflict with a later immutable
attempt. The randomized captured source root remains common to every pass and worker. The
preflight seals the canonical `nvidia-smi` inventory and executable identity; captured planners
and reducers revalidate that binding without rerunning the GPU query or receiving GPU-device
authority. Only fresh render workers receive the exact frozen GPU-device allowlist. The
target-forbidden terminal-Lloyd branch remains a single target-free pass. No confirmation path is
authorized or accessed. A new empty output directory and complete new source-bound lifecycle are
required; failed replay attempts remain invalid/no-decision provenance and may not be resumed as
evidence.

The dependency is now satisfied by the independently audited COMP-010 v2r2 repair artifact:
preflight `a00a7a553468dbe1f2e00e1d6e5ecb38fb3227356e090c73bbf564b39ec1bc00`, repair seal
`adbf2c48e1721b6d4b74960211e08ec3c770bebde1d068ddfc88e5a83f78f3a8`, identical randomized-child
seal `b482a866742d472d02d7c80db1c86640e5e17a85fdf025a56096173c08fd7ac4`, and identical captured
worker seal `0d7d74a8e6f6a552eeb4f1d3df4a0aeb8f6e6977496400d7c65845c4fd494309`. Its post-result audit
returned GO for codec lifecycle provenance only; it explicitly excludes renderer-binary replay and
new compression evidence. The complete COMP-011 source closure, including this insertion, must now
be rebound before preflight. Until a COMP-011 preflight in a fresh empty output directory succeeds,
no command may open a COMP-008/009 development `stream.sspl1`, target PNG,
`absolute_symbols.npz`, field artifact, pixel payload, or any confirmation material.

The sixteen COMP-009 development cell identities, aggregate results, and ordinary artifact
metadata were exposed while this task was designed. No COMP-011 VQ fit, SSP2Z/SSP2V byte count,
VQ reconstruction, target-relative quality value, or COMP-011 resource measurement has been
observed. SSP2L development rates were already exposed; their separately reported survival screen
is therefore explicitly outcome-informed and cannot be called confirmation.

Protocol revision `v2` names this task/evidence contract; the frozen SSP2V wire header remains
`version=1` because no revision-v1 stream was ever produced and the pre-data audit did not change
the wire grammar. Any main-candidate failure closes this exact formulation as
`ABANDON_FIXED_SSP2V_V1_UNDER_COMP011_V2`. Do not tune the menu, syntax, starts, Lloyd rules,
target metrics, quality bands, primary-format set/tie order, rate gates, or resource gates on
these cells. The v1-to-v2 pre-data repair does not relax this
no-retuning rule once v2 first imports a development stream. A two-tuple pass authorizes only a
separately source-bound confirmation task; COMP-011 never opens confirmation.

## Question and claim boundary

On the sixteen already-fitted and already-QAT-quantized COMP-008 development fields, can the
complete self-identifying **operational SSP2V bundle under protocol v2** save at least 5% against
the strongest of all six frozen exact complete formats while conservatively preserving target
quality and bounded decode resources?

COMP-011 does not refit a Gaussian field, alter QAT, change geometry, or optimize a codebook against
pixels. VQ fitting sees only the ordered absolute RGB symbol triples from a cold-parsed SSPL1
stream. Target pixels become available only after every frozen VQ variant has been fit, encoded,
and cold-decoded; they are used solely for the preregistered quality filter and complete-stream
selection.

The task can establish complete-stream compression, fixed-symbol Lloyd convergence, target-relative
quality preservation, bytes-to-boundary decode time/RSS, and descriptive matched flat/RVQ results.
It cannot establish improved field-fitting convergence, renderer speed, a better renderer, global
state of the art, or confirmation evidence. The SSP2V-versus-SSP2Z control changes an operational
bundle: lossy RGB reconstruction, codebook syntax, index models, arithmetic index streams, and
container structure. It therefore cannot isolate VQ, RVQ, entropy coding, or any one component as
a causal mechanism. Flat/RVQ matched pairs are descriptive regardless of direction or magnitude;
neither an RVQ win nor an RVQ loss supports an RVQ-specific causal claim.

## Frozen provenance and inputs

Use only the eight frozen development images and two tuples `(12,6,6,8)` and `(16,8,8,8)` from
COMP-008/009. Bind, without opening a development payload at preflight:

- COMP-008 binding
  `9c5dca682d490567a56ce4fc4043112a5bacf72b2fee908e95e86e691a1ea0f2`;
- COMP-008 analysis seal
  `3368a3f96926b0eaf2e18119ae587d4bb3822c450c0f6f5a33e883c4e87e6792`;
- COMP-008 lifecycle replay seal
  `95829d0c563842adcc0a7788d7ff02a8e6bd74596f80a3a3966d78c2da9b1397`;
- COMP-008 `cells.jsonl` SHA-256
  `9f14774f091afb0b8c2e9c471f9e9bd21f2d8800f44da6e413a076acf33ccb8b`;
- COMP-009 source manifest
  `d75ee551ee650f0d465a01fc8f49b2ee0c4eee15579cf362ee3dfda2b96ad7d6`;
- COMP-009 source archive
  `fac4ca0978891b3cd16d477ffc04d12dc120168478192293d39af635fca7eb50`;
- COMP-009 input, run, and benchmark manifest seals recorded by its valid artifact;
- COMP-009 analysis record
  `435f011fe598263cd304b1cbe0754ca82e5fe1e7a49536db4099bdfff9166201`;
- COMP-009 analysis
  `f10c4a1906e4bd240c10253508f38c4053cfe332dec13b220262fee7ae990b30`;
- COMP-010 v2r2 repair seal
  `adbf2c48e1721b6d4b74960211e08ec3c770bebde1d068ddfc88e5a83f78f3a8`, bound to preflight
  `a00a7a553468dbe1f2e00e1d6e5ecb38fb3227356e090c73bbf564b39ec1bc00` and the identical child and
  worker seals recorded under Status.

### Authorities, artifact copies, and exact primary formats

An **authority** is a sealed upstream identity and its prescribed verifier, never an absolute path
or a convenient local copy:

- SSPL1 authority is the COMP-008/009 per-cell complete-byte length/hash plus absolute-symbol and
  decoded-boundary hashes, verified by strict cold parse of the sealed stream;
- SSP2E, SSP2S, SSP2L, and SSP2F authority is each sealed COMP-009 per-cell complete-blob
  length/hash, model/state record, decoded symbol/boundary hashes, and the frozen captured source
  that reproduces it;
- target authority is each COMP-008 bound prepared-development PNG encoded-byte hash, strict RGB
  sample hash, dimensions, and per-cell target hash;
- renderer authority at preflight is the actually loaded binary identity plus its bound source,
  ABI, loader, compiler, CUDA, driver, GPU, and Torch identities.

An **artifact copy** is a byte-identical, artifact-relative materialization made after its legal
import point. It is not promoted into a new authority. `import-streams` reparses each of the sixteen
SSPL1 authorities, verifies every sealed identity, and stores an immutable verbatim artifact copy.
During the following fit/encode stage, frozen COMP-009 captured source must reproduce SSP2E,
SSP2S, SSP2L, and SSP2F byte-for-byte and store verified artifact copies. Do not use
`absolute_symbols.npz`, a QAT field NPZ, or an old result row as a decoded-array authority.
Reconstruct every field boundary by cold-decoding its complete stream.

Only the later `import-targets` stage may first open the eight target authorities. It verifies them,
stores both the verbatim encoded PNG bytes and the canonical strict-RGB `uint8[H,W,3]` sample bytes
as immutable artifact copies, and records both hashes. All quality work uses those copies. Captured
source replay must first reproduce and cold-decode-seal the complete stream/candidate set, then may
open only the verified artifact target copies; it must not reopen an upstream target path. The
copied CUDA binary is similarly the only later execution binary after byte-identity verification,
but remains an artifact copy of the preflight authority.

No path belonging to the frozen confirmation set may be opened, hashed, stat-inspected beyond
already bound metadata, decoded, copied, or rendered.

For image `i`, define complete stream byte lengths:

```text
B0_i = complete SSPL1 bytes
Z_i  = complete SSP2Z bytes
E_i  = complete frozen SSP2E bytes
Sh_i = complete frozen SSP2S bytes
L_i  = complete frozen SSP2L bytes
F_i  = complete frozen SSP2F bytes
Q_i  = min(B0_i,Z_i,E_i,Sh_i,L_i,F_i)
V_i  = selected complete SSP2V bytes
```

Resolve a complete-byte tie for `Q_i` in exactly this order: SSPL1, SSP2Z, SSP2E, SSP2S, SSP2L,
SSP2F. The selected format and blob supply the primary bytes, exact quality-reference boundary, and
resource denominator. Every arm is a complete self-identifying stream. SSP2E/SSP2S use the exact
frozen COMP-009 syntax/table/model; SSP2L/SSP2F use their exact frozen COMP-009 syntax/models;
SSPL1 is the exact canonical upstream stream; SSP2Z is defined below. All six formats are mandatory:
any reproduction, complete-byte, strict cold-decode, or canonicality failure makes the cell
invalid/no-decision rather than silently removing that format from the minimum. Only after all six
pass do they participate in `Q_i`. No selector byte, out-of-band codebook, or free model state is
assumed or added to the minimum.

## Common fixed header

SSP2Z and SSP2V use little-endian structs:

```text
outer     <5sBBBQI>       # 20 bytes
header    <III4B4i12f5I>  # 100 bytes
directory <IQQ>           # 20 bytes per stream
```

The header reproduces the SSPL1 field and rendering semantics exactly: `N,H,W`; four bit depths;
four signed mean-range endpoints; `(scale_lo[2],scale_hi[2],color_lo[3],color_hi[3],
aa_dilation,sigma_cutoff)` as exact float32 values; and:

```text
render_chunk, renderer_id=1, support_fade=0, aux_model_id=0, zlib_level=9
```

The two formats therefore carry byte-identical 100-byte headers for a cell. Their version fixes
Morton mean deltas, byte-planar means, raw scales, circular rotation, opacity absence, and the
declared RGB grammar. Validate all field ranges, tuple membership, raw lengths, fixed flags, and
header-to-SSPL1 semantic equality.

## SSP2Z binary-framing control

SSP2Z changes framing only and must reproduce all eight ordered absolute symbol arrays and the
decoded field boundary exactly.

Outer:

```text
magic=b"SSP2Z", version=1, flags=0, stream_count=4,
total_bytes, CRC-32 over every byte after the outer record
```

Directory IDs and order:

```text
1 means_zlib
2 scales_zlib
3 rotation_zlib
4 colors_zlib
```

The prefix is exactly `20+100+4*20=200` bytes. All four payloads are copied byte-for-byte from
canonical SSPL1, are contiguous and nonempty, and must each be exactly one locally canonical
zlib-9 member with the header-implied raw length and no unused data, suffix, or unconsumed tail.
SSP2Z isolates fixed binary framing from the RGB representation change.

## SSP2V complete syntax

SSP2V has the same header and byte-identical SSPL1 `means`, `scales`, and `rotation` zlib payloads
as SSP2Z. Only the RGB payload grammar changes.

Outer:

```text
magic=b"SSP2V", version=1, flags=0,
stream_count=6+actual_stage_count,
total_bytes, CRC-32 over every byte after the outer record
```

Directory IDs and order:

```text
1 means_zlib
2 scales_zlib
3 rotation_zlib
4 descriptor_raw
5 codebook_raw
6 index_model_zlib
7 index_0_arith
8 index_1_arith       # RVQ only
9 index_2_arith       # three-stage RVQ only
```

Flat, two-stage RVQ, and three-stage RVQ therefore have prefix sizes 260, 280, and 300 bytes.
Offsets begin at the appropriate prefix and payloads are contiguous in directory order with no
gap, overlap, alias, or suffix.

The descriptor is exactly:

```text
<4sBBBBHHI>
tag=b"VQ1\0"
descriptor_version=1
family_id=0 flat, 1 RVQ
actual_stage_count
matched_rvq_stage_count
base_k
codebook_entries
reserved=0
```

`codebook_entries` is the total number of serialized entry-major RGB vectors: it is `L` for a
flat arm and `s*K` across all stages for an `s`-stage RVQ arm. It is not the per-stage `K` for
RVQ. The decoder must derive and require the exact value from the remaining descriptor fields and
must reject every mismatch before reading the codebook payload.

Only these eight variants are legal:

| family | matched stages | base K | actual stages | entries/stage | `codebook_entries` |
|---|---:|---:|---:|---:|---:|
| flat | 2 | 64 | 1 | 192 | 192 |
| flat | 2 | 128 | 1 | 384 | 384 |
| flat | 3 | 64 | 1 | 320 | 320 |
| flat | 3 | 128 | 1 | 640 | 640 |
| RVQ | 2 | 64 | 2 | 64 | 128 |
| RVQ | 2 | 128 | 2 | 128 | 256 |
| RVQ | 3 | 64 | 3 | 64 | 192 |
| RVQ | 3 | 128 | 3 | 128 | 384 |

There is no scalar fallback, escape, literal correction, alternate VQ menu, hidden selector,
codebook compression, or unused-entry pruning.

Codebooks are entry-major RGB:

- flat: `uint8[L,3]`;
- RVQ stage 0: `uint8[K,3]`;
- every later RVQ stage: `int16_le[K,3]`.

The raw codebook charge is exactly equal within each matched pair:

```text
two-stage RVQ:   3K + 6K = 9K bytes; flat has 3K entries * 3 = 9K bytes
three-stage RVQ: 3K + 6K + 6K = 15K bytes; flat has 5K entries * 3 = 15K bytes
```

Unused entries remain serialized and charged. Flat reconstruction is `codebook[index]`. RVQ uses
signed int32 accumulation and then:

```text
rgb = clip(stage0[index0] + sum(stage_j[index_j]), 0, 255)
```

The result is three ordered `uint32[N]` absolute symbol arrays. All non-RGB arrays remain exact.

Complete-byte accounting is:

```text
flat:
V = 260 + means_zlib + scales_zlib + rotation_zlib
        + 16 + 3L + index_model_zlib + index_0_arith

RVQ with s stages:
V = 20 + 100 + 20*(6+s)
        + means_zlib + scales_zlib + rotation_zlib
        + 16 + [3K + 6K*(s-1)]
        + index_model_zlib + sum_j index_j_arith
```

Persist framing/header/directory, common exact payloads, descriptor, raw codebook, frequency
model, each arithmetic index stream, and total independently. No report may conflate those terms.

## Canonical empirical index coding

Reuse the unchanged COMP-009 Python proof and native 32-bit arithmetic coder with total frequency
`T=32768`. For an index alphabet `A`, counts `c[a]`, and `N=8192`:

```text
R = 32768-A
alloc[a] = floor(R*c[a]/N)
remainder[a] = (R*c[a]) mod N
```

Distribute the remaining units by descending remainder, ties by ascending index, then set
`frequency[a]=1+alloc[a]`. Every frequency is positive and the sum is 32768.

Every imported cell header must have `N==8192`; reject any other value before constructing an
empirical model, fitting a codebook, or evaluating the formulas below. `8192` is a frozen protocol
constant, not an observed row count to substitute after import.

Concatenate stage frequency arrays in stage order as `uint16_le`, then encode that raw block as
exactly one locally canonical zlib-9 `index_model_zlib` member. Its uncompressed length is `2L`
for flat and `2sK` for RVQ. Each index stream uses its constant empirical CDF and the existing
canonical arithmetic termination.

Decode exactly 8192 indices per stream, reject out-of-range indices, re-encode every arithmetic
payload byte-identically, recompute the empirical model from decoded indices, and require a
byte-identical canonical model member. Require Python/native payload and decode parity for
alphabets `{64,128,192,320,384,640}` and persist the exact COMP-009 frequency-product certificate
for every index stream.

## Exact deterministic flat/RVQ Lloyd fitting

Fitting uses only the source stream's ordered absolute RGB triples. Target pixels, rendered
quality, and rate gates are unavailable to the fitter.

### Weighted unique-vector state

For each stage, collapse vectors into a lexicographically sorted unique-vector histogram with
exact integer counts. Lloyd assignment and updates operate on this weighted set. Materialize
per-row indices only after choosing the terminal codebook. Flat and RVQ stage-0 inputs lie in
`[0,255]^3`; later RVQ stages use signed int32 residuals. Later centroids must be exactly
int16-representable or the fit fails.

### Four frozen starts

Use exactly four starts, IDs `0..3`, for every flat fit and every RVQ stage. For a unique signed
int32 vector `v`, define:

```text
key(v) = SHA256(
    b"structsplat-comp011-vq-start-v1\0"
    || uint8(family_id)
    || uint8(matched_stage_count)
    || uint16_le(base_k)
    || uint8(actual_stage_index)
    || uint8(start_id)
    || int32_le(v.r) || int32_le(v.g) || int32_le(v.b)
)
```

The first centroid minimizes `(key,vector)`. Each later centroid is the unselected unique vector
maximizing its exact squared distance to the nearest selected centroid, ties by `(key,vector)`
ascending. If the unique count is smaller than the codebook size, select every unique vector and
fill remaining slots cyclically in `(key,vector)` order. Canonicalize centroid labels
lexicographically after initialization and every update.

### Integer Lloyd sweep

Use signed int64 component/distance arithmetic and checked uint64 or arbitrary-precision objective
accumulation. Each sweep:

1. assigns every unique vector to the nearest centroid by exact squared Euclidean distance, ties
   by lowest centroid index;
2. updates each coordinate of every nonempty cluster to the integer minimizing exact weighted
   squared error, with a half-integer tie choosing the smaller integer;
3. leaves an empty-cluster centroid unchanged;
4. lexicographically sorts centroids and remaps assignments;
5. stops only when the canonical codebook is unchanged.

For signed `sum_value` and positive `count`, the scalar update is:

```text
q,r = divmod(sum_value,count)
new = q if 2*r <= count else q+1
```

Hash every canonical codebook state. A repeated non-fixed state is a cycle and therefore
nonconvergence. A fixed point must be reached within 100 update sweeps. Cap/cycle is a valid method
failure, not protocol invalidity. All four starts for every flat fit and every RVQ stage must
converge; selecting around a failed start is forbidden.

Choose a stage winner by lowest exact terminal weighted SSE, then lexicographically smallest
serialized codebook, lexicographically smallest materialized index array, and smallest start ID.

RVQ is strictly sequential: fit/select stage 0, subtract its un-clipped integer reconstruction,
fit/select the next residual stage, and repeat once for a three-stage arm. Clip only the final
decoded sum. There is no joint refinement, entropy-constrained update, ECVQ, STE, renderer-aware
loss, rate-aware start choice, or post-fit codebook adjustment.

## Persisted CUDA quality authority

Full-image CPU rendering for nine arms per cell is not a feasible authority. Quality uses the
unchanged normalized CUDA renderer, but treats its atomic-order variation explicitly rather than
claiming bit determinism.

During preflight, build or resolve the exact CUDA extension once, copy the actual loaded extension
binary into the new artifact as an immutable artifact-relative file, and bind:

- binary relative path, size, SHA-256, ELF class/machine/build ID and dynamic-section digest;
- complete `ldd`/loader dependency identity and system `libstdc++` ABI;
- source and compiler command/hash, compiler version, CUDA toolkit/runtime, driver, GPU name/UUID,
  compute capability, Torch ABI, and loader environment;
- the required `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` state;
- `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.

Every later stage, subprocess, and captured-source replay must load that copied binary by its
verified artifact-relative path and recheck its hash before import. It may not rebuild, rediscover,
or silently load a cache path. Semantic equality uses the binary hash and artifact-relative
identity, never the original absolute build path. Replay copies/uses the persisted binary from the
evidence artifact after safe source extraction.

For every cell, the prior lifecycle stage must cold-decode and seal all eight SSP2V variants before
any target may be imported. Quality then independently cold-decodes the selected exact primary
`Q_i` and every variant. Render `Q_i` and each variant exactly three times. The render schedule is:

```text
repetition 0: Q_i, then variants in frozen descriptor order
repetition 1: variants in reverse frozen descriptor order, then Q_i
repetition 2: Q_i, then variants in frozen descriptor order
```

Synchronize CUDA before and after each render. Each render begins from independently reconstructed
decoded-boundary tensors; no numerator/denominator or renderer workspace may be reused across
arms. Clamp the resulting float32 image to `[0,1]`, copy it to CPU, and compute metrics there.
Rendering is not included in the codec decode-resource gate.

For every real arm, reconstruct `GaussianField` from the exact decoded float32 boundary and call
the unchanged production adapter with `field.conics(aa_dilation)`,
`field.radii(sigma_cutoff, aa_dilation)`, `field.effective_scales(aa_dilation)`, rotations,
colors, opacity absence, the header's `render_chunk`, `support_fade`, and `sigma_cutoff`, with
`mode="cuda"`. Construct the field as
`GaussianField(means, decoded_log_scales, rotations, colors, opacities=None)` exactly as in the
COMP-009 boundary path. Do not substitute decoded log-scales into a physical-scale slot, pre-
exponentiate them before that constructor, use an alternate conic/radius path, fall back to the
Torch renderer, or add renderer-only normalization. Pre-data tests must compare this call record
and output to the frozen COMP-009 production render semantics on a synthetic field.

At `import-targets`, decode each verified prepared PNG as stored strict RGB samples with no ICC,
gamma, alpha, resizing, cropping, or color-management transform. Let `T_u8` be the copied
C-contiguous `uint8[H,W,3]` samples and evaluate exactly

```python
T_hwc = T_u8.to(dtype=torch.float32, device="cpu") / 255.0
T = T_hwc.permute(2, 0, 1).unsqueeze(0).contiguous()
```

where `/ 255.0` is one CPU IEEE-float32 elementwise division. For a render, clamp the CUDA
float32 `HWC` result to `[0,1]`, synchronize, copy it to a new C-contiguous CPU float32 tensor, and
form `C` in `NCHW` identically without another normalization. Persist hashes of `T_u8`, `T`, and
each clamped render tensor.

For each `(cell,arm,render_repetition)`, one fresh single-thread CPU metric worker executes PSNR,
MS-SSIM, and LPIPS once each in the shown order. The worker uses
`torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`,
`torch.use_deterministic_algorithms(True)`, `torch.manual_seed(0)`, inference mode, and all four
thread environment variables fixed to one. The exact calls are:

```python
# C and T are CPU float32 [1,3,H,W]. Both must be finite and in [0,1].
delta = C.to(torch.float64) - T.to(torch.float64)
mse = delta.square().sum(dtype=torch.float64) / (3 * H * W)
psnr = 10.0 * math.log10(1.0 / float(mse))

ms_ssim_value = pytorch_msssim.ms_ssim(
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

`mse` must be finite and strictly positive and all returned metrics must be finite; otherwise the
cell is invalid/no-decision. Preflight must call each metric twice on byte-identical inputs in the
same fresh worker and require identical scalar IEEE-754 bytes. Bind Python, Torch, Pillow,
`pytorch-msssim`, and LPIPS source/version; bind the explicit MS-SSIM weights; and bind a canonical
hash over every LPIPS state-dict key, dtype, shape, and contiguous tensor byte. Missing, drifting,
or nondeterministic metric dependencies invalidate the run.

### Atomic repeat-spread caps

Preflight constructs one target-free overlapping synthetic field with `N=8192`, `H=W=192`. For
row `i`, let

```text
d_i = SHA256(b"structsplat-comp011-render-repeat-v1\0field\0" || uint32_le(i))
mean_x = float32(48 + (d_i[0] mod 96))
mean_y = float32(48 + (d_i[1] mod 96))
physical_scale_x = float32(1 + (d_i[2] mod 16)/4)
physical_scale_y = float32(1 + (d_i[3] mod 16)/4)
rotation = float32((d_i[4] mod 32) * float64(Python math.pi) / 32)
red   = float32(d_i[5]/255)
green = float32(d_i[6]/255)
blue  = float32(d_i[7]/255)
```

Division and the multiplication by pi are evaluated in IEEE float64 in the shown order and cast
once to IEEE float32. The named scale values are **physical pixel scales**, not log-scales. Build
the fixture only through

```python
GaussianField.from_numpy(
    means=means_f32, scales=physical_scales_f32, angles=rotations_f32,
    colors=colors_f32, opacities=None, scale_max=None, device=bound_cuda_device,
    dtype=torch.float32, color_grads=None, background_mask=None,
    filter_variance=None,
)
```

so the production boundary performs its single `torch.log` into `field.log_scales`. Passing the
physical values directly to the `GaussianField(..., log_scales=...)` slot, pre-logging and then
calling `from_numpy`, or otherwise double/excluding the exponential is forbidden. Persist hashes
of physical inputs, `field.log_scales`, `field.scales()`, `field.effective_scales(0)`,
`field.conics(0)`, and `field.radii(3,0)`. Use opacity absence, `aa_dilation=0`, `sigma_cutoff=3`,
`support_fade=false`, and `render_chunk=512`, and pass those exact production `GaussianField`
boundary products into the bound renderer adapter. For pixel `(y,x)` define

```text
t_yx = SHA256(
    b"structsplat-comp011-render-repeat-v1\0target\0"
    || uint16_le(y) || uint16_le(x)
)
target_u8[y,x,c] = uint8(t_yx[c]), c=0,1,2
target = target_u8.to(CPU float32) / float32(255.0)
```

Persist the complete named tensor/reference hashes and independently recompute them in the
preflight proof subprocess that loads the copied binary, not the original cache binary. Render the
fixture nine times through the copied binary and require:

```text
max(PSNR)-min(PSNR)       <= 0.001 dB
max(MS-SSIM)-min(MS-SSIM) <= 0.00001
max(LPIPS)-min(LPIPS)      <= 0.00001
```

Failure occurs before any development import and requires a new named protocol. Apply the same
spread caps to the three repeats of every real selected-`Q_i`/variant arm; exceeding a cap is
invalid/no-decision rather than a quality failure.

### Conservative metric envelopes

For selected-primary metric repetitions `B[0:3]` and variant repetitions `C[0:3]`, define:

```text
psnr_loss      = max(B_psnr)    - min(C_psnr)
ms_ssim_loss   = max(B_msssim)  - min(C_msssim)
lpips_increase = max(C_lpips)   - min(B_lpips)
```

These conservative values, not a chosen repeat or average, determine quality and selection.
Persist all individual metrics, min/max envelopes, spreads, and conservative differences.

### Decision exclusion bands

Freeze threshold exclusion half-widths:

```text
PSNR loss:       0.002 dB
MS-SSIM loss:    0.00002
LPIPS increase:  0.00002
median PSNR loss: 0.002 dB
```

For a threshold `T` and band `b`, a metric definitely passes if `value<=T-b`, definitely fails if
`value>T+b`, and is ambiguous otherwise. A variant is definitely quality-qualified only if all
three metrics definitely pass; it definitely fails if any metric definitely fails; otherwise it
is ambiguous.

The original quality thresholds remain:

```text
PSNR loss       <= 0.05 dB
MS-SSIM loss    <= 0.0005
LPIPS increase  <= 0.001
```

Thus definite qualification requires at most `0.048 dB`, `0.00048`, and `0.00098`, respectively.
An observed value in an exclusion band is never rounded or coerced to one side.

Order all variants by the exact tuple
`(complete_bytes, family_rank, matched_stage_count, base_k, ASCII_label_bytes,
complete_blob_bytes)`, where `family_rank=0` for flat and `1` for RVQ and byte comparisons are
unsigned lexicographic. Traverse that order:

- skip a definitely failed variant;
- select the first definitely qualified variant;
- if an ambiguous variant occurs before the first definite qualifier, the cell is invalid/no-
  decision because atomic variation could change selection;
- if no variant definitely qualifies and at least one is ambiguous, the cell is invalid/no-
  decision;
- if all variants definitely fail, the cell is a valid method failure with no scalar fallback.

Ambiguity in a larger variant after an already selected definite qualifier cannot affect selection
and is diagnostic only. Fit, encode, decode, and score all eight variants regardless; early
stopping is forbidden.

For the eight selected candidates in a tuple, sort the eight **conservative** `psnr_loss` values in
ascending numeric order, breaking an exact float tie by frozen image index. The conventional
median is `(sorted_loss[3] + sorted_loss[4]) / 2`, where brackets are zero-based. It passes if
`<=0.018 dB`, fails if `>0.022 dB`, and is invalid/no-decision in between. This preserves the
scientific threshold `0.02 dB` while preventing atomic ULPs from flipping the decision.

Captured-source replay uses the persisted CUDA binary and the same three-render schedule for the
decision-relevant variant prefix defined under Lifecycle and replay. For every baseline/candidate
minimum and maximum endpoint in that prefix, the absolute replay-minus-original difference may be
at most:

```text
PSNR:    0.001 dB
MS-SSIM: 0.00001
LPIPS:   0.00001
```

Replay must independently satisfy all repeat-spread caps, reproduce each decision-relevant
qualification class and selected variant, and remain outside every decision exclusion band. Exact
`float.hex()` metric replay is neither expected nor claimed; later nongating variant metrics are
not replay obligations.

## Candidate selection and quality diagnostics

The frozen variant order used above is:

```text
flat_s2_k64_l192
flat_s2_k128_l384
flat_s3_k64_l320
flat_s3_k128_l640
rvq_s2_k64
rvq_s2_k128
rvq_s3_k64
rvq_s3_k128
```

The complete-byte ordering is primary; the listed family order participates only after an exact
byte tie as specified. No target metric may choose a Lloyd start or change a codebook. For source
symbol `x[i,c] in [0,255]`, RVQ raw signed-int32 accumulator `a[i,c]` (equal to the flat codebook
value for flat), and final reconstructed symbol

```text
y[i,c] = min(255,max(0,a[i,c]))
e[i,c] = int64(y[i,c]) - int64(x[i,c])
rgb_sse = sum_i sum_c e[i,c]^2
sum_abs_error = sum_i sum_c abs(e[i,c])
sum_signed_error = sum_i sum_c e[i,c]
max_abs_error = max_i,c abs(e[i,c])
clip_low_components = count_i,c(a[i,c] < 0)
clip_high_components = count_i,c(a[i,c] > 255)
clip_components = clip_low_components + clip_high_components
clip_rows = count_i(any_c(a[i,c] < 0 or a[i,c] > 255))
```

use checked signed/unsigned 64-bit arithmetic and require equality with arbitrary-precision Python
recomputation. A raw value exactly `0` or `255` is not clipped. Flat arms must have all clip counts
zero. `distinct_input_rgb` and `distinct_reconstructed_rgb` count unique `x[i,:]` and final clipped
`y[i,:]` triples. For each reconstructed triple `u`, let `n_u` be the number of **distinct** input
triples mapped to `u`; define `merged_input_colors=sum_u max(0,n_u-1)` and
`collision_pairs=sum_u n_u*(n_u-1)/2`. Repeated rows of the same input triple do not create a
collision. Persist for every variant these exact final clipped-error/collision values, used entries
per stage, unique index tuples, convergence trajectories, component bytes, target metric envelopes,
and qualification state.

## Primary compression, quality, and operational-bundle gates

Apply independently to both tuples. A tuple cannot pass if any cell is invalid/no-decision or has
no definitely qualified VQ candidate.

### Scientific promotion endpoint: SSP2V versus strongest exact stream

Require conjunctively:

1. exact geometric mean `V/Q<=0.95`;
2. at least `7/8` strict `V<Q` wins;
3. every `V/Q<=1.02`;
4. the reused 100,000-replicate bootstrap upper order statistic is strictly `<0.98`;
5. every selected candidate definitely passes all three target-quality limits;
6. conservative median PSNR loss definitely passes the `0.02 dB` threshold;
7. every prescribed Lloyd start reaches a fixed point;
8. every decode-resource gate passes.

Use exact integer comparisons:

```text
GM <=0.95:      prod_i(V_i)*20^8 <= prod_i(Q_i)*19^8
win:            V_i < Q_i
worst <=1.02:   100*V_i <= 102*Q_i
bootstrap <.98: prod_j(V[b_r,j])*50^8 < prod_j(Q[b_r,j])*49^8
```

All products use arbitrary-precision positive integers. Reuse the exact COMP-008 bootstrap
matrix/hash with rows `r=0..99999` and columns `j=0..7`; `b_r,j` is the zero-based frozen image
index sampled at that entry. Represent each replicate ratio by its unsimplified positive integer
pair `(prod_j(V[b_r,j]),prod_j(Q[b_r,j]))`. Sort ratios by exact cross-product; break an equal-ratio
tie by ascending replicate index `r`. The upper statistic is the element at zero-based sorted index
`97499` and is strictly `<0.98` iff its numerator times `50^8` is strictly less than its denominator
times `49^8`. Equivalently, at least 97,500 replicate ratios satisfy that strict integer predicate.

### Operational-bundle isolation endpoint: SSP2V versus SSP2Z

Require per tuple:

1. geometric mean `V/Z<1`;
2. at least `7/8` strict `V<Z` wins.

The exact predicates are `prod_i(V_i) < prod_i(Z_i)` using arbitrary-precision integers and
`sum_i 1[V_i<Z_i] >= 7`. These conjunctive gates show only that fixed binary framing plus the exact
SSPL1 RGB zlib payload does not match the complete operational SSP2V bundle's rate. They do not
identify VQ, RVQ, codebook design, lossy reconstruction, or arithmetic indexing as a causal
mechanism. The 5% effect-size, worst-case, bootstrap, quality, and resources gates remain on the
stronger `V/Q` endpoint. Report `Z/B0`, `V/Z`, and all exact components separately.

### Matched residual-versus-flat endpoint

For every `(stages,K)` pair report codebook-byte equality, complete bytes and components, exact
symbol SSE, target-quality envelopes, utilization, clipping/collisions, convergence sweeps, and fit
time. This endpoint is descriptive and cannot rescue the primary decision or support a causal
VQ/RVQ claim. If a passing stream is flat, do not claim RVQ reuse succeeded; if it is RVQ, report
only that exact operational arm's result, not an RVQ mechanism effect.

## Decode-resource protocol

Use one COMP-011 unified fresh-worker entry point for SSP2V and every possible selected exact
primary format (SSPL1, SSP2Z, SSP2E, SSP2S, SSP2L, SSP2F), only after each format's exact
Python/native parity and cold-decode seal. The worker dispatches solely from strict complete-stream
magic/version after common startup. In each fresh child, importing modules, loading verified native
binaries/tables, reading the artifact-relative blob into immutable bytes, and verifying its
prebound blob hash occur before the timer for both arms. The timed scope starts immediately before
strict complete-container parse and ends after exact decoded-boundary tensor construction. No JIT,
build, filesystem read, target, or renderer is inside either timer.

All primary resource samples are new, contemporaneous COMP-011 measurements of the selected
SSP2V blob and the selected `Q_i` blob. Reusing COMP-009 samples, old medians/RSS values, or a sample
from a different exact format is forbidden. Apply the paired schedule:

- pin one thread to the lowest available CPU;
- one untimed fresh-worker warm-up per arm, in selected-`Q_i`-then-candidate order;
- nine measured fresh subprocesses per blob;
- for frozen image index `i=0..7`, tuple index `t=0` for `(12,6,6,8)` and `t=1` for
  `(16,8,8,8)`, and repetition `r=0..8`, run candidate first iff
  `(i+t+r) mod 2 == 0`, otherwise selected `Q_i` first;
- complete both arms of repetition `r` before `r+1`.

Candidate timing covers strict parse/CRC, zlib decode of means/scales/rotation/index model, native
index arithmetic decode and internal canonical re-encode, codebook lookup, signed sum/clip, and
exact decoded-boundary tensor construction. Primary timing covers the selected exact format's
strict parse/CRC, all of its zlib/arithmetic/model work including required canonical re-encode, and
the same decoded-boundary construction. Both exclude rendering, target metrics, fitting, and
encoding.

For each arm, sort its nine measured `(nanoseconds,repetition_id)` pairs ascending and use
`sorted_ns[4]` (zero-based) as the median; exact nanosecond ties break by ascending repetition ID.
Use the maximum measured child peak RSS, with ties broken by smallest repetition ID only for
provenance. Each worker reports Linux `resource.getrusage(RUSAGE_SELF).ru_maxrss` in KiB; multiply
the integer by `1024` exactly to obtain `peak_rss_bytes`. Define each image time ratio as the exact
rational `candidate_median_ns/Q_median_ns`.
Sort the eight ratios by integer cross-product, breaking exact-ratio ties by frozen image index;
the across-image upper median is `sorted_ratio[4]` (zero-based). Per tuple require:

- across-image upper-median time ratio `<=1.5`;
- worst-image time ratio `<=2.0`;
- every peak RSS ratio `<=1.5`.

The exact resource predicates are:

```text
upper median <=1.5:  2*candidate_ns <= 3*Q_ns at sorted_ratio[4]
worst <=2.0:         candidate_ns <= 2*Q_ns for every image
RSS <=1.5:           2*candidate_peak_rss <= 3*Q_peak_rss for every image
```

All denominators must be positive. Persist every launch order, PID, affinity, environment, blob
format/hash, raw nanoseconds, raw peak RSS, selected sample, rational order, and predicate. Persist
VQ fit/encode, CUDA quality rendering, metric evaluation, and total encoder search wall/RSS as
nongating diagnostics; do not relabel them as decode performance. Replay collects a new complete
paired sample set against the same selected `Q_i`; samples need not equal, but every resource gate
boolean must reproduce or the artifact is invalid/no-decision.

## Secondary SSP2L development survival screen

Independently recompute SSP2L versus SSPL1 for each tuple. Let `L_i` and `B0_i` retain the meanings
above. Require:

1. geometric mean `L/B0<=0.98`;
2. at least `7/8` strict `L_i<B0_i` wins;
3. frozen bootstrap upper `<1`;
4. exact symbol and decoded-boundary reconstruction;
5. fresh-process decode upper-median ratio `<=1.25`.

The exact predicates are `prod(L)*50^8 <= prod(B0)*49^8`; at least seven strict byte wins; after
sorting the 100,000 frozen-bootstrap ratios by exact cross-product with replicate-index ties, the
ratio at zero-based index `97499` has numerator strictly less than denominator; and, after a
separate unified contemporaneous SSP2L/SSPL1 schedule and exact-ratio ordering,
`4*L_ns <= 5*B0_ns` at zero-based upper-median index `4`. No COMP-009 timing sample or primary
`V/Q` sample is reusable for this separately labeled screen, even when `Q` happens to be one of
those formats.

That separate schedule uses the same worker, affinity, environment, untimed-then-nine-fresh
structure, timer boundary, RSS definition, zero-based per-arm median index `4`, and exact rational
ordering as the main protocol. Its untimed warm-ups run SSPL1 then SSP2L. In measured repetition
`r`, SSP2L runs first iff `(i+t+r) mod 2 == 0`; otherwise SSPL1 runs first. Complete both arms before
the next repetition.

A two-tuple pass may authorize a separately bound SSP2L confirmation task independently of SSP2V.
This is an outcome-informed engineering screen because COMP-009 development rates were visible
before the `0.98` threshold was written. It supplies no fresh confirmatory rate evidence and must
not be described as confirmation.

## Pre-data complexity and correctness gates

Before development import, use a synthetic `N=8192` RGB fixture with

```text
d_i = SHA256(b"structsplat-comp011-complexity-v1\0" || uint32_le(i))
rgb_i = uint8(d_i[0]), uint8(d_i[1]), uint8(d_i[2])
```

Run the complete eight-arm, four-start production menu. Require all synthetic starts to converge,
total wall time `<=300 s`, and peak RSS `<=4 GiB` on the bound host. The exact engineering
predicates are `wall_ns <= 300_000_000_000` and `peak_rss_bytes <= 4*2^30`.

Separately run a timing-only kernel mode that forces exactly ten full assignment/update sweeps for
each of four starts of the worst flat `L=640` case even if a fixed point occurs earlier; forced
sweeps may not change the scientific fitter or its convergence record. Require the linear
100-sweep projection `10*measured_ten_sweep_ns <= 300_000_000_000`. Failure occurs before
development import and requires implementation optimization or a newly named protocol, not opening
data and hoping real cells are easier. These are engineering feasibility gates, not scientific
endpoints.

Pre-data proofs must also include:

- brute-force integer-mean/tie checks, including negative residuals;
- exact assignment ties and centroid-label canonicalization;
- input-row permutation invariance of the unique-histogram fit;
- `U<K`, duplicate, dead-cluster, fixed-point, cycle, and cap fixtures;
- Python/native Lloyd parity on exhaustive small and signed fixtures;
- exact matched codebook-byte accounting and descriptor `codebook_entries` validation for all four
  pairs;
- strict SSP2Z/SSP2V offset/order/gap/alias/suffix/CRC/reserved/menu rejection;
- complete-container single-bit CRC rejection;
- empirical-frequency and canonical-zlib proofs for every legal alphabet;
- Python/native arithmetic payload/decode/re-encode parity and certificates;
- int16 residual-bound checks and exact final clipped RGB SSE/error/clip behavior;
- exact strict-RGB `uint8/255` conversion, full metric-call arguments, repeated-call determinism,
  target-quality envelope, spread, ambiguity, tie, and no-qualified-candidate tests;
- exact six-format `Q` selection/tie order and geometric-mean, win, worst, bootstrap,
  conservative-median-band, and operational-bundle gates;
- unified seven-magic fresh-worker dispatch (SSP2V plus six exact primary formats),
  contemporaneous paired schedule, zero-based medians, exact rational tie order, RSS aggregation,
  and rejection of old timing samples;
- persisted-renderer-binary load/hash/ABI tests, including cache-path substitution rejection;
- safe source-archive extraction and artifact relocation replay;
- explicit filesystem guards proving preflight cannot open development payloads,
  `import-streams`/fit/encode/cold-decode cannot open target or pixel payloads, replay cannot open
  target copies before regenerating its candidate-set seal, and no stage can access confirmation.

Any correctness, parity, dependency, binary identity, metric spread, source, environment, input,
stage-order, or replay failure is invalid/no-decision rather than evidence against VQ.
Nonconvergent prescribed Lloyd starts and absence of a quality-qualified VQ stream are valid method
failures.

## Lifecycle and replay

Stages are:

```text
preflight
import-streams
fit-encode-cold-decode-dev
import-targets
quality-select-dev
benchmark-dev
analyze
replay
```

Preflight in an empty output directory binds the task; all new modules/tests/native source and
binaries; full transitive source closure; COMP-008/009/010 metadata; compiler/ABI/CPU/GPU/driver/
affinity/thread environment; zlib and arithmetic coder; VQ menu and starts; bootstrap matrix;
metric packages/weights; the copied CUDA renderer binary; all synthetic proof results; and a safe,
relocatable source snapshot. Absolute paths are operational metadata only and never semantic asset
identity.

The legal transition is linear and fail-closed:

1. `import-streams` is the only original-run stage that may first open the sixteen development
   SSPL1 authorities. It verifies them and creates the immutable artifact stream copies. It cannot
   open, stat, hash, copy, or decode a target/pixel payload.
2. `fit-encode-cold-decode-dev` is append-only/resumable by image/tuple. From only the copied
   SSPL1 streams it reproduces and verifies SSP2E/SSP2S/SSP2L/SSP2F, creates SSP2Z, runs every
   prescribed start/stage, and encodes all eight variants for all sixteen cells. It then launches
   strict cold decoders for all six exact baseline formats and all `16*8=128` SSP2V blobs,
   verifies canonical re-encode plus symbol/boundary hashes, freezes every per-format component
   length/hash, selects `Q_i` by the frozen tie order, and writes one immutable candidate-set seal.
   No target access is permitted. A nonconvergent required start is a valid preregistered method
   failure and stops without target import; it may not be retried.
3. `import-targets` is enabled only if the complete candidate-set seal exists and its transitive
   inventory still matches. This is the first point at which any target authority may be opened.
   It performs the authority/copy procedure above, writes a separate immutable target-copy seal,
   and cannot modify any stream, model, codebook, index, blob, or `Q_i` choice.
4. `quality-select-dev` independently cold-decodes the sealed `Q_i` and all eight sealed variants,
   executes the frozen three-render/metric schedule, persists every envelope/classification, and
   selects or fails closed. It scores all eight arms; early stopping remains forbidden.
5. `benchmark-dev` runs only the unified contemporaneous fresh-worker protocols for the selected
   `V_i`/`Q_i` pairs and the separately labeled SSP2L/SSPL1 screen. It may not import an upstream
   timing record.
6. `analyze` reads only verified persisted records and computes the frozen exact predicates,
   bootstrap, quality/resource gates, and decisions.

The original artifact must guard every open and persist a stage journal containing prior-stage
seal, path class, purpose, and monotonic sequence number. A resumed stage must reproduce the
existing prefix byte-for-byte; deleting/replacing a record or entering a later stage early is
invalid/no-decision.

If a required Lloyd start fails before a complete 128-blob candidate-set seal exists, seal that
exact terminal trajectory as a valid main-method failure and stop without importing targets; no
SSP2V or SSP2L authorization is issued. If all 128 blobs are sealed but quality leaves any cell
without a definitely qualified candidate, finish scoring all arms, seal the valid main-method
failure, skip the unavailable SSP2V `V/Q` resource schedule, and allow only the independently
specified SSP2L/SSPL1 schedule to continue. The full primary resource gate is evaluated only when
all sixteen `V_i` selections exist. An invalid/no-decision condition never follows a valid-failure
shortcut.

### Decision-relevant captured-source replay

Replay is decision-relevant, not a promise to reproduce every nongating diagnostic. After safe
source extraction in a different randomized root, it uses the verified artifact stream copies and
captured sources, never upstream absolute paths. It must, before opening any artifact target copy:

- reparse SSPL1 and reproduce SSP2Z, SSP2E, SSP2S, SSP2L, and SSP2F byte-for-byte;
- rerun every prescribed Lloyd start/stage and reproduce all eight SSP2V blobs per cell
  byte-for-byte;
- strict-cold-decode all `16*8` variants and all exact baselines, reproduce the candidate-set seal,
  decoded symbols/boundaries, complete lengths/hashes/components, and selected `Q_i`.

Only after that replay seal matches may replay open the target artifact copies and load the
persisted renderer binary. For a cell with a selected variant at position `k` in the frozen
complete-byte/tie ordering, its **decision-relevant quality prefix** is positions `0..k` inclusive.
If no variant was selected, all eight positions are decision-relevant. Replay renders that prefix,
using the three-render schedule with only that subset and ordering the subset by frozen descriptor
order (or its exact reverse) inside each repetition. It must reproduce every prefix qualification
class and selected variant under the frozen endpoint tolerances and recompute the conservative
tuple median. Later variants were scored in the original run but cannot change selection and their
metric values/classifications are nongating diagnostics, so they are not replay obligations.

For a full candidate set, replay must recompute every exact `V/Q` rate, win, worst, bootstrap,
operational `V/Z`, convergence, quality, and SSP2L gate input; rerun every originally applicable
unified fresh-worker `V_i/Q_i` and SSP2L/SSPL1 schedule; and reproduce every component-gate boolean
and both final decisions. For a preregistered earlier failure branch, replay instead must reproduce
the exact stopping condition, terminal seal, forbidden-target non-access, inapplicable-gate state,
and failure decision; it must not fabricate missing candidates or measurements. Fresh timing/RSS
samples and metric envelope endpoints need not be bit-identical, but all applicable frozen
schedules, tolerances, exclusion bands, classifications, selected formats/variants, gate booleans,
and final decisions must agree. Fit/encode/render wall time, later-variant metrics, and matched-pair
descriptive statistics are nongating and need not replay. Replay may not load NPZ authorities,
rebuild or rediscover the renderer, reopen an upstream target path, access a target before its
regenerated candidate-set seal, or access confirmation.

## Decisions and fallback

The main decision is:

```text
both tuples pass every validity, primary V/Q rate, conservative quality,
Lloyd convergence, operational-bundle, and V/Q decode-resource gate
    -> AUTHORIZE_SEPARATELY_BOUND_SSP2V_CONFIRMATION
otherwise, for a valid method failure
    -> ABANDON_FIXED_SSP2V_V1_UNDER_COMP011_V2
```

The SSP2L survival screen independently yields either
`AUTHORIZE_SEPARATELY_BOUND_SSP2L_CONFIRMATION` or no SSP2L confirmation authorization.

COMP-011 computes no ambiguous post-hoc "zero-opt oracle." Failure authorizes no tuning on these
cells. A future separately bound task may, before touching a genuinely new development split,
define a renderer-aware or ideal-rate headroom oracle with its capacities, quality authority, and
complete-rate accounting frozen in advance. The only inherited contingency threshold is at least
10% ideal complete-rate headroom at no more than `0.25 dB` loss. Passing would authorize only a
renderer-aware STE/ECVQ experiment on new development data; failing would abandon that VQ lineage.

## Acceptance criteria

- [x] A real successful independently audited COMP-010 captured-source replay seal has been
      inserted into this task.
- [ ] COMP-011 preflight succeeds before any development or confirmation payload is opened.
- [ ] The copied CUDA renderer binary is hash/ABI/environment bound and used by every later stage.
- [ ] All synthetic correctness, complexity, repeat-spread, arithmetic, container, and replay
      proofs pass pre-data.
- [ ] `import-streams`, the sealed 128-variant cold-decode set, and only then `import-targets`
      occur in that order; all sixteen SSPL1 inputs and eight targets match their distinct sealed
      authorities and artifact-copy identities.
- [ ] All six `Q` formats are exact and eligible, the tie order is frozen, and all eight SSP2V
      variants are complete, canonical, independently decodable streams with exact component
      accounting.
- [ ] Every prescribed Lloyd start either reaches its frozen fixed point or produces a valid
      preregistered method failure; no retry/tuning occurs.
- [ ] CUDA quality envelopes use three independent renders, stay within spread caps, and never
      permit an exclusion-band ambiguity to determine selection or a gate.
- [ ] Candidate/selected-`Q` fresh decode measurements use the unified contemporaneous frozen
      schedule/scopes and no old timing samples.
- [ ] Analysis and captured-source replay agree on every decision-relevant exact artifact, prefix
      quality classification/selection within tolerances, component-gate boolean, and decision.
- [ ] Independent quantitative/artifact review passes before any confirmation action.

## Interfaces allowed

New COMP-011 benchmark modules/native fitter source, tests, this task, later research documentation,
ignored result evidence, and ARA records only. Do not edit production codec/fitter/renderer/CLI,
COMP-008/009 frozen scientific sources/tasks/artifacts, COMP-010 repair evidence, or any
confirmation material.

## Minimal implementation decomposition

- `benchmarks/ssp2v_lloyd.py`: Python proof fitter and exact starts/ties/convergence.
- `benchmarks/ssp2v_lloyd_ext.cpp`: checked native exact-integer fitter.
- `benchmarks/ssp2v_container.py`: SSP2Z/SSP2V grammar, empirical model, strict decode.
- `benchmarks/ssp2v_quality.py`: persisted CUDA binary loading, metric envelopes, qualification.
- `benchmarks/ssp2v_execute.py`: eight-arm cell execution and component records.
- `benchmarks/ssp2v_actual_coder.py`: exact analysis, bootstrap, exclusion bands, decisions.
- `benchmarks/ssp2v_decode_worker.py`: fresh bytes-to-boundary timing/RSS.
- `benchmarks/ssp2v_actual_run.py`: lifecycle, source binding, journals, persisted binary, replay.
- Focused tests for each module plus one hostile lifecycle/replay integration suite.

## Depends on

COMP-001/002/003/004/006/007/008/009, successful COMP-010 replay repair, BENCH-016, the frozen
COMP-009 arithmetic coder, and the unchanged normalized CUDA renderer.
