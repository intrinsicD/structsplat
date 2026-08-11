# HIER-027 — Cold pure-additive capacity threshold confirmation

## Context

HIER-026 gives the first prospectively source-bound evidence that a denominator is not the only
route to high image fidelity. Exact four-array projected progressive N=896 and cold N=960 fields
beat normalized N=640 by `+0.75388/+0.94493 dB` mean PSNR and improve mean MS-SSIM and local
maxima. They still fail the frozen composite gate because dense-forest detail raises per-cell
LPIPS and produces material directional smear; same-count additive remains `-0.84193 dB`.

Only after that decision was sealed, ordinary cold projected additive counts were tested on the
now-consumed HIER-026 cells. N=1024 still fails the forest LPIPS guard. N=1088 passes every numeric
clause on all eight cells with mean/minimum PSNR deltas `+1.68200/+0.98761 dB` and worst LPIPS
delta `+0.00335`; N=1152 also passes at `+1.97971/+1.04845 dB`. Those post-hoc probes select this
killing test but cannot confirm it. The simplest remaining hypothesis is raw additive capacity,
not progressive topology, gauge, coefficient optimization, or normalization continuation.

## Goal

Confirm on eight newly selected untouched official DIV2K validation images whether one ordinary
cold-fitted, one-pass, denominator-free, opacity-free additive field at N=1088 robustly matches
normalized N=640 under the unchanged HIER-026 perceptual, structural, local, and visual gate; use
N=1152 only as a predeclared fallback and report exact count/work exchange.

## Method and endpoint contract

All fit arms use the maintained `aniso_onedge`/WSE initializer with `flank_offset_frac=0`, feature
scale cap 12 pixels, bilinear signed RGB, the declared seed, L1 + 0.3 SSIM, Adam learning rates
means `5e-2`, scales `3e-2`, rotations `1e-2`, RGB `3e-2`, best-PSNR/final-count checkpoints every
25 updates, hard three-sigma support, no AA dilation, 256-row render chunks, and exactly 500
updates. No progressive births, proxy target, staged loss, adaptive count, or topology change is
allowed.

Every pure-additive endpoint strips training-only scale caps and persists exactly means,
log-scales, rotations, and signed RGB. It cold-renders all rows in one `cuda_additive` pass. No
normalization denominator, opacity, mass, level, mask, residual image, target, optimizer,
auxiliary RGB, or second-pass metadata survives.

The unchanged HIER-024 projection may change RGB only or return its incoming field exactly. Its
PCG limit 48, ridge `1e-8`, coefficient limit 16, input-centered start/regularization, explicit
frozen base, strict-lower raw MSE, MS-SSIM tolerance `1e-5`, LPIPS noninferiority, and pixel/7x7
tolerance `1e-6` remain unchanged. Projection/operator/metric work is charged separately.

## Phase A — procedural preflight

Reuse HIER-026's focused CPU/CUDA tests for four-array stripping, deterministic initialization,
finite gradients, coefficient bounds, persistence/cold parity, decision clauses, and projection
rollback. Add narrow driver/decision tests proving exact per-arm counts, 500-update work
(`N*500` Gaussian-row updates), shared N=640 and N=1088 pre-projection endpoints, source bindings,
and the N=1088-then-N=1152 decision ladder. Procedural fixtures cannot select a count or threshold.

## Phase B — frozen untouched DIV2K validation confirmation

Reuse the already verified official archive without opening additional members:

- URL: `https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip`
- bytes: `448993893`
- archive SHA-256: `20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc`

Exclude all four HIER-026 files. Rank the remaining canonical `0801.png`–`0900.png` names by
`SHA256("HIER-027-confirm-v1:" + filename)` before member extraction or image decoding. The first
eight names are frozen:

| rank | file | selection SHA-256 | file SHA-256 |
|---:|---|---|---|
| 1 | `0859.png` | `03488568c8031c428e16d4365ce5c3241276d460b4eb944204aec6dbe1cdfe42` | `3ada872de7c5def1d408920385db278b1ff3a5a0cfcab83105a789ff540a1827` |
| 2 | `0833.png` | `03f45d5a4ad1a7e29466b4bf012b4b4ba1ae96cbf1bcecc07cff36ac3c98e8ce` | `2e9668b3a318284ec90c9bbdd940317ecd2f7b95314e68c48c94d2380fad679a` |
| 3 | `0874.png` | `08dafd50533c303e3375e55fa7cb1b04f36067caa8694ffd175506b10c5cc5a3` | `11cb511247d70d84adad5557a720254e5f73e3786dbcd399c6053a1982ce1784` |
| 4 | `0880.png` | `0a05d43823a705d32c5b2daf099b7901d9ad0a8d1c62d32a976a52c296a02f5b` | `db5773c6e460824c5132c23917492fda7acd370c87e9ae6293a0103fee2b642d` |
| 5 | `0802.png` | `0a31d512d0f0b526a503c3a51eb2f0c274984e156e6bf4eac75479b564cefd99` | `4ad6f3ca8bf740192042978121f05ec493ddbe5a3da5584eaf0d9699c25ee431` |
| 6 | `0808.png` | `0e0d3b42d9d4ee8fbe42f756119af883e9d47ec6ec58e6825c65ae99c2530824` | `956528ab3e0fadad1ed8ce93f93a30bf9f58c36ffa9dd775e2ad362ffdcf5ace` |
| 7 | `0815.png` | `1225e9713eb595e0f3482a4fc07b26459f50c929ea94d48a5ce6648bd7bdebf8` | `c8f278e51f2bc9be7a696935b7e386eb4adafde24572d8ecdd4edf8adf4b4108` |
| 8 | `0889.png` | `151d9fb642f2afc1b96797072e537accdbfe2798591498e1ff09a59952edfe9d` | `a8f73c42065e3193c4deb883dcb3bc432a3f838e9be5bacea708ee39eb2c6e04` |

Compute and insert each member SHA-256 from `unzip -p` byte streams before extracting or decoding
these images. Do not change sources, arms, counts, schedules, metrics, or gates afterward.

Use max-side 160, seeds 0/1, exact owned CUDA renderers, required LPIPS, and these frozen arms:

1. `normalized_plain_n640`: normalized N=640, 500 updates;
2. `additive_plain_n640`: ordinary additive N=640, 500 updates;
3. `additive_projected_n640`: the exact same endpoint plus safe projection;
4. `cold_additive_projected_n1024`: ordinary additive N=1024 plus safe projection, a
   non-selectable boundary control because it already has a consumed counterexample;
5. `cold_additive_plain_n1088`: ordinary additive N=1088, 500 updates;
6. `cold_additive_projected_n1088`: the exact same endpoint plus safe projection, primary; and
7. `cold_additive_projected_n1152`: ordinary additive N=1152 plus safe projection, fallback.

The N=640 plain/projected arms and N=1088 plain/projected arms must share exact pre-projection
field digests. Rows bind archive/source/selection/initial/pre-projection/proposal/final hashes and
save fields, histories, native reconstructions, errors, worst crops, metrics, calls, work, time,
memory, projection clauses, payload keys, and parity receipts.

Run exactly:

```bash
PYTHONPATH=src python scripts/experiments/hier027_cold_additive_capacity.py \
  /tmp/structsplat-hier027-div2k-valid-20260811/DIV2K_valid_HR \
  results/hier027_div2kvalid8_s160_capacity_s01_confirmation_2026-08-11 \
  --max-side 160 --seeds 0 1 --device cuda --lpips
```

## Frozen decision ladder

All 112 cells must complete. Every pure endpoint must be finite, exact-count, coefficient-bounded
(`<=16`), exactly four-array, direct one-pass, and internal/cold/repeated parity safe (`<=2e-5`).
Every projection must satisfy all safety clauses or return its incoming field exactly. Any
integrity failure rejects that rung.

Compare each projected additive rung to paired `normalized_plain_n640`. A rung is numerically
quality-capable only if the unchanged HIER-026 clauses all pass:

- mean PSNR at least normalized and every cell within `0.10 dB` of normalized;
- mean MS-SSIM at least normalized minus `1e-4`;
- mean LPIPS at most normalized plus `0.002`, every cell within `+0.01`;
- mean pixel and 7x7 RMSE maxima each within normalized plus `0.005`, every cell within `+0.02`.

Native review must additionally find no material new lattice, checker, ringing, hole, wash, color
lobe, or blur. N=1024 is contextual and cannot be selected even if its fresh cells pass, because
the consumed `0860` counterexample remains part of the knowledge state. Select N=1088 only if it
passes all numeric and visual clauses. If N=1088 fails, select N=1152 only if it passes those same
clauses, mean PSNR is at least `0.50 dB` above normalized, and no paired cell is below normalized.
Otherwise retain the negative result and do not tune this bank.

A selected rung establishes only that normalization is unnecessary for this tested max-side-160
fidelity target when paying the measured 1.70x or 1.80x Gaussian count and 1.70x or 1.80x
Gaussian-row-update proxy. It does not make additive better at N=640, compare equal bytes, establish
full-resolution/rate/downstream behavior, define a codec, or authorize a production/default change.

## Non-goals

- No new method module, renderer/fitter/pipeline/default/semantic/codec change, opacity, hidden
  payload, progressive topology, adaptive count, perceptual-loss tuning, or threshold relaxation.
- No replay or retuning on these eight images after the one frozen matrix.
- No equal-byte, full-resolution, convergence-speed, downstream, or general-corpus claim.

## Acceptance criteria

- [x] Fresh names, controls, counts, schedules, work, metrics, exact command, and gates freeze before
      member extraction or pixel decode.
- [x] Member hashes bind before selected-image extraction/decode.
- [x] Focused procedural/driver/decision checks pass (`27 passed`; HIER-027 driver ladder plus the
      reused HIER-026 capacity and HIER-024 projection contracts).
- [x] The complete 112-cell matrix executes once into an immutable checker-valid report.
- [x] Results/visual audit records the frozen negative: N=1088/N=1152 fail only isolated
      per-cell pixel-maximum clauses despite broad aggregate gains.
- [x] Tasks/docs/ARA and focused/structural/full verification outcomes synchronize; the full gate
      retains only the nine inherited baseline/environment failures recorded below.

## Interfaces touched

One experiment driver, a narrow report-checker schema, focused driver/decision tests, this
task/Index/session brief, and results-driven docs/ARA only. No method or maintained public interface.

## Depends on

HIER-026/025/024, FIT-046/048, CORE-009/013, BENCH-002, ADR-0003/0006

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `5e808e14599b18cd7369b377ad1818f23106099f86deb419b0ce2e47bf6c21f1`

### Handoff log

Producer-frozen confirmation and outcome audit complete without a distinct prospective reviewer.
The checker-valid immutable report contains 112/112 cells. N=1088/N=1152 gain
`+1.84883/+2.19555 dB` mean PSNR and improve every aggregate perceptual/structural/local metric,
but isolated per-cell pixel maxima fail the unchanged gate. The task is a strict negative; do not
relax thresholds or retune this bank. Distinct review remains pending.

### Outcome

Neither selectable rung passes. Projected cold N=1088 and N=1152 have minimum paired PSNR gains
of `+1.34241/+1.55374 dB`, and every PSNR, LPIPS, aggregate MS-SSIM, and aggregate local clause
passes. The all-cell pixel-maximum clause fails at two cells per rung (`0833/s1`, `0874/s1` for
N=1088; `0859/s0`, `0833/s1` for N=1152). Native review finds no material frame-scale artifact,
but cannot override the frozen numeric rule. Ordinary capacity therefore remains a near-miss and
routes the sparse-allocation mechanism to HIER-028. Evidence:
`ara/evidence/hier027-cold-additive-capacity-2026-08-11/run.md`.

### Handoff

#### Objective

Confirm whether ordinary projected cold additive N=1088, with N=1152 fallback, robustly meets the
unchanged normalized-N=640 fidelity gate on untouched official validation images.

#### Changes

Added a frozen seven-arm confirmation driver, focused source/count/work/decision tests, report
checker coverage, and results-driven task/docs/ARA records. No maintained method or default changed.

#### Evidence

The 2,037-file source-snapshotted bundle is immutable and checker-valid. Focused HIER-027 plus
reused capacity/projection tests pass 27/27. All endpoint/procedural gates pass; the frozen quality
gate fails only isolated per-cell pixel maxima for both selectable counts.

#### Assumptions

Gaussian-row updates are a work proxy rather than equal FLOPs or equal bytes. The source-name and
member-hash binding preceded decode; dirty sources and producer review keep the result provisional.

#### Uncertainties

The result is max-side 160, eight images, two seeds, one device, unequal count/work, and not
independently reviewed. It does not measure complete bytes or full-resolution behavior.

#### Review focus

Check pre-decode bindings, exact four-array persistence, shared pre-projection hashes, fail-closed
projection, work accounting, and the four isolated pixel-maximum counterexamples.

#### Protected actions not taken

No threshold relaxation, report mutation, consumed-bank tuning, maintained renderer/pipeline/
default/semantic/codec change, or broad quality claim.

#### Recommended next action

Use the prospectively frozen HIER-028 residual-allocation test rather than another cold count sweep.

### Review

#### Verdict

Provisionally accepted as a strict cold-capacity near-miss

#### Self-reviewed

Yes

#### Correctness

Focused tests, bundle validation, read-only metric recomputation, source/count/work hashes,
projection rollback, payload audit, and cold/repeated parity pass. The decision matches all paired
metrics without averaging away local failures.

#### Evidence quality

The official source binding preceded decode and the report retains every arm, field, history,
metric, and visual. Dirty sources, one device, and absent distinct review prevent independent
acceptance.

#### Simplicity

The rejected candidates use the existing cold fit and ordinary four-array additive endpoint; the
negative result adds no maintained method.

#### Missing cases

Full resolution, more images/devices, complete bytes, rate-distortion, downstream response, and
independent protocol/outcome review.

#### Required changes

None for retaining the HIER-027 result. Do not relax its gate or tune its consumed bank.

#### Optional improvements

Compare equal complete bytes only after a serialization contract exists; do not infer rate from
Gaussian count alone.

#### Verification outcome

The focused HIER-022--028 set passes `83/83`; Ruff and every structural checker pass. The full
portable gate reports `1,952 passed, 26 skipped, 9 failed`. All nine are inherited and outside this
task: one rank-deficient affine condition-number expectation, six subprocess imports resolving an
external installed StructSplat without `PYTHONPATH`, one Torch 2.7 CUDA-property mismatch, and one
pre-existing descriptor-swap race. No HIER-027/HIER-028 test fails.

### Notes

The reversible fallback is omission of the driver and retention of the report. HIER-026's gate
and consumed counterexamples remain authoritative; this task does not rewrite them.
