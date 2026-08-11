# HIER-028 — Residual-pursuit pure-additive confirmation

## Context

HIER-027 prospectively confirms that ordinary cold additive capacity is globally strong but not
yet robust under the frozen local gate. On eight untouched DIV2K validation images and two seeds,
projected N=1088/N=1152 fields beat normalized N=640 by `+1.84883/+2.19555 dB` mean PSNR while
improving mean MS-SSIM, LPIPS, pixel maximum, and 7x7 maximum. Both nevertheless fail the exact
per-cell pixel-maximum clause: N=1088 has isolated `0833/s1` and `0874/s1` regressions of
`+0.06168/+0.02780`, and N=1152 has isolated `0859/s0` and `0833/s1` regressions of
`+0.03087/+0.04570`. Their paired 7x7 maxima and every global/perceptual clause pass. Native
inspection finds no frame-scale new artifact, so the remaining problem is sparse support
allocation rather than a normalization-denominator or broad-capacity deficit.

The result matches the motivation of GaussianImage++'s distortion-driven densification: allocate
new Gaussian primitives at the highest reconstruction distortion rather than redistributing a
whole cold initialization. A killing test on the already consumed HIER-026 bank appended
deterministic residual Gaussians to its projected cold N=960 endpoint. At the existing minimum
scale of 0.35 pixels, the smallest tested tail (64 rows, N=1024 total) passes all unchanged
HIER-026 PSNR, MS-SSIM, LPIPS, pixel, and 7x7 clauses on all eight cells. It yields
`+1.44896/+0.97387 dB` mean/minimum paired PSNR and worst LPIPS delta `+0.00594`; direct-render
parity is `<5e-7`. This consumed diagnostic selects one prospective recipe but cannot confirm it.

## Goal

Confirm on eight newly selected untouched official DIV2K validation images whether an ordinary
projected cold additive N=960 base plus 64 deterministic residual-pursuit Gaussians gives a
robust, one-pass, denominator-free, opacity-free N=1024 field under the unchanged HIER-026
perceptual, structural, local, integrity, and native-visual gate.

## Method and endpoint contract

Fit the N=960 base with the maintained `aniso_onedge`/WSE initializer, `flank_offset_frac=0`,
feature scale cap 12 pixels, signed RGB, L1 + 0.3 SSIM, Adam learning rates means `5e-2`, scales
`3e-2`, rotations `1e-2`, RGB `3e-2`, best-PSNR/final-count checkpoints every 25 updates, hard
three-sigma support, no AA dilation, 256-row render chunks, and exactly 500 updates. Apply the
unchanged HIER-024 appearance projection transaction to this N=960 endpoint before pursuit.

Append exactly 64 tail Gaussians sequentially. At each step, cold-render the accumulated additive
field analytically, rank all pixels by raw RGB MSE with row-major index as the deterministic tie
break, place one Gaussian at the highest-error integer `(x,y)` coordinate, set both scales to
exactly `0.35` pixels and rotation to zero, and set signed RGB to that pixel's current raw residual.
Update the analytic reconstruction with the exact hard-three-sigma additive kernel before choosing
the next row. Do not optimize, project, split, prune, merge, or adapt the count after the tail.

The N=1024 endpoint must retain the N=960 base prefix bit-exactly and persist exactly means,
log-scales, rotations, and signed RGB. It cold-renders all rows in one `cuda_additive` pass. No
normalization denominator, opacity, mass, level, mask, target, residual image, pursuit coordinates,
optimizer, auxiliary RGB, or second-pass metadata survives. Encoder-side pursuit telemetry and
source-derived state stay outside the endpoint.

## Phase A — focused method preflight

Add a typed, default-off residual-pursuit method with lazy torch imports. Tests must cover invalid
contracts, deterministic row-major selection, exact base/tail/total counts, base-prefix bit
identity, fixed tail geometry, signed corrections, coefficient limits, finite gradients, no input
mutation, exact four-array persistence, analytic/cold/repeated parity, CPU/CUDA parity, and reduced
worst residuals on procedural constant/ramp/edge/blob/texture fixtures. Procedural tests cannot
select the method.

## Phase B — frozen untouched DIV2K validation confirmation

Reuse the already verified official archive without opening additional members:

- URL: `https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip`
- bytes: `448993893`
- archive SHA-256: `20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc`

Exclude all twelve HIER-026/HIER-027 files. Rank the remaining canonical `0801.png`–`0900.png`
names by `SHA256("HIER-028-confirm-v1:" + filename)` before member extraction or image decoding.
The first eight names are frozen:

| rank | file | selection SHA-256 | file SHA-256 |
|---:|---|---|---|
| 1 | `0804.png` | `0686f57768896183a307e62c52b53806515c65b82856225f0053c3b51c7da0c3` | `16b5fdbe808b868bed0be32f235208a1716d44e271a37b79cbc77ab53d2f6bdb` |
| 2 | `0830.png` | `0c84c4de7ca7ce6cfb42573327b2c34933b88bc53c939e0b5a403f747e5bca5f` | `4eb18566ab01447a06daf0314a3711aa78cea5ca0eaa47cfedafbceeb6dd0a3e` |
| 3 | `0822.png` | `130cdf4d4c1a67dab7b4ce502044a2ecc5f6f1b8bd01365dce3ffc4f11311db3` | `a1d308fd62adecb1ea8b0fa8d0c687c92d3cf0d3358e598c8b97aca1b9cf8ad0` |
| 4 | `0812.png` | `132e21bc39e02a6cde90ba28d3a64c12d575bb6d8e2a001c5154924edda6a63c` | `49e45b8922872b44ece90db047756f3a5356612bb6ee30bdc23df2bd208ec861` |
| 5 | `0810.png` | `1704a6e1b96ad30381b0dfba6e4ab8a5d3ee7a61df23689ac625c9fe46a996fd` | `6940c660b97d2c5f1113101c3e6360d1d6886743c5796cad52224b8076b903f8` |
| 6 | `0862.png` | `18240279a254669300683c105df63f9584d1a396417d783ef5db734a05eb2313` | `31a02d7392ee9dadd4b8a2c1b5b9d670943135d0d40e85d4178ab77923c75548` |
| 7 | `0803.png` | `1a48bfa234e74bd95c2f7875565809acaedf73de995088c0b532c105f1eb0e06` | `4b0148a9a1ff877ad9f76e65736a50cc36e10822b5d8ccd2abb2988ff4e1782b` |
| 8 | `0826.png` | `1ac09ff808f01c4e326025121790ba7aa336e7889bf4ad34437fc1dc7042729c` | `b0f675a14e8fe9f2ec0b705bee98d75f8a22478eafdc1a0a0afc0f820bc5ab4d` |

Compute and insert each member SHA-256 from `unzip -p` byte streams before extracting or decoding
these images. Do not change sources, arms, counts, schedules, method, metrics, or gates afterward.

Use max-side 160, seeds 0/1, exact owned CUDA renderers, required LPIPS, and these frozen arms:

1. `normalized_plain_n640`: normalized N=640, 500 updates;
2. `cold_additive_projected_n960`: ordinary additive N=960 plus safe projection, the exact base;
3. `residual_pursuit_additive_n1024`: that exact N=960 endpoint plus 64 frozen pursuit rows; and
4. `cold_additive_projected_n1024`: ordinary cold additive N=1024 plus safe projection, the
   same-count allocation control.

The N=960 arm and N=1024 pursuit arm must share an exact base field digest. Rows bind archive,
source, selection, initial, pre-projection, base, tail, and final hashes and save four-array fields,
native reconstructions/errors/worst crops, metrics, calls, work, time, memory, projection clauses,
payload keys, pursuit trajectories, and parity receipts.

Run exactly:

```bash
PYTHONPATH=src python scripts/experiments/hier028_residual_pursuit_additive.py \
  /tmp/structsplat-hier028-div2k-valid-20260811/DIV2K_valid_HR \
  results/hier028_div2kvalid8_s160_residual_pursuit_s01_confirmation_2026-08-11 \
  --max-side 160 --seeds 0 1 --device cuda --lpips
```

## Frozen decision

All 64 cells must complete. Every pure endpoint must be finite, exact-count, coefficient-bounded
(`<=16`), exactly four-array, direct one-pass, and internal/cold/repeated parity safe (`<=2e-5`).
Every projection must satisfy all safety clauses or return its incoming field exactly. Pursuit must
retain the base prefix bit-exactly, append exactly 64 rows with fixed geometry, and match its
analytic construction within `2e-5`. Any integrity failure rejects the result.

Compare pursuit N=1024 and cold-control N=1024 to paired `normalized_plain_n640`. A candidate is
quality-capable only if the unchanged HIER-026 clauses all pass:

- mean PSNR at least normalized and every cell within `0.10 dB` of normalized;
- mean MS-SSIM at least normalized minus `1e-4`;
- mean LPIPS at most normalized plus `0.002`, every cell within `+0.01`;
- mean pixel and 7x7 RMSE maxima each within normalized plus `0.005`, every cell within `+0.02`.

Select pursuit only if it passes all numeric and native-visual clauses, its mean PSNR is at least
`0.50 dB` above normalized, no paired PSNR is below normalized, and its displayed pixel/7x7 maxima
do not regress versus its N=960 base in any cell. Native review must find no material new lattice,
checker, ringing, hole, wash, color lobe, blur, or tail speckle. The same-count cold arm diagnoses
allocation; it cannot substitute for pursuit in this frozen task.

A positive result establishes only that normalization is unnecessary for this tested max-side-160
fidelity target when paying 1.60x Gaussian count, 1.50x base Gaussian-row-update proxy, 64 full
residual scans, and tail construction. It does not make additive better at N=640, compare equal
bytes, establish full-resolution/rate/downstream behavior, define a codec, prove novelty, or
authorize a production/default change.

## Non-goals

- No maintained renderer/fitter/pipeline/default/semantic/codec change, opacity, side payload,
  adaptive tail count, alternate scale/count/loss, or threshold relaxation.
- No replay or retuning on these eight images after the one frozen matrix.
- No equal-byte, full-resolution, convergence-speed, downstream, or general-corpus claim.

## Acceptance criteria

- [x] Fresh names, arms, counts, method, schedules, work, metrics, exact command, and gates freeze
      before selected-member extraction or pixel decode.
- [x] Member hashes bind from archive byte streams before selected-image extraction/decode.
- [x] Focused method/driver/decision checks pass (`24 passed`; pursuit invariants, frozen driver
      ladder, and reused HIER-024 projection transaction).
- [x] The complete 64-cell matrix executes once into an immutable checker-valid report.
- [x] Results/visual audit accepts pursuit without tuning: every numeric clause passes and native
      review finds no material new artifact or tail speckle.
- [x] Tasks/docs/ARA and focused/structural/full verification outcomes synchronize; the full gate
      retains only the nine inherited baseline/environment failures recorded below.

## Interfaces touched

One default-off method module, one experiment driver, a narrow report-checker schema, focused
method/driver/decision tests, this task/Index/session brief, and results-driven docs/ARA only. No
maintained public default or pipeline interface.

## Depends on

HIER-027/026/024, FIT-046/048, CORE-009/013, BENCH-002, ADR-0003/0006

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `e9d36d18147bc46072aadf712b63ca63f041e8cd53fc038ea79726eade4b7c5e`

### Handoff log

Producer-frozen confirmation and outcome audit complete without a distinct prospective reviewer.
The immutable report contains 64/64 cells and selects the N=960+64 candidate numerically; an
external native audit closes its intentionally pending visual clause. This is bounded positive
evidence only. The method remains default-off and normalization remains the maintained default
until distinct review and broader/equal-rate production gates.

### Outcome

Accept `residual_pursuit_additive_n1024` for the frozen max-side-160 target. It beats normalized
N=640 by `+1.62037 dB` mean PSNR with a `+1.14979 dB` worst paired gain, improves mean MS-SSIM,
LPIPS, pixel maximum, and 7x7 maximum, and passes every aggregate/per-cell clause. Pixel and 7x7
maxima also improve versus the exact N=960 base in all 16 cells; the same-count cold N=1024 arm
fails the all-cell local clause. Endpoint/base-prefix/payload/parity/work checks pass, and native
review finds no material lattice, ringing, holes, wash, color lobes, blur, or tail speckle. This
shows normalization is unnecessary at the tested fidelity when paying 1.60x rows, a 1.50x base
row-update proxy, and 64 target-known residual scans. It does not establish equal-count/equal-byte,
full-resolution, general-corpus, downstream, codec, novelty, or default superiority. Evidence:
`ara/evidence/hier028-residual-pursuit-additive-2026-08-11/run.md`.

### Handoff

#### Objective

Determine whether 64 deterministic worst-residual rows can turn an exact projected N=960
pure-additive base into a robust N=1024 substitute for normalized N=640 on untouched images.

#### Changes

Added a lazy-torch default-off residual-pursuit module, focused method/decision tests, a frozen
four-arm confirmation driver, report-checker coverage, and synchronized task/docs/ARA evidence.
The maintained renderer, fitter, pipeline, semantics, and defaults are unchanged.

#### Evidence

The 1,334-file source-snapshotted bundle is immutable and checker-valid. Focused pursuit plus
projection tests pass 24/24. All numeric, endpoint, prefix, payload, parity, work, local-
nonregression, and producer native-visual clauses pass on 64/64 cells.

#### Assumptions

The encoder knows the target image and may scan its residual 64 times. Gaussian-row updates and
residual scans are explicit work proxies, not equal FLOPs or bytes. Source selection preceded
decode, but dirty-source producer review keeps the positive result provisional.

#### Uncertainties

The result is max-side 160, eight images, two seeds, one device, unequal rows/work, and not
independently reviewed. The fixed tail count/scale, complete bytes, and full-resolution scaling are
not established beyond the frozen point.

#### Review focus

Check deterministic row-major selection, exact analytic kernel/render parity, bit-exact N=960
prefix, four-array persistence, target-derived state removal, paired local non-regression, native
speckle audit, and the distinction from the failing cold N=1024 control.

#### Protected actions not taken

No post-decode tuning, adaptive tail, threshold relaxation, residual side payload, report mutation,
or maintained renderer/pipeline/default/semantic/codec change.

#### Recommended next action

Obtain distinct review, then test complete-byte and full-resolution scaling on a new source bank
before considering production integration or a renderer-default decision.

### Review

#### Verdict

Provisionally accepted as a bounded positive pure-additive solution

#### Self-reviewed

Yes

#### Correctness

Focused tests, bundle validation, read-only metric recomputation, exact source/base/tail/count/work
hashes, projection rollback, four-array payload audit, and analytic/cold/repeated parity pass. The
frozen decision and external native audit jointly satisfy every declared clause.

#### Evidence quality

The official source binding preceded decode and the report retains every arm, field, trajectory,
metric, and visual. Eight fresh images and two seeds support the bounded result; dirty sources, one
device, and absent distinct review prevent formal acceptance.

#### Simplicity

The endpoint is one ordinary Gaussian sum with 64 fixed-geometry rows and no optimizer tail or
side payload. The encoder-side selection loop is deterministic and default-off.

#### Missing cases

Equal complete bytes, full resolution, broader corpora/devices, adaptive-rate behavior, downstream
response, production latency, and independent protocol/outcome review.

#### Required changes

None for retaining the bounded result. Keep normalization as the maintained row-efficient default.

#### Optional improvements

Measure whether multi-row or region-aware pursuit reduces target scans without reintroducing local
speckle, but only on a separately frozen development task.

#### Verification outcome

The focused HIER-022--028 set passes `83/83`; Ruff and every structural checker pass. The full
portable gate reports `1,952 passed, 26 skipped, 9 failed`. All nine are inherited and outside this
task: one rank-deficient affine condition-number expectation, six subprocess imports resolving an
external installed StructSplat without `PYTHONPATH`, one Torch 2.7 CUDA-property mismatch, and one
pre-existing descriptor-swap race. No HIER-027/HIER-028 test fails.

### Notes

The reversible fallback is omission of the method/driver and retention of the report. HIER-027's
negative cold-capacity result and unchanged gate remain authoritative; this task does not rewrite
either.
