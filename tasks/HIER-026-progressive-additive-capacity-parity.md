# HIER-026 — Progressive pure-additive capacity parity

## Context

HIER-022 through HIER-024 show that changing coverage mass, gauge, or RGB optimization does not
transfer normalized rendering's N=640 advantage into an exact additive endpoint. HIER-025 then
rejects a disconnected 16/624 low-pass/residual basis because proxy-stage fitting creates material
fine-detail blur. Those failures isolate the denominator's practical benefit: it lets irregular,
overlapping Gaussian geometry share colors efficiently. They do not show that a denominator is
mathematically necessary for a high-fidelity image field.

After HIER-025 was sealed, bounded feasibility probes on its already consumed four-image bank
tested this remaining capacity hypothesis. Cold projected additive fields at N=960 beat the
normalized N=640 control on all four seed-0 cells (mean `+1.3415 dB`; minimum `+0.5750 dB`). A
full-target additive N=640 base followed by 256 residual births and 200 joint updates reached N=896
and beat normalized on both seeds of the previously difficult aircraft cell (`+0.4995/+1.1885
dB`). N=832 and N=864 were not robust. These post-hoc probes selected a killing configuration;
they are not retained natural-image evidence and cannot support the outcome.

HIER-025 prospectively selected four official DIV2K validation names without acquiring or opening
their pixels. Its Phase C remained sealed after the HIER-025 failure. This new task inherits that
untouched name binding for a different, already frozen capacity/topology question.

## Goal

Determine whether one cold-replayable, denominator-free, opacity-free additive `GaussianField`
can match normalized N=640 quality on untouched natural images, and measure the Gaussian-count and
fit-work cost without claiming same-count, equal-byte, or codec superiority.

## Endpoint and method contract

Every pure-additive arm persists exactly one ordinary constant-color `GaussianField` and renders
all rows in one `cuda_additive` pass. Only means, log-scales, rotations, and signed RGB
coefficients survive. No normalization denominator, opacity, mass, level, mask, residual image,
target, optimizer, auxiliary RGB, or second-pass metadata is allowed.

The candidate `progressive_residual_n896` is fixed as follows:

1. Initialize exactly 640 rows from the full target using maintained `aniso_onedge`/WSE,
   `flank_offset_frac=0`, feature scale cap 12 pixels, signed bilinear RGB, and the declared seed.
2. Fit that ordinary additive field to the full target for 500 L1 + 0.3 SSIM updates. Select only
   under maintained best-PSNR/final-count checkpoint policy at cadence 25.
3. Render the selected base once and form the signed encoder-side residual `R = I - render(G640)`.
   Initialize exactly 256 additional rows from `R` with the identical initializer family, seed,
   scale cap, and signed bilinear RGB. No residual-only optimization stage is allowed.
4. Append the rows, immediately discard the residual raster, and jointly optimize all 896 rows
   against the original full target for exactly 200 more L1 + 0.3 SSIM updates. No geometry or RGB
   prefix is frozen. Use the same checkpoint policy and cadence.
5. Optionally apply HIER-024's unchanged target-known safeguarded all-row RGB projection. It may
   change RGB only or return its incoming field byte-for-byte in parameter value. Its frozen PCG,
   ridge, coefficient, metric, and local-error transaction remains unchanged.

Learning rates are means `5e-2`, scales `3e-2`, rotations `1e-2`, RGB `3e-2`; support is a hard
three-sigma cutoff with no AA dilation; render chunks are 256. Candidate optimizer work is exactly
`640*500 + 896*200 = 499,200` Gaussian-row updates. Residual selection, checkpoint observers,
projection operator calls, metric calls, wall time, and peak memory are recorded separately.

## Phase A — correctness and procedural killing fixtures

Add a typed default-off fitter and focused CPU/CUDA tests proving:

- exact 640 + 256 = 896 allocation and exact 500/200 attempted-update boundaries;
- the base endpoint supplied by the caller is not mutated and all 896 rows are trainable jointly;
- deterministic finite signed-residual birth construction, including negative RGB coefficients;
- append render parity, returned-field render parity, persistence/cold parity, and CPU/CUDA parity;
- coefficient bound 16, finite gradients/endpoints, deterministic digests, and exact work/call
  accounting; and
- no training-only or source-derived payload survives in the returned field.

Constant, ramp, edge, blob, and texture fixtures are procedural only. They may kill the
implementation but may not select a count, schedule, loss, threshold, or natural-image conclusion.
A Phase-A failure seals Phase B.

## Phase B — frozen untouched DIV2K validation confirmation

Acquire the official validation archive only after this protocol exists:

- URL: `https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip`
- observed content length: `448993893`
- observed last-modified: `Tue, 14 Feb 2017 01:58:09 GMT`
- observed ETag: `"1ac31a65-5487e628d1f06"`
- archive SHA-256: `20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc`

Use the first four names already ranked prospectively in HIER-025 by
`SHA256("HIER-025-confirm-v1:" + filename)` over canonical `0801.png`–`0900.png`. None has been
decoded or visually opened in this lineage:

| rank | file | selection SHA-256 | file SHA-256 |
|---:|---|---|---|
| 1 | `0895.png` | `0644b064658788ac2695cfa2d57d4c2704d3d5e3173f310daf06262914deb703` | `a1c0888648fed4eb909c6e7f5f5db220ae98861294ebfdfa14b2c72567e96b2b` |
| 2 | `0860.png` | `082cd6a3d95e3b16ec770c3502325c1fcb6cc890e9791a7a27b61614e028ef4e` | `eac29d623ecfab9e2299c04b49e5da3f282a576eb7f9107d0b88076c972ac3ef` |
| 3 | `0898.png` | `0b554a43bfb78b6ebda36539d5d3f2cdd1568a394ac430981ee0ac5d96aaab7c` | `4cd6696b8e59615ceacff729181dd9b0cc5ea936ea9a57e089bb1fe4fe87c347` |
| 4 | `0847.png` | `10494b910838e73fad90d013d95d07dfc4ffd618f6416f819d372d9788c6d096` | `ce39eab49b45fc08177556f7c9ae0d0e928e283fb3cd471bddf0fbf17db8ca73` |

Compute and insert the archive and selected member hashes without using an image decoder. Do not
change names, arms, counts, schedules, metrics, or gates after that binding.

The archive was acquired on 2026-08-11 at exactly 448,993,893 bytes. The SHA-256 values above were
computed from the archive and `unzip -p` member byte streams before extraction or image decoding.

Use max-side 160, seeds 0/1, exact owned CUDA renderers, required LPIPS, no opacity, hard
three-sigma support, no AA dilation, and 256-row chunks. Frozen arms are:

1. `normalized_plain_n640`: maintained normalized N=640 fit, 500 updates;
2. `additive_plain_n640`: the exact shared additive N=640 base, 500 updates;
3. `additive_projected_n640`: that endpoint plus the unchanged safe RGB projection;
4. `cold_additive_projected_n896`: ordinary full-target N=896 initialization, 500 updates, then
   safe projection;
5. `progressive_residual_n896`: the exact shared N=640 base plus the frozen 256-birth/200-joint
   procedure;
6. `progressive_residual_projected_n896`: that exact endpoint plus safe projection; and
7. `cold_additive_projected_n960`: ordinary full-target N=960 initialization, 500 updates, then
   safe projection.

The shared base must have the same field digest in arms 2, 3, 5, and 6 before their branch-specific
operations. The projected and unprojected progressive arms must share an identical pre-projection
endpoint. N=896 cold versus progressive isolates staging at equal endpoint count. N=960 is the
predeclared robust capacity control. Rows bind archive/source/selection/initial/base/birth/pre-
projection/final hashes and save all fields, histories, native reconstructions, error maps,
worst-pixel crops, calls, work, time, memory, metrics, safety clauses, and payload/parity receipts.

Run exactly:

```bash
PYTHONPATH=src python scripts/experiments/hier026_progressive_additive_capacity.py \
  /tmp/structsplat-hier026-div2k-valid-20260811/DIV2K_valid_HR \
  results/hier026_div2kvalid4_s160_capacity_s01_confirmation_2026-08-11 \
  --max-side 160 --seeds 0 1 --device cuda --lpips
```

## Frozen decision ladder

All seven arms must complete all eight cells. Every additive candidate must be finite, exact-count,
coefficient-bounded (`<=16`), cold/internal render-parity safe (`<=2e-5`), direct one-pass, and
payload-clean. Every projection must satisfy all HIER-024 safety clauses or return the incoming
field exactly. Any integrity failure rejects the affected rung.

For each projected pure-additive rung, compare paired cells to `normalized_plain_n640`. A rung is
quality-capable only if:

- mean PSNR is at least normalized mean and every cell is within `0.10 dB` of normalized;
- mean MS-SSIM is at least normalized mean minus `1e-4`;
- mean LPIPS is at most normalized mean plus `0.002`, with no cell worse by more than `0.01`;
- mean pixel and 7x7 RMSE maxima are each at most normalized mean plus `0.005`, with no cell worse
  by more than `0.02`; and
- native review finds no material new lattice, checker, ringing, hole, wash, color lobe, or blur.

The progressive mechanism is supported only if projected N=896 is quality-capable and its mean
PSNR is at least cold projected N=896 mean (zero tolerance); otherwise any N=896 success is a
capacity result, not a staging result. The N=960 robust rung additionally requires mean PSNR at
least `0.10 dB` above normalized and no paired cell below normalized. Choose the smallest-count
quality-capable rung; use N=960 only if neither N=896 rung qualifies. If no rung qualifies, retain
the full negative result and do not tune on these pixels.

A positive result establishes only that normalization is not required for fidelity at this tested
resolution and that its N=640 advantage can be exchanged for the measured count/work. It does not
establish equal-count superiority, equal bytes, rate-distortion, full-resolution parity, a codec,
or a production/default change. Additive N=640 is better than normalization only if that exact arm
passes the same quality-capable gate; higher-count success must not be described that way.

## Non-goals

- No maintained renderer/fitter/pipeline/default/semantic/codec change, opacity, hidden residual,
  adaptive count, second decode pass, publication novelty, or HIER-025 Phase-C claim.
- No tuning or replay on these four images after the single frozen matrix.
- No equal-byte, full-resolution, convergence-speed, downstream, or general corpus claim.

## Acceptance criteria

- [x] The question, inherited untouched names, controls, counts, schedules, work units, metrics,
      exact command, and killing rule freeze before Phase-B pixel access.
- [x] Phase-A implementation and focused CPU/CUDA/procedural tests pass.
- [x] Official archive/member hashes bind before any selected image decode.
- [x] The complete 56-cell frozen matrix executes once into an immutable checker-valid report.
- [x] A results/visual audit selects the smallest passing pure-additive rung or records a negative.
- [ ] Tasks/docs/ARA synchronize and focused/structural/full verification outcomes are recorded.

## Interfaces touched

One default-off module under `src/structsplat/`, focused tests, one experiment driver, a narrow
report schema, this task/Index/session brief, and results-driven docs/ARA only.

## Depends on

HIER-025/024/023/022, FIT-046/048, CORE-009/013, BENCH-002, ADR-0003/0006

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `079976d3425ffbc4660a3d89336a82b997a44023b1126189a045b55b570f61d8`

### Handoff log

Fresh confirmation protocol frozen by the producer without a distinct prospective reviewer. Any
result is provisional and must not be promoted as independently accepted evidence.

### Notes

The reversible fallback is omission of the default-off module/driver and retention of the complete
negative or positive report. No maintained behavior changes in this task.

### Confirmation outcome

The frozen 56-cell matrix completed once at
`results/hier026_div2kvalid4_s160_capacity_s01_confirmation_2026-08-11`; its report checker passes
and its immutable manifest SHA-256 is
`079976d3425ffbc4660a3d89336a82b997a44023b1126189a045b55b570f61d8`.

Every integrity clause passes. All pure endpoints contain exactly four arrays, cold-replay in one
additive pass, remain coefficient-bounded, preserve exact shared-base/pre-projection hashes, and
complete the frozen work. Progressive N=896 uses exactly 640+256 rows, 500+200 updates, and
499,200 Gaussian-row updates. Every projection selects safely or returns its incoming field.

The fidelity signal is strong but the declared gate is negative. Mean PSNR is `26.7509`
normalized, `25.9090` projected additive N=640, `27.4206` cold N=896, `27.5048` projected
progressive N=896, and `27.6959` cold N=960. Progressive and N=960 beat normalized by
`+0.75388/+0.94493 dB`; their minimum paired deltas are `+0.04411/+0.35273 dB`, and both improve
mean MS-SSIM and local maxima. Progressive improves mean LPIPS too. However `0860` seed 0 raises
progressive LPIPS by `0.05447`, cold N=960 raises LPIPS there by `0.02910`, and N=960 also breaches
one LPIPS and one pixel-maximum guard. Dense-forest native review confirms material diffuse
directional smear, so no rung is selected and the visual gate does not override the metrics.

After sealing the decision, bounded probes on these now-consumed cells found that ordinary cold
projected N=1088 and N=1152 pass every numeric clause across all eight cells; N=1088 has
mean/minimum PSNR deltas `+1.68200/+0.98761 dB` and worst LPIPS delta `+0.00335`. N=1024 still
fails the `0860` killing cell. These post-hoc probes only select HIER-027's fresh counts and are not
part of HIER-026 evidence. Full details and limitations are in
`ara/evidence/hier026-progressive-additive-capacity-2026-08-11/run.md`.

### Handoff

#### Objective

Determine whether a single pure-additive Gaussian field can exchange measured count/work for
normalized N=640 fidelity on untouched natural images.

#### Changes

Added a default-off 640+256 progressive fitter, explicit stripping to four persisted arrays, 19
focused method/decision tests, a hash-bound seven-arm confirmation driver, and a report checker
schema covering source, count, work, branch, projection, payload, and cold-render integrity.

#### Evidence

The 1,022-file source-snapshotted bundle is immutable and checker-valid. Focused progressive plus
projection tests pass 23/23. All endpoint/procedural gates pass; the perceptual/visual gate fails
on the dense-forest counterexample despite strong PSNR/MS-SSIM/local improvements.

#### Assumptions

Gaussian-row updates are a transparent optimizer-work proxy, not equal FLOPs or equal bytes.
Projection and target-known metrics are separately charged. The source-name selection preceded
decode, but producer-only review keeps the outcome provisional.

#### Uncertainties

The result is max-side 160, four images, two seeds, one device, dirty-source, unequal count/work,
and not independently reviewed. It does not locate a full-resolution or complete-byte exchange
rate.

#### Review focus

Check pre-decode archive/member binding, exact four-array persistence, 640+256 and 499,200 work
accounting, shared base and pre-projection hashes, fail-closed RGB projection, paired LPIPS/local
counterexamples, and the separation of post-hoc count probes from confirmation evidence.

#### Protected actions not taken

No threshold relaxation, in-place replay, maintained renderer/fitter/pipeline/default/semantic/
codec change, formal claim, commit, or push.

#### Recommended next action

On a new untouched source selection, confirm ordinary cold projected additive N=1088 with N=1024
as the boundary control and N=1152 as the robust fallback; keep the exact HIER-026 quality clauses.

### Review

#### Verdict

Provisionally accepted as a near-miss and capacity-threshold result

#### Self-reviewed

Yes

#### Correctness

Focused tests, report-schema validation, exact source/count/work hashes, projection rollback,
payload audit, and cold/repeated parity pass. The immutable decision matches recomputed paired
metrics.

#### Evidence quality

The protocol and official source binding preceded pixel decode and retain every arm, field,
history, metric, and visual. Dirty sources, one device, four images, and absent distinct review
prevent independent acceptance.

#### Simplicity

The endpoint is one ordinary four-array additive field. The rejected progressive fitter remains
default-off; the successor can use the simpler existing cold additive fit without new semantics.

#### Missing cases

Fresh confirmation at N=1088/N=1152, full resolution, more images/devices, complete bytes,
rate-distortion, downstream response, and independent protocol/outcome review.

#### Required changes

None for retaining the HIER-026 result. Do not relax its gate or tune its consumed bank.

#### Optional improvements

If fresh capacity confirmation succeeds, quantify the count/work exchange separately from codec
bytes and test whether perceptual objectives can lower the additive count without adding payload.
