# HIER-024 — Gauge-geometry safeguarded appearance projection

## Context

HIER-023 fixes HIER-022's gauge confound and reaches ordinary additive quality within `0.0326 dB`
after only 250 exact-additive steps, with better mean perceptual/local metrics and AUC. It still
retains none of normalized rendering's `0.6648 dB` fixed-count advantage. Seven of eight endpoints
select the final step, so a coefficient-optimization explanation remains possible, but merely
lengthening the consumed schedule would confound quality with extra work.

HIER-012--014 already provide a matrix-free safeguarded all-row direct-additive RGB solve. Their
broad-bank result on contracted HIER-005 geometry was mostly negative, but they did not compare
ordinary-additive and exact-normalized-warm-start geometries after the HIER-023 path. Applying the
same fixed solve to both is the cheapest discriminator before changing basis, topology, or count.

## Goal

Determine whether HIER-023's remaining normalized-quality gap is primarily suboptimal additive RGB
coefficients or the fixed direct-additive basis geometry. Retain a pure, cold-replayable,
opacity/mass/denominator-free Gaussian sum only when a frozen target-known transaction is safe.

## Method contract

- Fit identical `normalized_plain`, `additive_plain`, and HIER-023 no-reset unit-gauge fields.
- Adapt each additive endpoint exactly to `ObservationField2D`, mark all rows trainable and none
  protected, and run the existing matrix-free direct-additive PCG with tolerance `1e-6`, at most
  48 iterations, ridge `1e-8`, coefficient limit 16, input-centered regularization/start,
  explicit frozen base, no unsafe stage-zero reconditioning, and transaction selection.
- Project both ordinary-additive and unit-gauge endpoints. Geometry, support, count, filtering,
  and coefficient domain stay fixed; only RGB coefficients may change.
- A second frozen target-known safety transaction chooses the projection only when it is finite,
  bounded, strict-lower raw MSE, MS-SSIM no lower by more than `1e-5`, LPIPS noninferior, and
  displayed pixel/complete-7x7 maxima noninferior within `1e-6`. Otherwise it returns the incoming
  field byte-for-byte. This selection work and every proposal metric are charged and logged.
- Persist only a plain additive `GaussianField`. Projection matrices, target pixels, optimizer
  state, denominator, mass, opacity, auxiliary RGB, or selection residuals cannot enter the field.
- No maintained renderer, fitter, conversion, semantic, codec, or pipeline default changes.

## Phase A — correctness

Add a narrow wrapper and tests proving exact Gaussian/Field-V2 adaptation, geometry/count
immutability, pure-additive cold parity, bounded fail-closed projection, deterministic CPU behavior,
and owned-CUDA parity. Test the safety selector on accepted and rolled-back metric fixtures, including
nonfinite and coefficient-bound failures.

## Phase B — frozen development diagnostic

Exclude HIER-023's four selected filenames, rank the remaining eight repository DIV2K filenames by
`SHA256("HIER-024-v1:" + filename)`, and bind the first four before opening pixels. Every file is
historically consumed development data; this is only a new HIER-023-distinct selection.

| rank | file | selection SHA-256 | file SHA-256 |
|---:|---|---|---|
| 1 | `0002.png` | `1bcbb155cb0655237ab13649cb130bcc5e67c77a7cecd31e556967679a67b0e0` | `82325cea74c2cd4681f69a10e36ba15c896d99ec47dc2c687ef07f7497781e09` |
| 2 | `0268.png` | `3869b823b071815d2dbfd4a2fc859c959fbe1338e38001050548c28f65948065` | `455a05afcc60e0638259bb6dd98018606786cd73ee7118049cff94b48b5d4e7b` |
| 3 | `0800.png` | `546604fa43486d31bebebcc71c956629e6e14dbb73ab06f10648bb8d1112e6de` | `eb6df5bfeacd04334062b6103f6ee8f33af1abd3e1375a7f2c2a4831fa701221` |
| 4 | `0571.png` | `a7de9fdb532299ab993d55457b0104dcc124e29ec0e2f6cac4c5c74730996fd8` | `6de58e0706300b3496f538dca3b80d478062f4c4396990b3b5e6479300ed71ef` |

Use max-side 160, N=640, seeds 0/1, 500 attempted fit steps, exact owned CUDA renderers, required
LPIPS, identical `aniso_onedge`/WSE initial fields, 12-pixel feature caps, L1 + 0.3 SSIM, and
25-step logging. Unit gauge keeps HIER-023's 175/75/250 no-reset schedule.

Frozen report arms:

1. `normalized_plain`;
2. `additive_plain`;
3. `additive_projected_safe` — same ordinary-additive field plus projection/safety transaction;
4. `gauge_locked_no_reset`;
5. `gauge_projected_safe` — same unit-gauge field plus projection/safety transaction.

The candidate is `gauge_projected_safe`; `additive_projected_safe` is the causal coefficient-solve
control. Rows bind incoming/proposal/final field hashes, selected/rollback reason, PCG residuals and
operator counts, raw/display metrics for all three states, fit/projection/metric/render work,
trajectory/AUC, coefficients/coverage, payload keys/parity, and native full/error/worst-crop assets.

The bounded mechanism passes only if:

- all eight candidate endpoints are finite exact N=640 additive fields, coefficient maximum
  `<=16`, cold parity `<=2e-5`, unchanged geometry/count/support, and contain no opacity, mass,
  denominator, optimizer, target, or auxiliary RGB payload;
- each projected arm either satisfies every frozen safety clause or returns its incoming field
  exactly; no failed/unsafe proposal is hidden;
- the unit-gauge hold stays within `0.05 dB` of matched normalized in every cell;
- candidate mean PSNR is at least `0.10 dB` above `additive_projected_safe` and closes at least half
  of any positive `normalized_plain - additive_projected_safe` gap;
- candidate mean LPIPS/MS-SSIM/pixel/7x7 are noninferior to the projected additive control, with no
  per-cell LPIPS regression above `0.01` and no local-maximum regression above `0.005`;
- the projection's mean PSNR gain over its incoming field is at least `0.05 dB` larger for gauge
  geometry than for ordinary-additive geometry, and at least four gauge cells select projection;
- candidate PSNR-AUC through the 500-step fit exceeds additive; projection and selection work are
  reported separately rather than inserted into attempted-step AUC; and
- native review finds no lattice, checker, ringing, hole, wash, or material new blur.

If the gate fails, fixed-geometry coefficient optimization is rejected as the missing mechanism.
Any successor must change basis geometry/topology/count under a new task/output/data selection; no
HIER-024 cell or threshold may be tuned.

Intended command:

```bash
python scripts/experiments/hier024_gauge_geometry_projection.py \
  tests/test_images/DIV2K_train_HR \
  results/hier024_div2k4_s160_n640_i500_s01_diagnostic_2026-08-11 \
  --max-side 160 --budgets 640 --seeds 0 1 --iters 500 \
  --arms normalized_plain additive_plain additive_projected_safe \
    gauge_locked_no_reset gauge_projected_safe --device cuda --lpips
```

## Non-goals

- No retuning HIER-022/023, new projection solver, topology/count change, maintained integration,
  semantic/codec/rate/default/downstream claim, equal-work claim, or novelty claim.
- Do not call the historically consumed filename selection held out or independent confirmation.

## Acceptance criteria

- [x] Typed wrapper and focused CPU/CUDA projection/safety tests satisfy Phase A.
- [x] Data selection, hashes, solver, transaction, controls, and gates freeze before pixel access.
- [x] The complete frozen matrix executes once into an immutable checker-valid portable report.
- [x] Adversarial results/visual audit and scoped docs/ARA disposition distinguish coefficient
      optimization from basis geometry without post-hoc rescue.
- [ ] Focused and structural checks pass; full verification and inherited failures are recorded.

## Interfaces touched

One default-off wrapper under `src/structsplat/`, focused tests, one driver, narrow report schema,
this task/Index/session brief, and results-driven docs/ARA records only.

## Depends on

HIER-023/014/013/012, FIT-046, CORE-013, BENCH-002, ADR-0003/0006

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `f761b2834aa394d5a6c3af5648460ac0b67ea108d252d6ce03a487f06f738b9a`

### Handoff log

Dirty-source consumed-development diagnostic; no formal prospective or independent review claim.

### Execution amendment — invalid harness run retained

The first natural command wrote
`results/hier024_div2k4_s160_n640_i500_s01_diagnostic_2026-08-11` and is retained unchanged as an
invalid harness run. All paired fits, projections, selection metrics, and visual assets executed,
but all 40 rows failed final emission because the reused HIER-023 cell writer indexed its own
four-name selection-hash map before HIER-024 could replace that field (`KeyError` for every new
filename). No metric row or decision from this output is method evidence.

The repair makes that reusable metadata lookup optional; HIER-024 still writes and verifies its own
frozen selection hash. It also copies the read-only projection raster before metric conversion to
avoid a PyTorch writability warning. No source, method, fit, projection, transaction, arm, gate, or
threshold changes. The complete matrix reruns once into
`results/hier024_div2k4_s160_n640_i500_s01_diagnostic_rerun1_2026-08-11`; the invalid bundle is not
resumed, overwritten, selected from, or cited.

### Notes

The reversible fallback is omission of the wrapper/driver and retention of ordinary additive or
normalized controls. Existing projection and renderer implementations remain unchanged.

### Result and disposition

The valid repaired 40-row bundle is
`results/hier024_div2k4_s160_n640_i500_s01_diagnostic_rerun1_2026-08-11`; its report checker
passes. Ordinary-additive and unit-gauge projection gain `0.12996/0.17191 dB`, respectively. The
candidate ends only `0.01046 dB` above projected additive, closes 1.91% of the remaining normalized
gap, and fails mean/per-cell local guards. Two hold cells narrowly exceed `0.05 dB`. Both projected
arms select 7/8 proposals and roll back the other field exactly; endpoint integrity and safety pass.

Producer native review finds only the common N=640 blur and seed-sensitive support-placement error,
not a new lattice/checker/ring/hole/wash artifact. The fixed-geometry coefficient explanation is
therefore rejected. Keep the wrapper default-off and move any successor to a new basis/topology
task and new data selection. Full receipts and limitations are in
`ara/evidence/hier024-gauge-geometry-projection-2026-08-11/run.md`.

### Handoff

#### Objective

Determine whether HIER-023's remaining normalized-quality gap is caused by suboptimal additive RGB
coefficients or by the fixed Gaussian basis geometry.

#### Changes

Added a default-off exact `GaussianField` projection adapter, fail-closed metric transaction,
four focused CPU/CUDA tests, a hash-bound five-arm 40-cell driver, report validation, and synced
task/docs/ARA records. The maintained renderer, fitter, field semantics, codec, and defaults do not
change.

#### Evidence

The repaired bundle has manifest
`f761b2834aa394d5a6c3af5648460ac0b67ea108d252d6ce03a487f06f738b9a` and passes the report checker
with `--allow-dirty`; the first invalid harness output remains intact and excluded. The same solve
gains `0.12996/0.17191 dB` on additive/gauge geometry, but candidate advantage is `0.01046 dB`,
gap closure is 1.91%, and local gates fail. All transaction, geometry, payload, coefficient, and
parity checks pass; both projected arms select 7/8 cells and roll back the other exactly.

#### Assumptions

The target-known projection is a diagnostic coefficient oracle, not a decoder stage. Equal fit
steps are not equal work, and all selected DIV2K files are historically consumed development data.

#### Uncertainties

The screen is small-resolution, N=640, two seeds, one device, dirty-source, and producer-reviewed.
It does not rule out other pure-additive geometries, topology changes, counts, or optimizers.

#### Review focus

Exact geometry preservation, direct additive cold parity, transaction fail-closure, proposal-work
accounting, invalid-run exclusion, the gauge-versus-additive gain attribution, and restraint of the
conclusion to the tested fixed bases.

#### Protected actions not taken

No solver or threshold retune, invalid-output reuse, maintained default/semantic/codec change,
unpriced residual, representation-limit or novelty claim, unrelated baseline repair, commit, or push.

#### Recommended next action

Freeze a new exact-count pure-additive basis experiment with counted broad low-frequency Gaussian
carriers and anisotropic residual/detail rows on a new mechanically selected image bank.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

Focused CPU/CUDA tests prove adaptation, geometry/count immutability, cold parity, deterministic
projection, bounded selection, and exact rollback. The checker-valid report confirms all endpoint
and transaction integrity clauses.

#### Evidence quality

The repaired 40-row matrix is complete, hash-bound, immutable, multi-metric, and visually audited.
The invalid first run, dirty source, consumed data, producer review, and target-known work are explicit.

#### Simplicity

The wrapper reuses the existing solver and changes only RGB on a fixed field. Its negative causal
answer removes coefficient tuning from the successor search space.

#### Missing cases

Distinct images, larger counts/resolutions, equal work, independently reviewed replay, alternative
basis geometry/topology, downstream utility, complete rate, and production performance remain absent.

#### Required changes

None for retaining the negative diagnostic. Do not interpret it as proof that normalization is
universally necessary.

#### Optional improvements

Run the separately frozen multiscale pure-additive basis discriminator; do not tune HIER-024's
solver, fields, or thresholds.
