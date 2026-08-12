# HIER-030 — Janelle 7k capacity and contained-mask diagnostic

## Context

HIER-029 applied HIER-028's literal max-side-160 counts (N=640/960/1024) unchanged to a
1200x1038 Janelle raster. The result was visibly under-capacity and its masked arm used masked
loss/selection without containment. It therefore did not test the user's expected approximately
7,000-row regime or the requirement that masked Gaussians remain inside the foreground mask.

This successor preserves HIER-029 as immutable evidence and answers both corrections on the same
exposed raster. It is a single-image, single-seed development diagnostic, not confirmation.

## Goal

Run a proportionally scaled HIER-028 ladder ending at exactly 7,000 Gaussians, paired between
full-frame fitting and hard mask-contained fitting, and produce a portable `index.html` with
metrics, native evaluation-size comparisons, error maps, containment receipts, and execution
errors.

## Frozen diagnostic protocol

- Reuse HIER-029's exact native C0001 JPEG/mask hashes and deterministic max-side-1200 raster
  (1200x1038). Seed 0, CUDA, required LPIPS, 256-row render chunks.
- Scale HIER-028's count ratios by `7000/1024`: normalized N=4,375; projected cold additive
  N=6,562; that exact base plus 438 pursuit rows for N=7,000; and projected cold additive
  N=7,000. The one-row rounding remainder is assigned to the pursuit tail.
- Keep the HIER-028 initializer, L1 + 0.3 SSIM objective, Adam learning rates, 500 updates,
  25-update best-final-count checkpoints, coefficient projection transaction, fixed 0.35-pixel
  pursuit geometry, and coefficient bound. This isolates capacity on the full-frame arm; it is
  not a resolution-density-matched run.
- `full_frame` keeps HIER-028's feature cap, hard support, and ordinary full-image objective.
- `masked_contained` uses masked-density initialization and target, `mask_contain=True`, the
  maintained anisotropic containment certificate, margin 0.75 px, C0 support fade, and
  mask-weighted L1/SSIM. Final log-scales materialize the certified caps before all endpoints are
  reduced to four arrays. Projection uses the same support-fade equation and mask domain.
- Pursuit rows in `masked_contained` may be selected only in the mask eroded by
  `0.75 + 3*0.35 = 1.80 px`; their fixed support is therefore inside the raw mask. No mask or
  scale-cap payload survives the endpoint.
- Every masked row must report all centres inside the raw mask, unit-coverage maximum outside the
  mask `<=1e-7`, and reconstruction maximum outside the mask `<=1e-7`. A failure remains visible
  as an error row and rejects integrity.
- Report common full-frame and black-matted foreground-crop PSNR, MS-SSIM, LPIPS, pixel maximum,
  and 7x7 maximum; count/work/time/memory; fields/histories/hashes; full and foreground errors;
  and direct/repeated parity. Include HIER-029's literal-count metrics as labeled historical
  context, not as rerun cells.
- No outcome-dependent schedule/count/margin/threshold retuning or report overwrite.

Exact command:

```bash
PYTHONPATH=src python scripts/experiments/hier030_janelle_7k_contained_diagnostic.py \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png \
  results/hier030_janelle_c0001_s1200_7k_contained_s0_diagnostic_2026-08-11 \
  --max-side 1200 --seed 0 --device cuda --lpips
```

## Non-goals

- No native-5328 run, approximately 57,600-row max-side-density-equivalent run,
  held-out/multi-image/multi-seed claim,
  equal-byte/rate comparison, optimizer retuning, maintained default change, or publication claim.
- No claim that 7,000 rows reproduce HIER-028's max-side-160 Gaussian-per-pixel density.

## Acceptance criteria

- [x] Focused support-fade pursuit and contained-driver tests pass without changing historical
      default behavior.
- [x] All eight 2x4 cells complete or remain visibly represented as errors.
- [x] Every masked endpoint passes centre, support, reconstruction, count, payload, and parity
      integrity receipts.
- [x] `index.html` includes metrics, full/foreground comparisons, fixed-scale errors, containment,
      provenance, and execution errors; the maintained report checker accepts the bundle.
- [x] Producer visual/results audit and task/docs/ARA synchronization complete; `verify.sh` passes.

## Interfaces touched

One backward-compatible support-fade flag in the default-off residual-pursuit method, a successor
experiment driver, focused tests, the report checker, task/Index/session brief, and result-driven
docs/ARA. No maintained public default.

## Depends on

HIER-029/028/024, CORE-010/011, BENCH-002, ADR-0003/0006/0017/0019/0028

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest
  `80c9c64a3b59730e9ab3fdd4dddbd3b55cfd8f50088054f5416b9764fe83f840`

### Handoff log

Protocol frozen before the successor driver decoded the hash-bound source. Distinct prospective
review was unavailable, so the outcome remains provisionally self-reviewed. The first complete
execution correctly rejected all four masked cells: their centres were inside the mask, but a
restored best checkpoint could retain anisotropic scale caps certified at an earlier position and
leak support by up to `0.003266`. That immutable failed bundle is retained at
`results/hier030_janelle_c0001_s1200_7k_contained_s0_diagnostic_2026-08-11_failed_stale_anisotropic_checkpoint_caps`.
The fitter now forces full cap recertification at terminal and restored-best checkpoints, with a
regression test. The exact preregistered matrix was then rerun from scratch into the evidence-
bearing bundle below; counts, schedule, mask margin, support semantics, and thresholds did not
change.

### Outcome

The user's capacity diagnosis is correct. HIER-029's endpoint was N=1,024, not approximately
7,000. At N=7,000, full-frame pursuit reaches `35.00091 dB`, a `+21.35407 dB` change from the
historical N=1,024 HIER-029 pursuit and `+1.34819 dB` over normalized N=4,375. The global
dot/hole lattice is gone. The separately cold-fitted projected N=7,000 control is slightly better
in PSNR/MS-SSIM (`35.05745 dB`/`0.98046` versus `35.00091 dB`/`0.97901`), while pursuit has
better LPIPS (`0.15497` versus `0.15788`) and much lower pixel/7x7 maxima
(`0.20265/0.11716` versus `0.52445/0.17125`). Residual pursuit is therefore not a clear 7k winner;
ordinary capacity explains most of the repair.

Hard mask containment now works exactly. Across the four masked endpoints, all 24,937 centres are
inside the raw mask, maximum unit support outside is `0.0`, maximum reconstruction magnitude
outside is `0.0`, and every persisted endpoint contains only means, log-scales, rotations, and
RGB. Masked N=7,000 pursuit reaches `20.97452 dB`; cold N=7,000 reaches `21.57449 dB`; normalized
N=4,375 remains best on PSNR/MS-SSIM at `22.78911 dB`/`0.96096`, while cold N=7,000 has best LPIPS
at `0.13494`.

The lower strict-mask aggregate is a boundary-closure failure, not background leakage or a broad
interior capacity failure. A labeled post-hoc distance-transform audit of the manifest-bound raw
arrays finds `94.33%`, `96.27%`, and `95.70%` of total foreground SSE within four pixels of the
mask edge for pursuit, cold N=7,000, and normalized respectively. Their greater-than-four-pixel
interior PSNRs are `32.893`, `35.313`, and `35.908 dB`. Every uncovered foreground pixel is
within `3.17 px` of the boundary. HIER-029 could use support across the boundary and therefore
reported a higher masked score while violating the containment requirement.

Visual review agrees with the metrics. The 7k full-frame fields remove the pervasive HIER-029
stipple but remain visibly smooth on the face, lace, hair, and machinery; pursuit adds a few small
bright tail speckles along the diagonal rail. Masked placement visualizations put every centre
inside the silhouette and show clean black outside support. Their fixed-scale error maps place the
dominant residual on the silhouette and fine foreground detail, with no report/reconstruction
mismatch.

Seven thousand rows still represent only one Gaussian per about 178 full-frame pixels. Preserving
HIER-028's max-side-160 final row density at 1200 would require approximately 57,600 rows. This
diagnostic establishes that N=1,024 was grossly under-capacity and that exact containment works;
it does not establish native-camera or density-matched quality.

Evidence:
`ara/evidence/hier030-janelle-7k-contained-mask-2026-08-11/run.md` and
`results/hier030_janelle_c0001_s1200_7k_contained_s0_diagnostic_2026-08-11/index.html`.

### Review

#### Verdict

Complete positive capacity/containment diagnostic; boundary-aware masked allocation remains open

#### Self-reviewed

Yes

#### Correctness

The maintained checker accepts the 268-file bundle. All source/raster/count/work/base-prefix,
projection, pursuit, endpoint-purity, containment, and direct/cold/repeated parity receipts pass.
The failed first run exposed rather than hid the stale-cap defect, and the new focused regression
forces terminal and restored-best anisotropic cap recertification.

#### Evidence quality

The report has complete machine-readable rows, fields, histories, raw reconstructions, placement,
coverage, fixed-scale errors, errors ledger, and hashes, but it is one exposed image/seed/device
from dirty executed sources. The post-hoc boundary decomposition is descriptive and was not a
frozen selection gate. `overall_pass` in this diagnostic report means integrity/containment pass,
not scientific-quality or promotion pass.

#### Simplicity

Support fade and mask selection remain opt-in encoder controls, and all endpoints reduce to the
ordinary four-array representation. The fitter correction enforces an existing containment
contract; it does not add a stored payload or change uncontained defaults.

#### Missing cases

Native 5328x4608, approximately 57,600-row density parity, boundary-aware initialization or
densification, a resolution-aware optimizer schedule, new images/seeds/devices, equal bytes/rate,
downstream behavior, and distinct review.

#### Required changes

Do not describe HIER-029 as a 7k test, do not treat loss-only masking as containment, and do not
promote pursuit over cold additive from this run. Keep the maintained renderer/pipeline defaults
unchanged.

#### Optional improvements

Freeze a new-source boundary-closure successor that explicitly allocates tangent-aligned rows to
the eroded silhouette band, and separately test density-matched full-frame capacity.

### Handoff

#### Objective

Determine whether the poor HIER-029 image was chiefly a count error and enforce the user's rule
that masked Gaussians and their rendered support stay inside the mask.

#### Changes

Added an opt-in support-fade pursuit equation, the 7k paired driver/report/checker, focused tests,
and forced terminal/restored-checkpoint anisotropic cap recertification. No maintained default or
endpoint schema changed.

#### Evidence

The corrected 268-file bundle is checker-valid; all eight rows completed and every masked
containment value is exactly zero outside. Native report-size reconstructions, placement sheets,
coverage maps, and fixed-scale errors were producer-reviewed.

#### Assumptions

The requested full-resolution comparison follows the repository's max-side-1200 Janelle regime.
The mask is known at encoding time, and exact three-sigma support containment is required.

#### Uncertainties

One exposed 1200x1038 raster/seed/device, dirty sources, producer-only review, 7k below density
parity, boundary-dominated foreground score, and no codec/downstream evidence.

#### Review focus

Check terminal cap refresh, four-array materialization, all-zero outside-support receipts, masked
pursuit erosion, common metric domains, the boundary-band decomposition, and whether the visual
smoothness supports the density caveat.

#### Protected actions not taken

No rewrite of either immutable run, outcome-dependent threshold/margin/count/schedule retune,
native-5328 execution, 57.6k run, default change, or claim promotion.

#### Recommended next action

Obtain distinct review. If higher masked fidelity is the priority, preregister boundary-aware
contained allocation on a new view before spending on a native-resolution or 57.6k full-frame run.
