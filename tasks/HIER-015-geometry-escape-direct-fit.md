# HIER-015 — Geometry escape and robust exact-7k dispatch

## Context

HIER-013 showed that the exact-7k contracted additive pipeline does not transfer to the requested
repository image bank: most fields have large cancelling coefficients, the global solve usually
fails closed, and the visible lattice remains.  HIER-014 removes much of that numerical nullspace
on three of four Kodak probes, yet improves PSNR by only `0.0314 dB`, worsens mean LPIPS/local
behavior, and leaves one field at its unsafe fallback.  The explicit-base arm is indistinguishable
from subtraction.  Fixed geometry, rather than the coefficient cap or accumulation algebra, is
therefore the next causal bottleneck.

There are two honest engineering outcomes.  Bounded appearance solves may become useful after the
7,000 contracted supports are allowed to move.  If they still cannot approach an ordinary
fixed-count fit, an unmasked 7k request should use the direct normalized fitter instead of forcing
the hierarchy.  A direct-fit control is required in this task so improvement over a weak HIER-005
baseline cannot be mistaken for a generally useful pipeline.

## Goal

Determine whether zero-centered appearance reconditioning alternated with trust-region all-row
geometry relaxation removes exact-7k contraction artifacts across prospectively hash-selected
natural images; otherwise establish the direct fixed-count normalized fit as the robust engineering
fallback and replay that frozen disposition on `tests/test_images`.

## Non-goals

- Do not retune HIER-013's 16 consumed images or HIER-014's Kodak quartet, raise the coefficient
  cap, append rows, change topology, or call any replay held-out/confirmation evidence.
- Do not silently compare additive and normalized equations as the same representation.  The
  direct normalized arm is an explicit engineering/representation control with separate semantics.
- Do not change `run_pipeline`, `scripts/convert.py`, a maintained default, Field V2 selection, or
  codec policy from this dirty one-seed diagnostic.  BENCH-017/BENCH-020/021 retain those decisions.
- Do not claim novelty for variable projection, block-coordinate descent, trust regions, PCG, or
  fixed-count Gaussian fitting.

## Diagnostic status and source selection

This remains a source-snapshotted development diagnostic because the working tree is dirty and no
distinct outcome-unseen reviewer is available.  Before any pixels or HIER outcomes were viewed,
the development bank was selected mechanically from the available COCO `train2014` directory by
sorting `SHA256("HIER-015-v1:" + basename)` and taking the first four files.  Repository search
found no prior reference to those basenames.  Exact bindings are:

| source | file SHA-256 |
|---|---|
| `COCO_train2014_000000371955.jpg` | `24c86916356edf9c00c17d74cd4f767f5e3fc33f1e5b56b239c05e914d87dfff` |
| `COCO_train2014_000000012379.jpg` | `82fa9d25824b7dd43480b4f64651d3106a91f3b8f7e6d474da221733d289ca90` |
| `COCO_train2014_000000090218.jpg` | `7789b17db08cd18831f615bafa0abf2a602297a6554810ce1c133a214d921c90` |
| `COCO_train2014_000000237851.jpg` | `05451ba10a92a5009889773abda7c042b254c0f2f54d7b1bab9b158154e8172b` |

All are deterministically resized with Pillow LANCZOS to maximum side 512 and use a full-frame
mask, exact N=7,000, seed 0, and required LPIPS.  Every output directory is immutable.

## Phase A — implementation and mechanism

Extend HIER-014's projection with an explicitly intermediate-only bounded-selection mode.  It may
select the lowest-SSE finite coefficient-bounded PCG iterate even when that transient field worsens
the displayed local guard; legacy/default projection selection remains unchanged.  A composite
geometry transaction must never return such an intermediate directly.

For each geometry block, freeze RGB, optimize all means/log-scales/rotations under raw full-frame
L2 with `cuda_additive`, and retain the lowest finite checkpoint.  Use Adam for 400 total steps,
checkpoint every 25, learning rates `0.01/0.006/0.002`, and clamp total drift from the incoming
HIER-005 geometry to `4 px`, `0.7` log-scale, and `0.7 rad`; means also remain inside the canvas
and scales remain finite/positive.  After a block, run the same zero-centered/origin-restarted
96-iteration, ridge-`1e-8`, coefficient-limit-16 solve.  The final composite chooses only among
the original HIER-005 field and round endpoints that do not worsen raw SSE or the displayed
pixel/7x7 normalized violation.  Stage zero therefore remains the exact external fallback.

Compare one `1x400` geometry block with `2x200` blocks at identical geometry-step budget.  Their
difference isolates whether an intermediate appearance reprojection matters.

Correctness tests must cover intermediate versus transactional coefficient selection, exact
fallback, frozen RGB during geometry steps, geometry trust bounds, checkpoint rollback, exact row
count/semantics, bounded final coefficients, final local/SSE transaction, CPU determinism on a
tiny fixture, and no dense pixel-by-row matrix.

## Phase B — frozen COCO development screen

Run five arms from shared source pixels:

1. `h005_control` — unchanged HIER-005 exact-count configuration from HIER-014;
2. `conditioned_transaction` — HIER-014 origin/explicit final transaction;
3. `relax_1x400` — bounded intermediate appearance solve, one 400-step geometry block, final solve;
4. `relax_2x200` — the same 400 geometry steps split by one intermediate appearance solve;
5. `direct_normalized_fixed7k` — an explicit different-semantic control: 7,000 rows from the
   start, `aniso_onedge`/WSE seed 0, feature cap scaled from C12's `12 px @ side 160` to `38.4 px`,
   no topology events, 750 Adam steps, shipped L1+0.3 SSIM objective/LRs, exact `cuda` normalized
   renderer, and same-final-count best-PSNR checkpointing.

All arms record exact configs, source/field hashes, counts, raw and displayed artifact metrics,
PSNR/SSIM/MS-SSIM/LPIPS, coefficient/geometry distributions, checkpoints/operator calls, wall
time, peak CUDA memory, cold/repeated render parity, portable full/error/worst-crop visuals, and
lossless field artifacts appropriate to their semantic type.

The hierarchy candidate is the lower-geometric-mean-MSE relaxation arm, but it passes only if all
four images have exact count, finite fields, coefficient max `<=16`, final additive parity
`<=2e-5`, nonzero geometry movement, no raw MSE/pixel-max/7x7-max regression versus HIER-005,
geometric-mean MSE ratio `<=0.80`, noninferior mean MS-SSIM and LPIPS, median added algorithm time
`<=50%` of contraction time, and no visible lattice/checker/ringing in full frames or worst crops.

The direct-fit engineering fallback passes only if all four fields are exact N=7,000 and finite,
it improves PSNR by at least `2.0 dB` on every image versus HIER-005, mean MS-SSIM and LPIPS are
noninferior, no image worsens displayed pixel or 7x7 maximum, and visual inspection is free of the
HIER lattice.  This gate does not claim normalized semantics are scientifically selected; it says
only whether an unmasked 7k request has a reliable existing path.

Decision order is fixed:

- If a hierarchy arm passes, freeze the better passing hierarchy arm for the consumed-bank replay.
- Otherwise, if the direct arm passes, disposition is `dispatch_unmasked_7k_to_direct_fit`.
- If neither passes, retain HIER-005 and report `no_robust_7k_candidate`; do not tune this bank.

## Phase C — consumed `tests/test_images` replay

Only after Phase B freezes a disposition, run the chosen recipe once over all 16 HIER-013 sources
at maximum side 512, exact N=7,000, and seed 0.  This is reporting-only consumed data.  For a
hierarchy winner, retain HIER-005 plus the selected candidate and apply the Phase-B relative gates.
For direct dispatch, run only the frozen direct arm and produce complete metrics/visuals; compare
against HIER-013 only at the documented aggregate/qualitative level because its local report is not
present in this workspace.  Any replay counterexample rejects “works everywhere” for this bounded
bank; it cannot trigger in-place tuning.

## Acceptance criteria

- [ ] Backward-compatible intermediate projection selection and the fail-closed alternating
      geometry transaction are typed, tested, exact-count, finite, and default-off.
- [ ] The five-arm Phase-B driver binds sources/configs, reuses one contraction per image, isolates
      cell failures, and emits a portable non-claim report with complete telemetry and visuals.
- [ ] The Phase-B gate is evaluated exactly once; only its frozen disposition reaches Phase C.
- [ ] The complete 16-image consumed replay is retained without post-hoc tuning, including every
      counterexample and a native visual review.
- [ ] Results receive an adversarial audit, ARA evidence/disposition records, task/Index/session and
      architecture/design synchronization, applicable bundle checks, and focused/full verification.

## Interfaces touched

`src/structsplat/contraction_refinement.py`, focused tests, one bounded driver under
`scripts/experiments/`, report-schema registration if warranted, `docs/architecture.md`,
`docs/additive_field_v2.md`, ARA evidence/trace records, this task, the Index, and the generated
session brief.  No maintained pipeline entrypoint is changed by the diagnostic itself.

## Depends on

HIER-014, HIER-005, FIT-046, BENCH-017, CORE-013, BENCH-002, ADR-0006

## Reversible fallback

All APIs are default-off.  The composite starts and ends with HIER-005 eligible as checkpoint zero;
legacy projection defaults are unchanged.  A direct arm is a separate returned field, not a silent
semantic conversion.  Removing the new task module/configuration restores current behavior.

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `b1ab01be6d4159c59019e8f0f0275bf43fa61f0b08d0c715f63689f530799183`

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`.  No formal
`### Protocol review` is claimed; the mechanical source selection and pre-outcome task text make
this a stronger diagnostic, not a claim-ready experiment.

Diagnostic note:

The scientific null is that coefficient conditioning plus 400 bounded geometry steps still cannot
close a material fraction of the gap to a direct fixed-count fit.  If that null survives, further
cap/ridge tuning is deprioritized in favor of representation dispatch or a genuinely new basis.

### Handoff

#### Objective

Test bounded additive geometry escape against an ordinary exact-7k normalized control.

#### Changes

Added intermediate-safe projection selection, bounded geometry alternation, the five-arm driver,
focused tests, exact artifacts, and report-schema support.

#### Evidence

The 20-cell bundle at `results/hier015_coco_geometry_escape_2026-08-10` has manifest
`b1ab01be6d4159c59019e8f0f0275bf43fa61f0b08d0c715f63689f530799183`. Relaxation passes scalar
gates but fails native lattice review; direct normalized fitting misses one worst-pixel clause.

#### Assumptions

The direct arm is a deliberately different normalized representation control, not an additive
semantic successor.

#### Uncertainties

One seed, four diagnostic images, dirty snapshotted source, and no distinct reviewer.

#### Review focus

Trust-region rollback, exact count, visual lattice rejection, and the direct-arm local maximum.

#### Protected actions not taken

No maintained dispatch/default, renderer equation, codec, consumed replay, commit, or push changed.

#### Recommended next action

Retain the negative geometry result and isolate the direct normalized local tail on new sources.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

The five-arm driver preserves exact count and separates additive geometry escape from the direct
normalized control. Rollback and scalar gates are fail-closed, and the visually rejected lattice
arm is not promoted despite its numeric scores.

#### Evidence quality

The immutable 20-cell bundle, native visual review, field artifacts, telemetry, and manifest
support a negative diagnostic. One seed, four development images, dirty snapshotted source, and
producer-only review prevent a claim-ready conclusion.

#### Simplicity

The experiment reuses the existing projection and fit machinery and leaves maintained dispatch,
renderer, and codec policy unchanged.

#### Missing cases

No consumed-bank replay was allowed after the fresh-bank gate failed. Broader seeds, devices, and
held-out images remain untested.

#### Required changes

None for retaining this bounded negative result. Distinct review is required before using it in a
publication-facing claim.

#### Optional improvements

Archive an independently reviewed native-resolution contact sheet if the negative is promoted.
