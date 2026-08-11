# HIER-016 — Tail-safe normalized exact-7k refinement

## Context

HIER-015 establishes two independent facts on four prospectively hash-selected COCO images.  First,
bounded appearance/geometry alternation makes HIER-005 numerically much better (the selected
`2x200` arm gains `3.658 dB` and reduces geometric-mean MSE to `0.4307`) but leaves an unmistakable
Gaussian lattice in every full-frame visual.  It therefore fails the mandatory visual gate and
cannot reach the consumed replay.  Second, the existing direct normalized exact-7k fitter is
visually clean and averages `+12.061 dB`, `+0.2859` MS-SSIM, and `-0.4802` LPIPS versus HIER-005,
but one high-contrast image has a single displayed-pixel maximum `0.0562` worse than HIER-005 even
while its worst 7x7 patch improves.  The direct arm consequently fails its frozen literal maximum
gate as well.

The shipped direct objective is L1 plus SSIM, which intentionally spends little gradient on one
isolated extreme.  At fixed normalized-renderer geometry, RGB is the only remaining linear
appearance degree of freedom.  A short color-only tail objective can test whether that isolated
failure is repairable without reopening geometry, topology, initialization, or the 750-step fit.

## Goal

Determine prospectively whether a short, fail-closed, fixed-geometry RGB tail refinement removes
normalized exact-7k worst-pixel counterexamples while preserving the direct fitter's global,
perceptual, local-patch, persistence, and visual advantages; if so, replay the frozen recipe on the
four consumed HIER-015 sources and all 16 consumed `tests/test_images` sources without retuning.

## Non-goals

- Do not inspect the new source pixels before this protocol and its bindings are recorded; do not
  tune on HIER-015's four images or `tests/test_images`.
- Do not change HIER-005, the shipped fitter, normalized renderer, initializer, topology, row
  count, codec, maintained defaults, or dispatch policy from this dirty one-seed diagnostic.
- Do not add residual rows, move geometry/opacities, blend additive and normalized fields, weaken
  the HIER-015 local gate, or describe a one-pixel maximum as a lattice artifact.
- Do not claim novelty for hard-example mining, top-k loss, color-only refinement, safeguarded
  checkpoint selection, or block-coordinate optimization.

## Diagnostic status and prospective bank

This remains dirty-source development evidence without an outcome-unseen distinct reviewer.  Before
any selected pixels were opened, all available COCO `train2014` basenames not already referenced
in the repository were sorted by `SHA256("HIER-016-v1:" + basename)` and the first four were bound:

| source | selection SHA-256 | file SHA-256 |
|---|---|---|
| `COCO_train2014_000000229559.jpg` | `00015a0657e9b73b9234a76e72fde86f4ab361f0f1e1423a9603165973780c1c` | `12909150daa0ca6162a5d2fa7cd7c87b5526c85ca583563952eb06d241397972` |
| `COCO_train2014_000000160926.jpg` | `00016d4361597b41788f3f0ae46c1a771633f453754701f9269c55357a92d4cc` | `0de693701b819d1a58fe8c4a84745029e52ccea8010ef637bee2181a36a73321` |
| `COCO_train2014_000000380591.jpg` | `0001e9e9b17208099d3b766d6f8ed96323844fe1de3320b5aace17fa585eb438` | `1a534dc62ff9d0c91bfdf68fcae720e0e9ee234120fd6b8e1b2be98e38f78511` |
| `COCO_train2014_000000198396.jpg` | `00021264a64ee08587d2fc0d4c330674dd999c5aca64524bc9b0957e958be088` | `c61f811798a871c8867c524f51ddc32f40ceff7f81bc6859c4da91dd5d27e0dd` |

All development images use deterministic Pillow LANCZOS maximum-side-512 rasters, full-frame masks,
exact N=7,000, seed 0, required LPIPS, and immutable output directories.

## Phase A — method contract

Add a default-off normalized color-tail primitive.  It accepts a fitted constant-color
`GaussianField`, target, bool mask, and the exact render `FitConfig`; clones all state, marks only
RGB colors trainable, and uses Adam for 100 steps at LR `0.01`.  Means, log-scales, rotations,
opacities, scale caps, background flags, affine gradients, and covariance-filter metadata stay
bit-exact.  Colors are clamped to absolute value 8 and to total drift 1 from the incoming fit.

For active-pixel channel-mean squared error `e`, optimize
`mean(e) + 4 * mean(top_k(e))`.  Compare frozen tail fractions `0.01` and `0.001`; this is a
mechanism comparison fixed before outcomes, not a ladder.  Checkpoint every five steps and include
the exact incoming field at step zero.  A candidate is eligible only when finite/bounded and no
worse than step zero in raw SSE or exact displayed-8-bit worst-pixel and complete-7x7 maxima.
Among eligible checkpoints choose lexicographically by displayed pixel maximum, displayed 7x7
maximum, raw SSE, then earlier step.  If none improves that key, return the bit-exact input.

Correctness tests cover validation, exact input fallback, RGB-only mutation, color trust bounds,
raw/display transaction safety, checkpoint rollback, count/metadata preservation, cold parity,
and deterministic CPU behavior on a tiny fixture.  The implementation must not materialize a
dense pixel-by-row matrix.

## Phase B — frozen development screen

Run four arms from shared source pixels:

1. `h005_control` — exact HIER-015 HIER-005 configuration, reused once per image;
2. `direct_normalized_fixed7k` — exact HIER-015 direct configuration: `aniso_onedge`/WSE,
   feature cap 38.4, no topology, 750 Adam steps, L1+0.3 SSIM, `cuda`, and same-final-count
   best-PSNR checkpointing;
3. `tail_top1pct` — shared direct field plus the frozen 1% tail transaction;
4. `tail_top0_1pct` — shared direct field plus the frozen 0.1% tail transaction.

Every cell records exact configs/source and field hashes, raw/full/perceptual/display-local metrics,
counts, color distributions and drift, checkpoint history, time/memory, cold/repeated render parity,
lossless fields, and full/error/worst-crop visuals.  Cell failures are isolated and retained.

A tail arm passes only if all four pairs are complete, exact-count, finite, non-color bit-exact,
within both color bounds, cold/repeated parity `<=2e-5`, and transaction-safe; each image gains at
least `2 dB` versus HIER-005 and does not worsen its displayed worst pixel or 7x7 maximum; mean
MS-SSIM and LPIPS are noninferior to HIER-005.  Relative to the shared direct input, no image may
worsen raw MSE beyond `1e-8` relative tolerance or either displayed local maximum, mean MS-SSIM
may fall by at most `0.001`, mean LPIPS may rise by at most `0.002`, and at least one of four cells
must select a nonzero refinement.  Full frames and worst crops must remain free of lattice,
checker, ringing, or new high-contrast color artifacts.

Among visually passing tail arms choose the one with the smaller maximum across-image
`tail_pixel_max / h005_pixel_max`; break ties by geometric-mean MSE versus direct and then select
`tail_top0_1pct` as the narrower intervention.  If neither passes, retain the direct fitter only as
a strong control and stop; do not tune this bank or access Phase C.

## Phase C — frozen reporting replays

Only after Phase B freezes one tail recipe, run direct checkpoint zero and the selected repair once
on both consumed banks:

- the four exact HIER-015 COCO bindings, where the frozen repair must eliminate the known direct
  worst-pixel regression versus the already-recorded HIER-005 rows without losing its 2 dB margin;
- all 16 exact HIER-013 `tests/test_images` bindings, reporting direct-versus-tail metrics and
  visuals without retuning or a held-out/confirmation claim.

On both banks require complete exact-count finite fields, non-color identity, parity, no raw MSE or
displayed local regression versus direct checkpoint zero, and lattice-free visuals.  Any miss is a
retained counterexample and rejects the bounded “works everywhere” statement; it cannot alter the
frozen recipe.

## Acceptance criteria

- [ ] Typed, tested, default-off RGB-tail refinement is count-neutral, metadata-preserving,
      bounded, deterministic on CPU, and fail-closed in raw/display domains.
- [ ] The source-bound four-arm driver isolates failures and emits complete reproducible telemetry,
      lossless fields, histories, and portable visual evidence.
- [ ] Phase B is evaluated exactly once; only a numerically and visually eligible frozen recipe
      may reach either consumed replay.
- [ ] Conditional replays retain every row/counterexample and never feed outcomes into tuning.
- [ ] Results receive adversarial audit, ARA evidence/disposition, task/Index/session and design
      synchronization, applicable bundle checks, focused tests, and full verification.

## Interfaces touched

`src/structsplat/normalized_refinement.py`, focused tests, one bounded driver under
`scripts/experiments/`, `docs/architecture.md`, ARA evidence/trace records, this task, the Index,
and the generated session brief.  No maintained pipeline entrypoint changes.

## Depends on

HIER-015, HIER-005, FIT-005/046, CORE-013, BENCH-002, ADR-0006

## Reversible fallback

The primitive is default-off, clones its input, and includes that exact input as checkpoint zero.
Removing the module/driver restores current behavior; no production consumer is added.

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `94e916d32a71a3a234dde1be3e4958cc46684a321f4f831df0378cf052b760cd`

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`.  No formal protocol
review is claimed; mechanical source selection and pre-outcome freezing strengthen only this
diagnostic.

Diagnostic note:

The causal hypothesis is narrow: normalized geometry is already visually adequate at exact 7k,
and L1+SSIM leaves a sparse high-contrast residual tail that a fixed-geometry RGB step can reduce.
Failure would deprioritize optimizer-tail repair and point instead to support allocation or a
different direct-fit objective selected on another fresh bank.

### Handoff

#### Objective

Test whether a fail-closed fixed-geometry RGB tail repairs direct exact-7k local maxima.

#### Changes

Added the default-off normalized RGB-tail primitive, two frozen tail fractions, focused invariant
tests, a source-bound driver, and complete persistence/visual telemetry.

#### Evidence

The recovered 16-cell bundle at `results/hier016_coco_normalized_tail_recovery_2026-08-10` has
manifest `94e916d32a71a3a234dde1be3e4958cc46684a321f4f831df0378cf052b760cd`. No arm passes; the 1%
tail returns step zero throughout and the 0.1% arm does not repair the known local ratio.

#### Assumptions

Checkpoint-zero rollback is the authority; fixed geometry isolates optimizer-tail adequacy.

#### Uncertainties

One seed/device and four development images; no distinct protocol or outcome review.

#### Review focus

Metadata/geometry identity, color bounds, raw/display rollback, and the no-replay stop rule.

#### Protected actions not taken

No topology, renderer, objective, maintained default, consumed bank, commit, or push changed.

#### Recommended next action

Retain the negative control and test support/normalization coverage on a disjoint bank.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

The fixed-geometry RGB tail is default-off, bounds colors, preserves geometry/metadata, and uses
checkpoint-zero rollback as the returned authority. Neither frozen fraction repairs the known
local failure, so the no-replay decision follows the protocol.

#### Evidence quality

The recovered 16-cell bundle contains exact artifacts, synchronized visual/metric telemetry, and
a stable manifest. It is a four-image, one-seed development diagnostic without independent
protocol or outcome review.

#### Simplicity

The method isolates one optimizer-tail hypothesis without adding topology or changing the
normalized renderer.

#### Missing cases

The consumed bank, additional tail fractions, other objectives, and more seeds/devices were not
run because the frozen fresh gate failed.

#### Required changes

None for retaining the negative control. Distinct review is required for any stronger claim.

#### Optional improvements

Use this bundle only as a baseline if a future support-allocation method is tested prospectively.
