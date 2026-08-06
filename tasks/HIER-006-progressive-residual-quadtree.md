# HIER-006 — Parent-preserving progressive residual quadtree

## Context

HIER-005 contracts an implicit pixel field downward and exposes a 4k artifact floor on the current
Janelle diagnostic. Its terminal fixed-geometry rescue improves average fidelity but does not
remove the worst local defect. The user proposes reversing the construction direction: begin from
a coarse quadtree field, keep its parents, and activate child-layer Gaussians only where current
error is high. Signed additive child coefficients then refine the already reconstructed signal.

The maintained normalized pyramid is a related control, not this method: it globally allocates a
level fraction and refits a normalized field. HIER-006 instead tests a direct-additive,
quadtree-addressed, parent-preserving residual prefix whose geometry is reconstructible from the
shared mask and split tree.

## Goal

Implement a default-off NumPy-first reference that constructs a progressive direct-additive field
from coarse mask-moment parents and selectively appended residual children, optimizes only each new
child layer against a frozen prefix, stops on an explicit local-artifact gate or row cap, and emits
a cold-rendered diagnostic report with the complete hierarchy trajectory.

## Method contract

- Start at a declared quadtree level. Every mask-intersecting cell contributes one deterministic
  parent Gaussian whose mean/covariance is the moment of its active pixel centers plus the declared
  leaf variance.
- A split retains the parent and appends one signed residual Gaussian for every mask-intersecting
  child cell. Its RGB coefficient starts from that child's mean current residual, while the exact
  unchanged prefix remains the checkpoint comparator. Child geometry is derived by the same rule;
  it is not optimized or stored as an independently selected spatial sample in the conceptual
  structured representation.
- The current prefix is immutable. Only the most recently appended child RGB coefficients are
  optimized, so every accepted stage is a valid independently renderable prefix and earlier LODs
  stay bit-exact.
- Candidate parents are ranked by mask-aware Gaussian-smoothed residual energy summed over the
  parent cell and divided by the number of appended children. Ties are deterministic. A stage
  greedily selects the highest-ranked frontier splits that fit its row allowance and global cap.
- New-layer optimization minimizes masked MSE plus a weighted worst-pixel tail mean. The unchanged
  prefix competes with checkpoints ordered lexicographically by normalized raw worst-pixel/7x7
  patch violation, then masked SSE. Numerically equal violations use a documented float32
  roundoff tie band before the SSE tie-break. A non-improving layer rolls back completely and its
  proposed parents are blocked for that run.
- The displayed 8-bit output remains final authority. Stop at the first stage satisfying the
  declared pixel-max and 7x7-patch-max gate, otherwise fail closed at the cap/frontier.
- Output semantics remain direct signed additive, peak-one AABB support. This method does not enter
  the maintained pipeline or change Field V2/default decisions.
- Full field bytes remain authoritative for the implemented container. Coefficient/tree-only byte
  values are explicitly labeled structural proxies until COMP-013 implements and measures a cold
  self-contained stream.

## Non-goals

- Claiming novelty, actual compression, production readiness, or superiority from one exposed
  resized image.
- Optimizing parent geometry/coefficients after acceptance, learning the split policy, or adding a
  neural encoder.
- Replacing HIER-005, the maintained normalized pyramid, renderer defaults, or codec policy.
- Consuming a held-out/confirmation split or promoting a semantic/default decision.

## Acceptance criteria

- [x] Typed deterministic APIs validate the image/mask/config and expose hierarchy stages, exact
      row counts, stop reason, immutable-prefix checks, field bytes, and structured byte proxies.
- [x] Coarse and child geometry follows the mask-moment quadtree rule for odd shapes, sparse masks,
      one-pixel cells, and numerical degeneracies without importing torch at module import time.
- [x] Parent-preserving splits append complete mask-present child groups, never exceed the cap,
      rank error per appended row deterministically, and roll back non-improving stages.
- [x] Optimization renders only the new layer against a detached prefix, changes only new RGB
      coefficients, and cold joint rendering agrees with the accumulated reconstruction.
- [x] Tests cover validation, moment geometry, hierarchy/tree invariants, prefix immutability,
      determinism on CPU, monotone accepted checkpoint order, cap handling, save/load parity, and
      a small end-to-end report.
- [x] A task-local report exposes source/config identity, all stable rows, complete stage history,
      reconstruction/error images, fields, JSON/JSONL/CSV, quality/local-artifact/count/time/byte
      curves, and passes `scripts/check_report_bundle.py --allow-dirty` for the diagnostic tree.
- [x] The frozen Janelle diagnostic below is executed without retuning after outcomes are opened;
      negative rows and a failed gate remain visible.
- [x] Architecture, Field V2 design, task Index/session brief, and ARA records stay synchronized;
      `./scripts/verify.sh` passes.

## Interfaces touched

`src/structsplat/progressive_residual_quadtree.py`, a task driver under `scripts/experiments/`,
focused tests, `docs/architecture.md`, `docs/additive_field_v2.md`, this task, the Index, generated
session brief, and result/ARA evidence.

## Depends on

HIER-005, CORE-013, BENCH-002, ADR-0006

## Agent workflow

- Driver: codex
- Reviewer: pending-distinct
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`. This dirty/exposed
diagnostic can guide implementation only; a later result-bearing comparison needs a clean tree and
distinct prospective protocol review.

## Notes

### Frozen exposed-image diagnostic protocol (2026-08-05)

- Source and mask are the exact HIER-005 C0001 inputs, SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b` and
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`, deterministically resized
  to 512x443 with the existing 15,929-pixel mask.
- Main arm: start level 6 (64-pixel cells), leaf scale 0.18 px, hard 3-sigma support, signed direct
  coefficients, mask-aware error smoothing sigma 1.5 px, at most 256 appended rows per stage, and
  a global cap of 8,192 rows.
- Base RGB optimization: 400 Adam steps. Child-layer optimization: 50 Adam steps at LR 0.05;
  objective is masked MSE plus four times the worst 1% foreground pixel-MSE mean. Evaluate the
  unchanged prefix and checkpoints every five steps by normalized raw pixel-max/7x7-patch
  violation, then masked SSE.
- Stop gate: exact displayed-8-bit foreground pixel RGB-RMSE maximum `<=0.02` and maximum complete
  black-matted 7x7 patch RMSE `<=0.01`. Preserve snapshots at the base, the last prefix at or below
  4,096, the first passing prefix, and the terminal prefix.
- Strong fixed-count context control: the existing hard-3-sigma touched-only HIER-005 rows at
  N=4,096 and N=8,192. They are not rerun or silently converted into equal-work/rate controls.
- Report masked PSNR/MSE; full black-matted SSIM/MS-SSIM/LPIPS; displayed pixel tails and
  3/7/15/31 patch maxima; attempted/accepted stages and rows; optimizer/render/total time; full
  uncoded/reference bytes; explicitly non-rate coefficient/tree proxies; fields, full/error/crop
  visuals, and curves over accepted Gaussian count.
- Abandon this fixed prefix-frozen mechanism for the exposed 4k goal if no prefix at or below 4,096
  passes. If no prefix by 8,192 passes, retain the complete negative result and do not weaken the
  gate. No parameter retuning on C0001 may become selection evidence after the run begins.

### Diagnostic outcome (2026-08-05)

The first frozen execution is retained at
`results/hier006_janelle_progressive_residual_quadtree_2026-08-05`. It exposed a numerical
implementation defect: an unchanged local maximum differed by about `8e-7` between the NumPy
float64 prefix key and Torch float32 checkpoint key, so the intended equal-violation/SSE tie-break
rolled back and blocked an improving batch. The correction computes checkpoint keys in one domain
and uses a documented 32-float32-epsilon equivalence band. No protocol parameter changed.

The source-snapshotted, corrected report is
`results/hier006_janelle_progressive_residual_quadtree_corrected_2026-08-05`; its module and driver
SHA-256 values are `fd693b24055f293f59d8fbb7859865aa02e0728a6d4e29a3cf02fba4ff83ef37`
and `eacac5294c5ec0057bb7e643fb541cb53b9be48797a48b655c5c7c61ff9a3370`.
`check_report_bundle.py --allow-dirty` passes. The exact-parameter CUDA repeat at
`results/hier006_janelle_progressive_residual_quadtree_precisionfix_repeat_2026-08-05` reproduces
both displayed gate metrics exactly; terminal PSNR differs by `1.7e-5` dB.

| prefix | PSNR | MS-SSIM | LPIPS | pixel max | 7x7 max | gate | full bytes | structural proxy |
|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| 12 | 15.619 | 0.938690 | 0.109243 | 0.801520 | 0.462759 | fail | 28,736 | 28,498 |
| 3,986 | 27.805 | 0.995514 | 0.021590 | 0.222276 | 0.085998 | fail | 155,904 | 76,683 |
| 8,192 | 32.882 | 0.998979 | 0.014489 | 0.107301 | 0.037518 | fail | 290,496 | 127,680 |

The native 14,268,226-byte JPEG ratios are 91.5x/49.1x for the 3,986/8,192 full fields, but they
compare against a resized raster and are not compression evidence. Against the exact 29,263-byte
evaluation PNG, the ratios are only 0.188x/0.101x for full fields and 0.382x/0.229x even for the
non-self-contained structural proxy. At 8,192 rows, 5,106 rows are retained ancestors and 3,086
are level-0 leaves. The worst displayed boundary pixel remains inside an unsplit level-1 cell.

Verdict: the literal immutable-prefix Gaussian quadtree fails the predeclared 4k and 8k artifact
gates and is not competitive with the contextual HIER-005 4,096/8,192 rows (30.481/52.356 dB; only
the latter passes). Retain HIER-006 as a negative control. A successor may use the quadtree as an
allocation/index schedule, but must independently test artifact-first selection, local or joint
coefficient reconciliation, and nested detail bases without treating ancestor or geometry side
information as free.

### Handoff

#### Objective

Implement and honestly kill-test the user's coarse-to-fine hierarchy idea as a default-off,
parent-preserving direct-additive method without changing maintained StructSplat behavior or
calling a geometry/tree proxy compression.

#### Changes

Added `progressive_residual_quadtree.py` with typed NumPy-first quadtree geometry/topology,
immutable prefix fields, signed residual child fitting, smoothed-error-per-row selection,
artifact-first/SSE checkpointing, numerical tie safety, cold acceptance/rollback, complete
trajectory telemetry, and explicit full/proxy byte ledgers. Added the generic HIER-006 report
driver and 20 focused tests. Synchronized the core skill, architecture/Field V2 design, task graph,
session brief, and ARA. Source-set digest for module/driver/tests is
`c562a28c9881c6211894c95695bcc825b112055c30f24d6d65d3e6986a14af1a`.

#### Evidence

The authoritative corrected bundle and exact-parameter CUDA repeat both pass
`check_report_bundle.py --allow-dirty`. The terminal corrected row is 32.8816 dB with displayed
pixel/7x7 maxima 0.107301/0.037518, so it fails the 0.02/0.01 gate. The 3,986 prefix also fails at
27.8050 dB and 0.222276/0.085998. The report exposes 58 curves, fields, all stage/checkpoint rows,
hierarchy maps, and worst crops. Focused hierarchy tests pass 20/20; the field regression slice
passes 84/84; `./scripts/verify.sh` passes with 1,618 tests and 4 skips before the final ARA-only
epilogue, whose structural checks are rerun separately.

#### Assumptions

The mask is shared decoder side information only for the explicitly labeled structural proxy.
Displayed 8-bit pixel/patch maxima are the frozen local gate, not a transferable human-visibility
threshold. The persisted HIER-005 rows are contextual controls, not equal-work jointly executed
arms.

#### Uncertainties

CUDA coefficient gradients are not bit-deterministic, although the repeat preserves displayed
gate metrics exactly. This one exposed downscaled image cannot establish general hierarchy
behavior. No complete tree grammar, quantizer, entropy model, cold decoder, matched work, disjoint
data, or independent prospective/result review exists.

#### Review focus

Check the RS moment orientation, prefix immutability, complete-child grouping, stable tie ordering,
float32 equivalence band, patch-gate arithmetic, cold accumulated/joint parity, snapshot history,
full-versus-proxy byte labels, and the diagnosis that retained ancestors plus non-artifact-first
selection—not the superseded comparison bug—cause the corrected failure.

#### Protected actions not taken

Did not retune C0001 after opening outcomes, weaken the gate, delete failed/superseded runs, promote
HIER-006 into the maintained pipeline, change renderer/Field V2 semantics, claim novelty or
compression, commit, push, or consume held-out confirmation data.

#### Recommended next action

Have a distinct reviewer reproduce the focused numerical invariants and audit the corrected bundle.
If hierarchy work continues, open a separate prospectively reviewed task on disjoint images for a
scheduler-only quadtree with artifact-first split priority and local/joint path coefficient
reconciliation, matched against HIER-005 and fixed-N controls under complete bytes and work.
