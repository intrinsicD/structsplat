# FIT-019: Opacity-split gauge-equivalence audit

## Status

Implemented/screened. Protocol v1 was frozen before its procedural run. A subsequent adversarial review
found that the v1 implementation logged but did not gate alpha-0.7 relative/top-k parity, used set
rather than multiset agreement, and checked the pre-recovery rather than recovered count. The v1
data pass the omitted checks, but v1 is retained only as a diagnostic pilot. Protocol v2 changes
only those gate/provenance implementations, not targets, seeds, arms, thresholds, actions, or
horizons, and was frozen before the corrected run. V2 confirms the exact commutation diagnostic but
rejects the recovery-utility guard. No production group metadata, fit mode, default change,
natural-image confirmation, or novelty claim is authorized.

## Context

For StructSplat's normalized renderer, replacing one row of opacity `o` by co-located copies with
fractions `f_j o`, identical geometry/color, and `sum_j f_j = 1` preserves numerator,
denominator, and rendered pixels up to reduction-order roundoff. The row-wise FIT-018 score

`S_i = E_i / M_i^alpha`

then transforms as `S_ij = f_j^(1-alpha) S_i` when the mass clamp is inactive (with the zero-mass
case handled separately). Thus `alpha=1` preserves each child value but not the number of top-k
tickets, while `alpha=0.7` changes both score magnitude and ticket count.
Aggregating components before scoring,

`S_group = sum_j(E_ij) / sum_j(M_ij)^alpha`,

is invariant to the exact opacity refinement.

Component splitting and non-identifiability are established mixture-model phenomena. In Gaussian
splatting specifically, Rota Bulo et al. correct clone opacity bias under alpha compositing, and
SteepGS models a parent as normalized weighted offspring, assigns half opacity to two children,
and derives a second-order split direction. FIT-019 therefore tests an operator-commutation
diagnostic for this normalized 2D renderer; it does not claim opacity conservation or component
splitting as new.

## Goal

Determine whether exact opacity refinements cause current row-wise allocation to choose different
physical actions, and whether aggregate-first alpha-1 group scoring improves recovery rather than
merely satisfying a constructed invariant. Alpha 0.7 remains an algebra/control arm only because
FIT-018 already rejected it as the selected recovery mechanism; it is not being rescued here.

## Frozen two-stage protocol (v2 audit-corrected execution)

### Stage A — algebra and deterministic counterexamples

- Implement a benchmark-only exact opacity refinement that preserves every non-opacity attribute
  and carries external integer `group_ids`; do not add lineage to `GaussianField`.
- Verify normalized-render parity, responsibility denominator parity, the analytical child-score
  law for equal and unequal fractions, and aggregate-first group-score invariance.
- Include the three-row overlap counterexample where alpha 0.7 changes the top-2 physical groups,
  and the alpha-1 counterexample where two equivalent rows consume both top-k slots.
- Cover sparse/permuted group labels, zero mass, filter variance, affine colors, and background
  masks. Reject mixed-background groups and non-identical geometry/color within an asserted exact
  equivalence class.

### Stage B — disjoint procedural recovery guard

- Procedural targets only: vertical step, diagonal step, corner, disk, narrow diagonal line,
  sinusoidal grating, checkerboard, and opposing RGB ramps; 48x48; seeds 0 and 1. No tracked
  natural image or held-out validation set is consumed.
- Fit one canonical `quadtree_wse` field per target/seed with explicit trainable opacity, N=32,
  40 deterministic CPU steps. Add exactly eight children and replay 20 and 100 recovery
  steps independently.
- Equivalent views are frozen to canonical and a deterministic score-blind half-opacity refinement
  of even canonical group IDs. Stage A tests unequal fractions algebraically, but Stage B does not
  sweep `rho`.
- Operators: canonical support, raw-row responsibility at alpha 0.7, raw-row responsibility at
  alpha 1, and aggregate-first group responsibility at alpha 1, evaluated on the canonical/gauge
  views where meaningful. Record projected top-k positional agreement, multiset/unique Jaccard,
  exact equality, ordered action hashes, selected group multisets, unique-site count,
  immediate/post-20/post-100 PSNR and MS-SSIM, PSNR AUC, action/recovery time, exact immediate and
  recovered counts, and finiteness.
- To isolate allocation from duplicated Adam state and unequal raw row count, map selected group
  IDs back to the same canonical field and apply eight sequential moment-preserving splits.
  Repeated raw selections split the highest-opacity descendant of that group, breaking ties by
  stable row/lineage ID; grouped selection chooses distinct groups. Every arm receives the same
  eight-row increment and finishes at N=40. Recovery is an independent fresh-Adam restart at each
  20/100 horizon, not a claim about production in-run optimizer-state continuation. Identical
  ordered action hashes share one cached recovery trajectory.

## Preregistered decisions

The **commutation hypothesis** is confirmed only if:

1. every equivalent render has max absolute error at most `2e-6` and every score is finite;
2. aggregate-first group scores at both alpha controls match canonical scores within `2e-5`
   absolute and
   relative tolerance, with identical top-8 groups in every view;
3. alpha-1 children obey invariance within `2e-5`, while direct raw top-k records ticket
   multiplicity rather than being incorrectly described as group invariant; and
4. raw alpha-1 changes the selected physical-group multiset on both seeds in at least 6/8 target
   clusters. If not, stop: the gauge sensitivity is algebraically real but benign in this regime.

The **recovery-utility guard** survives only if the commutation hypothesis is confirmed and,
against raw alpha 1 under the gauge view, grouped alpha 1:

1. has canonical/gauge ordered action hashes equal in all 16 target/seed cells;
2. gains at least `+0.05 dB` mean post-20 PSNR and wins at least 6/8 target clusters after
   averaging seeds;
3. is no worse than `-0.02 dB` mean at post-100;
4. is no worse than canonical support by `-0.05 dB` at both recovery horizons;
5. adds no more than 15% total-100 time versus canonical support; and
6. preserves exact N=40 with finite scores and renders.

A commutation-only pass records a diagnostic result but does not authorize production quotient
metadata. A utility pass authorizes only a separately named approximate-family experiment on
disjoint data; ordinary split siblings are not exact gauge classes. Do not tune targets, fractions,
subset rule, alpha, birth primitive, or thresholds after seeing this guard.

## Acceptance criteria

- [x] Benchmark writes source/input/config/environment provenance, per-cell CSV/JSON, aggregate
      decisions, and a concise Markdown summary.
- [x] Focused tests cover all Stage-A invariants, counterexamples, validation failures, and the
      Stage-B decision function.
- [x] The procedural Stage-B run completes under the frozen deterministic protocol.
- [x] Results are recorded whether positive, partial, or negative, with no production change when
      its authorization gate fails.
- [x] Full tests, Ruff, task/docs synchronization, and diff hygiene pass.

## Measured decision (v2, 2026-07-15)

The corrected 128-row run and untouched-source replay completed all eight procedural targets x two
seeds x eight arms. Every immediate, post-20, and post-100 field had exact N=40 and finite values.
The largest equivalent-view render error was `8.345e-7`; maximum quotient relative error was
`2.701e-6` at alpha 0.7 and `2.747e-6` at alpha 1. Grouped top-8 actions matched in all 16
checkpoints at both alphas. Raw alpha-1 physical-group multisets changed on both seeds for all 8/8
target families, while grouped alpha-1 canonical/gauge action hashes matched 16/16. The
**commutation hypothesis is confirmed**.

The recovery result does not promote that invariant into a method:

| Comparison | Post-20 PSNR | Post-100 PSNR | Target-family wins at 20 |
|---|---:|---:|---:|
| quotient alpha 1 vs raw gauge-row alpha 1 | `+0.2111 dB` | `-0.6007 dB` | `5/8` |
| quotient alpha 1 vs canonical support | `-0.0665 dB` | `+0.4202 dB` | — |

The candidate fails the preregistered 6/8 post-20 breadth, `-0.02 dB` post-100 retention, and
`-0.05 dB` post-20 support-floor gates. Primary/replay timing overhead was `+1.38%`/`-1.26%` and
is treated only as noisy accounting; both pass the 15% bound. Thus the **recovery-utility guard is
rejected** and no production quotient/lineage state is added.

A descriptive response audit found post-20 to post-100 sign reversals in 7/16 cells and near-zero
median effects. The large means are target-composition-sensitive, especially to the sinusoid.
The next defensible experiment is controlled perturb--recover response spectroscopy across
coverage levels and dense horizons, not a larger quotient confirmation or another alpha sweep.

Raw primary/replay artifacts are in `results/fit019_opacity_gauge_guard_v2_2026-07-15/` and
`results/fit019_opacity_gauge_guard_v2_replay_2026-07-15/`. Both contain a verified 24-file source
snapshot with combined SHA-256
`89f52281e5596e7225cf278be74eeaabc423c54db2f176ecf4d5bfa5d2b99f23`; every non-timing row and
aggregate field replays exactly. The research-manager epilogue binds the durable evidence copy.

## Interfaces touched

`benchmarks/gauge_equivalence_audit.py`, focused tests, benchmark documentation, this task, and the
research evidence record. Production `GaussianField`, config, fitter, renderer, and CLI remain
unchanged unless a later independently authorized task requires them.

## Depends on

FIT-007, FIT-009, FIT-018, BENCH-002, E1/T1 in the 2026-07-15 research portfolio.
