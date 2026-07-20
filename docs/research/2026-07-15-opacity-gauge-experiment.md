# FIT-019: opacity-gauge allocation experiment

**Date:** 2026-07-15
**Scope:** normalized 2D renderer, procedural mechanism guard, fresh-optimizer recovery
**Preregistration:** [`tasks/FIT-019-opacity-gauge-equivalence.md`](../../tasks/FIT-019-opacity-gauge-equivalence.md)
**Decision:** commutation confirmed; recovery utility rejected; no production state change

## Research question

Does a growth allocator choose the same physical action after a function-preserving refinement of
the Gaussian representation, and does enforcing that invariance improve quality or convergence?

For StructSplat's normalized renderer, write `w_i(x) = o_i b_i(x)`,

`D(x) = sum_i w_i(x)`, `V(x) = sum_i w_i(x)c_i(x)`, and `I(x) = V(x)/(D(x)+eps)`.

Replacing row `i` by co-located copies with identical geometry/color and weights
`w_ij = f_j w_i`, `sum_j f_j = 1`, preserves `D`, `V`, and `I`. Responsibility mass and error
split linearly: `M_ij = f_j M_i`, `E_ij = f_j E_i`. When the mass clamp is inactive,

`S_ij(alpha) = E_ij / M_ij^alpha = f_j^(1-alpha) S_i(alpha)`.

At alpha 1 every child keeps the parent score, but duplicate rows receive duplicate top-k tickets.
At alpha 0.7 a half-opacity child receives `2^-0.3 = 0.812252...` times the parent score. Scoring
the group after aggregation, `S_G = sum(E_i)/sum(M_i)^alpha`, is invariant.

This symmetry is not an ordinary moment-preserving split. That birth operator moves and shrinks
the children, so ordinary siblings cannot be asserted to be an exact equivalence class.

## Prior-art boundary

Component non-identifiability and split/merge methods are established. The closest direct threats
are:

- [SteepGS (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.html),
  which represents a parent through normalized weighted offspring and uses two half-opacity
  children in its split construction;
- [Revising Densification in Gaussian Splatting (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08041.pdf),
  which corrects opacity after cloning for ordered alpha compositing;
- [Splitting Steepest Descent (NeurIPS 2019)](https://papers.nips.cc/paper_files/paper/2019/hash/3a01fc0853ebeba94fde4d1cc6fb842a-Abstract.html),
  which supplies the broader mixture-component split foundation; and
- [Soft Anisotropic Diagrams](https://arxiv.org/abs/2604.21984), whose densification score already
  uses responsibility error divided by responsibility mass to a sublinear exponent.

The narrow apparently unexplored remainder is not opacity conservation or group scoring alone. It
is the operator-commutation test under exact opacity refinements, coupled to a controlled recovery
prediction in this normalized 2D setting. That novelty scope remains search-bounded, not a claim of
exhaustive novelty.

## Frozen experiment

Stage A checks exact rendering/denominator parity, the analytical child law, group-score parity,
and deterministic alpha-0.7/alpha-1 ticket counterexamples. Stage B uses eight 48x48 procedural
families (step, diagonal step, corner, disk, diagonal line, sinusoid, checkerboard, RGB ramps),
seeds 0/1, canonical N=32 quadtree-WSE fields after 40 CPU steps, and an exact half-opacity copy of
every even group.

Selectors are canonical/gauge support, raw responsibility alpha 0.7, raw responsibility alpha 1,
and quotient responsibility alpha 1. Each ordered selection is mapped back to the same canonical
checkpoint and receives eight sequential moment-preserving births. Repeated tickets act repeatedly
on the selected canonical group; grouped selection covers distinct groups. Independent fresh Adam
restarts measure immediate, post-20, and post-100 recovery at exact N=40. Thus the experiment
isolates action allocation, but does not emulate production continuation of optimizer state.

An independent review after v1 found three gate-implementation omissions. V2 added the missing
alpha-0.7 relative/top-k checks, count-aware multiset gating, Stage-A numerical gating, recovered
counts, explicit projected top-k metrics, exact cell-table validation, and a 24-file source
snapshot. It changed no target, arm, threshold, action, seed, or horizon. All 4,352 shared
non-timing v1/v2 row values match exactly.

## Results

The corrected v2 primary and source-frozen replay agree exactly on every deterministic row and
aggregate field.

| Check | Result |
|---|---:|
| Maximum equivalent-render error | `8.345e-7` |
| Maximum quotient relative error, alpha 0.7 / 1 | `2.701e-6` / `2.747e-6` |
| Quotient top-8 equality, both alphas | `16/16` checkpoints |
| Raw alpha-1 multiset changed on both seeds | `8/8` target families |
| Quotient alpha-1 canonical/gauge ordered actions | `16/16` equal |
| Immediate/post-20/post-100 counts | all exactly `40` |

Commutation is therefore confirmed. Raw alpha-1 gauge selection covers only 5 groups in 10/16
cells and 6 groups in 6/16, while quotient selection covers 8. Raw projected multiset Jaccard
averages `0.5091`; quotient projected order/multiset/unique agreement is exactly 1.

| Arm | Immediate PSNR | Post-20 PSNR | Post-100 PSNR | Post-100 AUC | Unique groups |
|---|---:|---:|---:|---:|---:|
| canonical support | 24.4108 | 29.6021 | 35.7529 | 32.4442 | 8.00 |
| raw alpha-1 gauge row | 24.4238 | 29.3245 | 36.7738 | 32.6394 | 5.38 |
| quotient alpha-1 gauge | 24.3191 | 29.5356 | 36.1731 | 32.4894 | 8.00 |

Quotient minus raw alpha-1 gauge is `+0.2111 dB` at post-20 but positive on only 5/8 target
families, then `-0.6007 dB` at post-100. Quotient minus canonical support is `-0.0665 dB` at
post-20 and `+0.4202 dB` at post-100. It fails three preregistered quality gates: target-family
breadth, post-100 retention, and the post-20 support floor. Primary/replay timing overhead is
`+1.38%`/`-1.26%`; this variability is gate accounting, not evidence of a speedup.

## Response audit

Across the 16 paired quotient-minus-raw cells, mean/median effects are:

| Horizon | Mean | Median | Positive cells |
|---|---:|---:|---:|
| immediate | `-0.1047 dB` | `-0.0186` | 7/16 |
| post-20 | `+0.2111 dB` | `+0.0390` | 11/16 |
| post-100 | `-0.6007 dB` | `-0.0117` | 8/16 |
| post-100 AUC | `-0.1501` | `-0.0413` | 8/16 |

Post-20 and post-100 signs reverse in 7/16 cells and 3/8 target means. The apparent negative
linear association is not rank-robust (Pearson `-0.593`, Spearman `-0.182`) and is strongly
affected by the sinusoid (`+2.1966 dB` at 20, `-5.6622 dB` at 100). Excluding that family reverses
both aggregate endpoint signs. These are descriptive post-hoc diagnostics, not a new causal test.

## Decision by requested axis

| Axis | Evidence | Decision |
|---|---|---|
| Quality | No consistent endpoint gain; three utility gates fail. | Do not promote quotient allocation. |
| Convergence | Frequent horizon reversals and weak immediate-to-late predictiveness. | Test trajectory response, not a larger endpoint study. |
| Performance | Only small, sign-changing CPU timing differences. | No speedup claim; grouping overhead remains unestablished. |
| Compression | No stream or rate change; production group metadata would add unmeasured state. | No compression claim or codec change. |
| Expressiveness | Exact opacity copies do not enlarge the rendered function class. | No expressiveness claim. |

ADR-0014 keeps exact groups benchmark-only. A next experiment should interpolate distinct-site
coverage from five to eight while holding action count fixed; sample recovery at steps
`0,1,2,5,10,20,40,60,100,200`; and test whether early slope, peak/crossover time, residual
concentration, gradient alignment, curvature, or displacement predicts late recovery across a
larger sinusoid/ramp family with enough seeds to use target family as the inference unit.

## Reproducibility

Primary: `results/fit019_opacity_gauge_guard_v2_2026-07-15/`
Replay: `results/fit019_opacity_gauge_guard_v2_replay_2026-07-15/`
Combined source SHA-256: `89f52281e5596e7225cf278be74eeaabc423c54db2f176ecf4d5bfa5d2b99f23`

Both artifacts contain the benchmark, helpers, task/test, `pyproject.toml`, and every top-level
StructSplat Python source. Algebra is byte-identical; all 128 rows match after removing only six
timing fields; normalized aggregates match exactly. Cross-host binary reconstruction remains
imperfect because wheel/library binaries and a detailed CPU model are not snapshotted.
