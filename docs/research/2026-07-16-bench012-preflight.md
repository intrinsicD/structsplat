# BENCH-012 spatial-connectivity policy assay: preflight unavailable

**Date:** 2026-07-16
**Decision:** close the frozen BENCH-012 formulation without retuning
**Scientific topology outcome:** unavailable; no candidate branch was rendered or recovered

## Why this experiment was selected

BENCH-009 and corrected BENCH-011 closed the current local affine/carrier linear-action
formulation: a coherent nested extension statistic still failed to predict native finite gain.
The adversarial next question was therefore discrete and nonlocal—whether reconstruction
connectivity can select a finite count-neutral reallocation that survives recovery.

The literature audit rejected the first raw-Betti proposal before execution:

- [Topology-GS](https://arxiv.org/abs/2412.16619) already applies persistent topology to Gaussian
  interpolation and rendered structure;
- [3DGS as MCMC](https://proceedings.neurips.cc/paper_files/paper/2024/hash/93be245fce00a9bb2333c17ceae4b732-Abstract-Conference.html)
  already establishes fixed-budget dead-Gaussian relocation;
- [LeGS](https://arxiv.org/abs/2605.00408) already treats Gaussian density control as delayed-credit
  discrete action selection;
- [Betti Matching](https://proceedings.mlr.press/v202/stucki23a.html) demonstrates that equal global
  Betti numbers can hide a feature at the wrong spatial location; and
- [Pitfalls of topology-aware segmentation](https://arxiv.org/abs/2412.14619) shows that digital
  connectivity and metric aggregation can reverse topology rankings.

BENCH-012 was therefore frozen as a narrower policy-value assay with no action novelty. It reused
COMP-006's exact 16 residual-NMS candidates and fixed minimum-activity donor. The proposed selector
was a spatially indexed anchored connectivity partition distance (ACPD), requiring one no-trade
winner across thresholds `{0.4,0.5,0.6}` and both legal foreground/background 4/8 conventions.
Inference included all targets, with immediate-objective fallback contributing exactly zero on
ties or unstable topology decisions. Immediate L1/SSIM, residual rank, 20-step rollout, and
continuous no-action were frozen controls.

## What the preflight established

The deterministic topology core itself passed:

- 24 source-bound `64x64` rectilinear binary targets (`6 families x 4 variants`);
- analytic `(beta0,beta1)` agreement under foreground-8/background-4 and the inverse convention;
- no diagonal-only contacts and at least four-pixel strokes/gaps;
- a frozen `16x16` target-independent anchor lattice;
- exact rational tie handling and separate foreground/background no-trade selection; and
- 13 focused topology tests, including a case with equal global Betti numbers but positive spatial
  ACPD.

The full focused BENCH-012 set passed `17/17` tests and Ruff before the source-bound action
preflight. The target-suite manifest SHA-256 is
`52eb4469f1eaa7343d0bc6c9028c1bcd94fe396cedb414e87ab6544610cbe975`.

The first frozen cell, `closed_frame_v0:seed0`, then failed the required action-work invariant:

```text
required feasible candidates: at least 4
observed feasible candidates: 2 (indices 2 and 15)
```

Every candidate used the unchanged COMP-006 isotropic scale `5.0087924 px`, rotation zero, and
renderer support radius `(16,16)`. The two valid candidates had untruncated `33x33 = 1089`-visit
support rectangles. The other 14 residual candidates lay too close to an image boundary and were
clipped, with work counts between `783` and `1023`. The frozen target margin was enough for digital
topology certification but not for the inherited candidate footprint.

The runner failed immediately after parent fitting, candidate construction, and the geometry-only
work check. It did **not** construct replacement branches, compute ACPD, choose a policy, render a
candidate action, run 20/100-step recovery, or inspect quality/convergence outcomes. The recorded
error explicitly sets `scientific_outcomes_scored=false`.

## Evidence and provenance

Canonical preflight artifact:

```text
results/bench012_topology_policy_preflight_v1/
```

- binding: `c68c5c624d052eaa7cc989ed03dac618fb5adbfe2e3270b54f6efef5c82063d8`
- config SHA-256: `fb6f5cc0940cfc1d704caac4a72b166a951f9cd9a95a2d299c9210d62ace4f7e`
- error ledger SHA-256: `0d53dcbee7be94903181e32c2a4b879bfbdef0c11f43b8996c25a681a388c3c9`
- executed-source archive SHA-256:
  `2645bc526a63b6c5fc1f6557e94462349b7a00ba2e05410c37e5850729901fdb`

The archive contains the eight files in the source-bound manifest. Execution used single-thread
CPU, Python `3.12.9`, NumPy `2.1.3`, Torch `2.9.0+cu128`, commit
`5dc649397c40e69cf3e96bd27df2c5e2812d003d`, and a recorded dirty-worktree digest.

## Claim disposition

| Claim | Disposition | Reason |
|---|---|---|
| ACPD predicts delayed action value | **Unavailable** | The action feasibility precondition failed before candidate scoring. |
| Topology is or is not useful in StructSplat | **Not answered** | No topology-selected branch existed. |
| Quality or convergence improvement | **Not answered** | No action recovery was run. |
| Performance | **Not answered** | CPU preflight time is diagnostic overhead only. |
| Compression | **Not answered** | Equal row count is not an encoded stream. |
| Expressiveness | **Not answered** | Grammar and count were fixed, and no finite action was scored. |

Per the preregistration, do not relax the untruncated/equal-work filter, move the exposed targets,
shrink the candidate scale, change the action bank, or lower the four-candidate minimum. Such a
change would be a rescue on spent data. This availability failure also does not falsify persistent
matching or the earlier ownership-adjacency defect-charge proposal; those are different state and
operator hypotheses and would need new disjoint preregistration.
