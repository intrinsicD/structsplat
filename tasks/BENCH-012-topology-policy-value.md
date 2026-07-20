# BENCH-012: Spatial-connectivity policy value for finite reallocation

**Status: completed unavailable at preflight (2026-07-16); close without retuning.** The
deterministic topology core and focused tests pass, but the first source-bound cell has only two of
the required four untruncated equal-work actions. The runner stopped before constructing or
rendering a candidate branch, computing ACPD, choosing a policy, or running recovery. This is an
availability failure, not evidence for or against topology. No relocation primitive, topology
method, production allocator, quality improvement, convergence improvement, renderer speedup,
compression result, or expressiveness result is claimed.

## Preflight outcome

`closed_frame_v0:seed0` inherited COMP-006's common candidate scale `5.0087924 px`, rotation zero,
and support radius `(16,16)`. Only candidate indices `{2,15}` had untruncated modal work
`33x33=1089`; the other 14 candidates were boundary-clipped at `783..1023` visits. The frozen
minimum was four, so the assay failed closed before scientific outcomes.

The canonical artifact is `results/bench012_topology_policy_preflight_v1/`: binding
`c68c5c624d052eaa7cc989ed03dac618fb5adbfe2e3270b54f6efef5c82063d8`, config SHA-256
`fb6f5cc0940cfc1d704caac4a72b166a951f9cd9a95a2d299c9210d62ace4f7e`, error-ledger SHA-256
`0d53dcbee7be94903181e32c2a4b879bfbdef0c11f43b8996c25a681a388c3c9`, and executed-source
archive SHA-256 `2645bc526a63b6c5fc1f6557e94462349b7a00ba2e05410c37e5850729901fdb`.
See `docs/research/2026-07-16-bench012-preflight.md`.

## Question

BENCH-009 and corrected BENCH-011 show that valid local extension capacity does not predict native
finite gain. The next materially different question is therefore discrete and nonlocal:

> Among the same finite, count-neutral residual replacements, does a spatial connectivity score
> choose actions with better delayed recovery than the strongest immediate-objective choice?

The action bank is deliberately inherited from COMP-006: 16 spacing-controlled residual rows and
one fixed minimum-activity donor. FIT-004 and 3DGS-MCMC already establish relocation as a known
mechanism. The only tested delta is selector value after continuous recovery.

## Prior-art and failure audit

- [Topology-GS](https://arxiv.org/abs/2412.16619) already uses persistent topology to choose
  Gaussian interpolation and constrain rendered topology.
- [3DGS as MCMC](https://proceedings.neurips.cc/paper_files/paper/2024/hash/93be245fce00a9bb2333c17ceae4b732-Abstract-Conference.html)
  already relocates low-contribution Gaussians at fixed budget.
- [LeGS](https://arxiv.org/abs/2605.00408) already treats Gaussian density control as delayed-credit
  discrete action selection.
- [Betti Matching](https://proceedings.mlr.press/v202/stucki23a.html) shows why global Betti counts
  can be correct at the wrong spatial location.
- [Pitfalls of topology-aware segmentation](https://arxiv.org/abs/2412.14619) shows that digital
  connectivity, resolution, label construction, and metric aggregation can reverse conclusions.
- [Topology-preserving binary downsampling](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03067.pdf)
  is the closest finite-budget binary-image approximation precedent.

Consequently BENCH-012 is an N1/N2 measurement program, not a topology-method novelty claim. Raw
global Betti error and a hand-labelled "repair orientation" are rejected as the primary selector.

## Frozen data and independent units

- Generate 24 new deterministic `64x64` grayscale binary targets: six families with four variants
  each. Families are connected dumbbell, separated pair, closed frame, open frame, connected
  double frame, and theta frame.
- Shapes are rectilinear, remain at least six pixels from the image boundary, have no diagonal-only
  contact, and have minimum stroke/gap width of four pixels.
- The generator records analytic expected `(beta0,beta1)`. Every target must reproduce those
  values under both legal dual conventions: foreground-8/background-4 and
  foreground-4/background-8. Failure stops before fitting.
- Targets and decoded-pixel hashes must be disjoint from BENCH-009, FIT-020, and COMP-006.
- Seeds `{0,1}` are repeated target-independent initialization measurements. Average seeds before
  inference; the 24 targets, not the 48 fits, are independent units.
- Family labels are available only to integrity checks and clustered inference. The selector sees
  target pixels and candidate renders, never family names, generator parameters, analytic Betti
  labels, a critical locus, or a preferred action.

## Frozen parent and objective

For each target and seed:

1. Initialize the unchanged no-opacity, constant-RGB normalized grammar at `N=64` from an `8x8`
   target-independent grid. The seed may add a frozen small target-independent jitter.
2. Fit for 600 Adam updates with the repository default aligned objective
   `0.7 * L1 + 0.3 * (1 - SSIM)`, default learning rates, no schedule, and no prune, split,
   relocation, QAT, opacity, color solve, curriculum, or checkpoint selection.
3. Preserve the exact parent field, complete Adam state, objective, and global step. No branch may
   start a fresh optimizer.

This assay uses deterministic single-thread CPU reference rendering. Its timings are descriptive
selector/experiment cost only and cannot support a production-performance claim.

## Frozen action bank and work control

- Invoke COMP-006's unchanged 16-row `residual_add_nms` action construction on the fitted parent.
  Use its unchanged rank order and fixed minimum-activity donor.
- Each branch removes that donor and appends exactly one candidate, so every branch remains `N=64`
  with the same parameter grammar.
- The feasible set contains only candidates with untruncated support, the modal discrete support
  work count, finite state, and the frozen common scale/rotation primitive. At least four candidates
  must remain in every cell; otherwise the run fails closed.
- Candidate branches carry every survivor's Adam tensors and global tensor step exactly. The new
  row starts with zero first/second moments. PyTorch Adam has one scalar step per parameter tensor,
  not a row-local step; that inherited scalar is explicitly recorded and identical across action
  branches.
- The no-action branch carries the untouched field and optimizer state continuously.
- Hashes must prove that candidate construction and every branch leave the shared parent unchanged.

## Primary spatial-connectivity selector

Use a target-independent `16x16` anchor lattice at pixel coordinates `{1,5,...,61}^2`. For a
thresholded image and one class `c` (foreground or background), define the anchored connectivity
relation

```text
R_c(i,j) = 1 iff anchors i and j are both class c and lie in the same c-component.
```

Include diagonal pairs so `R_c(i,i)` records anchor class. Compare target and candidate relations
with normalized symmetric difference

```text
d_c = |R_c(target) xor R_c(candidate)| / |R_c(target) or R_c(candidate)|,
```

with `0/0 := 0`, and use `d_fg + d_bg` as the anchored connectivity partition distance (ACPD).
This retains spatial correspondence that global `(beta0,beta1)` discards.

Compute ACPD for thresholds `{0.4,0.5,0.6}` using `gray = mean(RGB)`, foreground defined by
`gray >= threshold`, zero-valued exterior padding, and both dual connectivity conventions. A
topology winner exists only when the same candidate is the unique minimum in all six evaluations.
Otherwise the topology policy deterministically falls back to the immediate-objective policy.
Preserve foreground and background distances separately; an aggregate trade cannot be called a
stable topology decision.

Global `beta0`/`beta1`, per-threshold choices, leave-one-threshold-out choices, and threshold-margin
counts are descriptive robustness diagnostics only. They cannot replace ACPD after results are
seen.

## Frozen policies and recovery

At native step 0, score all feasible candidates and define:

- `residual`: lowest original COMP-006 action index;
- `immediate`: lowest aligned step-0 L1/SSIM objective, then action index;
- `topology`: the stable ACPD winner, otherwise `immediate` fallback;
- `rollout20`: lowest aligned objective after 20 continuous updates, then action index;
- `no_action`: continuous parent optimization without replacement.

Run all feasible action branches to step 20. Continue each unique candidate selected by
`residual`, `immediate`, `topology`, or `rollout20` to step 100, reusing a trajectory when policies
agree. Continue no-action to step 100. Log every five updates plus exact checkpoints 0, 20, and
100. Do not reselect after step 20 except for the explicitly named `rollout20` control.

Report aligned objective, MSE, PSNR, SSIM, PSNR AUC over steps 0--100, topology diagnostics,
support work, optimizer-state hashes, render hashes, wall time, and peak RSS. PSNR and objective
are separate gates because PSNR is not the training objective.

## Frozen development gate

The topology policy is evaluated over **all 24 targets** after averaging seeds. Agreement and
fallback cells contribute exactly zero topology-versus-immediate policy difference. A
disagreement-only table is descriptive and cannot be the inferential population.

The mechanism passes only if all conditions hold:

1. Stable ACPD changes the immediate choice on at least 12 targets spanning at least four
   families.
2. At step 100, topology beats `immediate`, `residual`, and `no_action` by at least `+0.15 dB`
   mean PSNR over all 24 targets.
3. For each of those three comparisons, a deterministic family-stratified target bootstrap has a
   95% lower bound above zero for both PSNR gain and aligned-objective gain.
4. On the topology-versus-immediate disagreement targets, at least 10 targets and at least 80%
   are PSNR wins at step 100; report the exact paired sign test.
5. Mean topology-versus-immediate PSNR and aligned-objective effects are non-negative at step 20,
   PSNR-AUC lower bounds are above zero versus immediate and no-action, and no family has mean
   step-100 PSNR effect below `-0.10 dB` versus immediate.
6. The topology policy is no worse than `-0.05 dB` mean PSNR versus the expensive `rollout20`
   selector. This is an operational-strength guard, not evidence that topology is novel.
7. Every target, support, parent immutability, optimizer continuation, manifest, exact-axis,
   completeness, and replay invariant passes.

## No-rescue rule and claim boundary

Any failed condition closes this exact ACPD selector over the unchanged residual-replacement bank.
Do not change anchors, thresholds, connectivity, targets, parent horizon, candidate construction,
objective, recovery horizon, fallback, or gates on these data. A failure does not falsify
persistent matching or the earlier ownership-adjacency defect-charge hypothesis; those contain
spatial/ownership state absent from ACPD and would require a new preregistration.

A pass authorizes only a disjoint natural-image validation against immediate objective, residual,
Sobel/structure-tensor, curvature, and delayed-rollout controls. Fixed `N` and grammar make this an
allocation result, not expressiveness. Equal row count is not equal encoded bytes. Diagnostic CPU
cost is not renderer/fitter performance. Production code and defaults remain unchanged.

## Allowed interfaces

New benchmark, focused tests, task/research documentation, ignored result artifacts, and executed
source archive only. Do not modify production fitting, rendering, codec, configuration, or CLI
paths.

## Depends on

BENCH-002/009/011, FIT-004/017/020, COMP-006, and the T4 defect-charge proposal in the 2026-07-15
frontier research report.
