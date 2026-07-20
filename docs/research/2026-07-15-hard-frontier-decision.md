# Hard-frontier decision: measure the missing function space before adding a method

**Date / literature cutoff:** 2026-07-15
**Repository:** StructSplat
**Decision status:** BENCH-009/011 local-linear branch closed; no method promoted
**Selected next question:** whether equal-cost finite topology repairs predict delayed survival
**Method novelty claim:** none; this is a new recipient-specific evidence program under the stated
search, built from established linearization, projection, and model-selection tools.

## 1. Decision

Do **not** implement another smooth scalar, initializer, generic predictor, post-hoc precision menu,
full affine primitive, WIPES-like primitive, or wholesale Faster-GS transplant next. BENCH-009 has
now run: its exact ledgers are complete, but the causal instrument failed and neither affine nor
carrier passed the frozen utility decision. The detailed result is in
[`2026-07-16-bench009-results-audit.md`](2026-07-16-bench009-results-audit.md).

The original bounded experiment asked which function space StructSplat's remaining residual
occupies:

```text
current grammar tangent | local affine appearance | localized carrier | finite standard birth
```

For a frozen fitted field `theta`, residual `r`, current scaled renderer Jacobian `J`, and candidate
zero-offset candidate-family columns `A_k`, the core statistic is

```text
Delta_k = ||P_[J,A_k] r||^2 - ||P_J r||^2.
```

The intended subtraction was the point. A raw residual correlation cannot distinguish a representation
deficit from a direction the current model already contains but Adam has not reached. Every oracle
prediction must then survive a finite render and matched `20`/`100`-step recovery, because this
repository has repeatedly observed immediate gains reverse during recovery.

In v3, independently truncating the base and joint scaled factorizations broke the required
nesting, and exact rows contained negative incremental energies. The resulting outside-`J`
classification is unavailable. BENCH-011 v1 changed the measurement algebra on the already-spent
parents but accidentally changed the randomized base seed as well. Corrected v2 reproduces the
exact BENCH-009 bases and all `96` native rows; all four unchanged calibration strata fail. The
current-identity local-linear formulation is therefore closed without retuning.

Finite normalized births have a nonzero denominator-induced offset and therefore use a separately
named exact finite affine-action gain, not the literal projector equation. The experiment keeps an
unpriced full-`J` causal-capacity ledger separate from its matched-six-appearance-DOF ledger; no
winner is selected by mixing those resource scopes.

The experiment is specified in
[`tasks/BENCH-009-residual-tangent-auction.md`](../../tasks/BENCH-009-residual-tangent-auction.md).
The reusable procedure that produced the decision is
[`docs/prompts/hard-frontier-research.md`](../prompts/hard-frontier-research.md).

## 2. Why this supersedes the earlier queue

The previous frontier report proposed “exact per-Gaussian backward” as the first performance
experiment. Live-code inspection changes that interpretation:

- `src/structsplat/cuda/render_ext.cu` already launches the exact backward with one block per
  Gaussian (`i = blockIdx.x`);
- each thread accumulates its strided pixel contributions locally;
- the remaining exact-path opportunity is an intra-block reduction replacing thread-level final
  atomics, followed by chain-rule/optimizer fusion and an end-to-end profile;
- the tiled backward is a different pixel/tile-owned path and still performs per-pixel/per-Gaussian
  global atomics.

Thus a Faster-GS-inspired performance profile remains useful, but it is a bounded engineering lane,
not the highest-information scientific experiment. The earlier report remains a dated record; this
report corrects the current priority rather than rewriting its history.

## 3. Prompt execution and evidence boundary

This pass used the repository's full evidence-first prompt as a baseline and added an adversarial
hard-frontier prompt with four additional constraints:

1. one simple causal core plus a separate robustness ledger;
2. live-code verification of every alleged open mechanism;
3. an independent kill memo written before choosing a winner; and
4. exactly one next experiment with a no-rescue abandonment rule.

Three independent lanes covered direct literature/prior art, hard/simple mechanism generation, and
adversarial rejection. The full earlier idea portfolio is in
[`2026-07-15-frontier-reuse-experiments.md`](2026-07-15-frontier-reuse-experiments.md); this report is
the narrowed decision pass after FIT-018--020 and COMP-006 resolved several branches.

No protected validation data were consumed in this pass. No compression improvement is claimed:
SSPL1 cannot encode affine or carrier state, and DOF/provisional packet sizes are not complete
cold-decodable rate.

## 4. Mechanistic state of the art

The important literature result is not that one paper “wins.” Different methods change different
parts of the system and their reported rates/renderers are not automatically comparable.

| Work | Mechanism | What it establishes | Direct implication for StructSplat |
|---|---|---|---|
| [GaussianImage](https://arxiv.org/abs/2403.08551) | Directly optimizes eight-parameter colored 2-D Gaussians and adds a quantized/VQ compression path. | A small explicit splat representation can fit and decode images rapidly. | Foundational baseline; its accumulation semantics and reported payload must not be silently equated with StructSplat's normalized complete stream. |
| [Image-GS](https://www.immersivecomputinglab.org/publication/image-gs-content-adaptive-image-representation-via-2d-gaussians/) | Error-guided progressive allocation, a custom renderer, top-contributor work control, and an LOD hierarchy. | Allocation and access structure matter, particularly at low rates and nonuniform content. | Generic residual growth, hierarchy, and top-K ideas are occupied; local analogues require their own renderer and actual-rate evidence. |
| [Soft Anisotropic Diagrams](https://arxiv.org/abs/2604.21984) | Softmax/top-K anisotropic ownership with learned temperatures and density control. | Smooth site ownership is already a first-class Gaussian-like image representation. | It is the closest direct threat to generic smooth-max/soft ownership proposals; a new smooth operator needs a different invariant and recipient-specific prediction. |
| [GaussianImage++](https://ojs.aaai.org/index.php/AAAI/article/view/37572) | Distortion-driven densification, content-aware Gaussian filtering, attribute-separated learnable scalar quantization, and QAT. | Allocation and attribute precision can be co-designed while retaining fast decoding. | A covariance/filter transplant already failed locally; entropy/attribute co-design remains open only with a real StructSplat stream. |
| [Structure-Guided Allocation](https://arxiv.org/abs/2512.24018) | Structural initialization, scale-dependent covariance precision, and geometry-consistent regularization. | Structure can drive both capacity and quantizer allocation. | Directly occupies generic tensor/bitwidth proposals; BENCH-007 already rejected StructSplat's old tensor-WSE actual-rate claim. |
| [EA-GI](https://www.sciencedirect.com/science/article/pii/S0165168426000356) | Chromatic/structural entropy drives adaptive quadtree allocation and thresholding. | Entropy-derived content allocation is also occupied in current 2-D Gaussian-image work. | Another entropy or quadtree initializer is not a frontier contribution without a materially different decoded-state or evidence claim. |
| [pre-trained attribute dictionaries](https://ieeexplore.ieee.org/document/11196898/) | Replace, predict, or retain Gaussian attributes from a learned dictionary and entropy-code indices. | Attribute redundancy, not only row count, can dominate rate. | A genuine dictionary task needs training data, dictionary bytes, indices, decoder, and cold-stream comparisons; it is not a small post-hoc precision tweak. |
| [SGI](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_SGI_Structured_2D_Gaussians_for_Efficient_and_Compact_Large_Image_CVPR_2026_paper.pdf) | Seed Gaussians plus lightweight generators impose local structure and support entropy coding at seed level. | Structured generation can trade per-image state for shared decoding structure on large images. | A feed-forward or generated-attribute direction changes the problem and must count the generator/training prior. |
| [AIR](https://arxiv.org/abs/2605.20820) | Stage-wise residual prediction followed by short optimize-and-distill cycles amortizes per-image fitting. | Learned allocation can accelerate fitting when a dataset-scale prior is allowed. | This is a different information regime from training-free per-image optimization; FF-001 cannot be upgraded by a small architecture tweak. |
| [SVGS](https://arxiv.org/abs/2411.18966), [Gaussian Billboards](https://arxiv.org/abs/2412.12734), [Textured Gaussians](https://openaccess.thecvf.com/content/CVPR2025/html/Chao_Textured_Gaussians_for_Enhanced_3D_Scene_Appearance_Modeling_CVPR_2025_paper.html) | Bilinear, kernel, neural, or texture-map appearance varies spatially across one Gaussian. | Constant color is a real expressiveness bottleneck in related Gaussian renderers. | Generic “spatial color per Gaussian” is known. StructSplat's possible contribution is the smallest byte-priced 2-D image grammar and evidence, not the broad concept. |
| [WIPES](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_WIPES_Wavelet-based_Visual_Primitives_ICCV_2025_paper.pdf) | A cosine-modulated Gaussian/Morlet-like primitive adds a learnable local frequency vector and a custom differentiable rasterizer. | Gaussian envelopes alone are low-pass; a localized carrier can represent textures with fewer primitives. | It is the direct control for any carrier. A fixed sine/cosine bank in BENCH-009 is a mechanism probe, not “WIPES in StructSplat.” |
| [Neural Gabor Splatting](https://openaccess.thecvf.com/content/CVPR2026/html/Watanabe_Neural_Gabor_Splatting_Enhanced_Gaussian_Splatting_with_Neural_Gabor_for_CVPR_2026_paper.html) | Per-primitive neural/Gabor appearance and frequency-aware density control target spatial texture variation. | The richer local-frequency design space extends beyond one analytic carrier. | Further lowers novelty of a generic Gabor/carrier proposal and strengthens the need for a minimal byte/work-controlled mechanism probe. |
| [3DGS-LM](https://openaccess.thecvf.com/content/ICCV2025/papers/Hollein_3DGS-LM_Faster_Gaussian-Splatting_Optimization_with_Levenberg-Marquardt_ICCV_2025_paper.pdf), [matrix-free LM with residual sampling](https://arxiv.org/abs/2504.12905) | Custom JVPs and damped normal-equation solves replace or augment Adam. | Sparse Gaussian Jacobians can support practical second-order optimization. | Jacobian/JVP machinery is known; BENCH-009's defensible delta is diagnostic comparison of heterogeneous extension spaces, not LM. |
| [PUP 3D-GS](https://openaccess.thecvf.com/content/CVPR2025/papers/Hanson_PUP_3D-GS_Principled_Uncertainty_Pruning_for_3D_Gaussian_Splatting_CVPR_2025_paper.pdf), [SteepGS](https://arxiv.org/abs/2505.05587) | Fisher/Hessian sensitivity prices pruning; optimization analysis derives split directions and opacity normalization. | First/second-order information already guides Gaussian topology in 3-D work. | “Use the Hessian/Jacobian for splats” is not new. The residual-family attribution and recovery calibration must carry the evidence claim. |
| [Faster-GS](https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf) | Per-Gaussian alpha backward, activation/kernel/optimizer fusion, and periodic spatial reordering reduce atomics and traffic. | Work ownership and memory layout can yield large training speedups without changing representation. | StructSplat already uses Gaussian-owned exact backward. Profile block reduction/fusion as a systems task after correcting that delta. |

### Mathematical and cross-domain foundations

- [Golub--Pereyra variable projection](https://epubs.siam.org/doi/abs/10.1137/0710036)
  eliminates linear coefficients from separable nonlinear least squares. StructSplat already has a
  constant-color conditional solve; extending the idea is an optimization transfer, not novel
  algebra.
- [Mallat--Zhang matching pursuit](https://doi.org/10.1109/78.258082) and
  [Pati--Rezaiifar--Krishnaprasad OMP](https://doi.org/10.1109/ACSSC.1993.342465) select residual-
  aligned atoms and re-solve coefficients. BENCH-009 borrows residual dictionaries but first
  subtracts the current nonlinear model's tangent and then demands finite/recovery realization.
- [GradMax](https://openreview.net/forum?id=qjN4h_wwUO) uses gradient structure and SVD to initialize
  new neural capacity without disturbing the learned function. It is a strong growth-selection
  threat, not a Gaussian-image implementation of the proposed assay.
- [MDL/stochastic complexity](https://research.ibm.com/publications/the-minimum-description-length-principle-in-coding-and-modeling)
  motivates pricing model state and data fit together. BENCH-009 reports DOF, trial packet bytes,
  and work separately because no trial code is yet an actual codec.
- [Arroyo--Ortiz local maximum entropy](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1534)
  and [Hormann--Sukumar maximum-entropy coordinates](https://www.inf.usi.ch/hormann/papers/Hormann.2008.MEC.pdf)
  provide the strongest unconventional method survivor: positive Gaussian-prior weights with a
  first-moment constraint reproduce affine fields exactly. The construction is old and direct;
  only its Gaussian-image/compression correspondence appears unexplored under this search.
- [Moving least squares](https://www.ams.org/mcom/1981-37-155/S0025-5718-1981-0616367-1/S0025-5718-1981-0616367-1.pdf)
  is the signed/closed-form first-order-consistency alternative to positive LME. It avoids the same
  convex-hull feasibility constraint but can introduce negative weights and color overshoot, so it
  belongs in the LME pre-gate as a robustness control rather than a second headline idea.

## 5. Frozen repository anti-library

| Tempting direction | Why it is not next |
|---|---|
| Smooth max/min or another smooth activation | The relevant nonsmooth operations are mostly discrete scheduling/selection, while the renderer's hard support has a semantic zero. C0 fade helped early AUC but lost final quality; vanilla smooth-maximum support would change tails and work. A new scalar smooth surrogate lacks a distinct prediction. |
| Another initialization or structure score | Initialization often washes out at long horizons; tensor-WSE missed the actual-rate gate; responsibility density failed recovery. |
| Another residual birth score | FIT-017's stronger immediate scores reversed after recovery; COMP-006's standard birth lost `-1.0714 dB` to the strongest actual-byte control. BENCH-009 uses a finite birth as a control, not a reopened selector. |
| Opacity gauge grouping | FIT-019 proved exact grouping commutation but rejected recovery utility. The grouping machinery remains an oracle only. |
| Recovery-response selector | FIT-020 found signal but rejected the frozen bend; no retuning on those fixtures is allowed. |
| More global precision mixes | COMP-006 already searched `875` mixes per field. Exact bytes improved fine selection (`+0.2131 dB`) but did not make standard birth competitive. |
| Generic predictor | FF-001 remains behind the hand prior; AIR/SGI-like amortization is a dataset-trained problem, not an MLP/activation tweak. |
| Direct full affine implementation | CORE-006 has a promising tiny smoke but no exact affine solve, codec, or native CUDA path. Spatially varying appearance is also heavily occupied by prior art. Diagnose the needed subspace first. |
| Direct WIPES transplant | WIPES already supplies the primitive. The open question is whether StructSplat residuals pay for carrier syntax/work versus affine or birth controls. |
| “Per-Gaussian backward” transplant | The exact CUDA kernel is already Gaussian-owned. Only block reduction/fusion and measurement remain. |
| Approximate top-K | Existing `K<=8` screens are lossy and `K=16` exposes only a one-field oracle ceiling without selection overhead. It is a performance approximation, not the next representation result. |

## 6. Functional problem signature

StructSplat infers a finite set of compact anisotropic kernels and attributes from a sampled image,
then decodes a normalized local mixture under rate, memory, and support-work constraints.

The unresolved hypotheses must remain separate:

| Deficit | Observable signature |
|---|---|
| Optimization | Residual projects strongly onto `J`; a finite current-grammar trust update realizes and retains the gain. |
| Representation | Residualized extension columns explain and realize energy that `J` cannot. |
| Allocation | Exact finite standard births beat richer extensions after recovery. |
| Rate allocation | Useful states exist, but cold-stream cost reverses DOF/parameter rankings. Not answered by BENCH-009. |
| Systems | Mathematics is adequate but gradient/update traffic dominates wall time. Independent profile lane. |
| Nonlocal/topological | All local projections predict poorly or reverse after recovery. |

This separation is the central assumption surgery. Earlier experiments repeatedly changed a local
score before determining whether the residual wanted an optimizer direction, a new atom, or a
finite topology change.

## 7. Survivor portfolio

The table reports apparent novelty of the **recipient-specific relationship**, not novelty of the
inherited method.

| Candidate | Class | Simple core | Strongest threat | Cheapest kill | Verdict |
|---|---|---|---|---|---|
| Residual tangent-space auction | N2 evidence program | Compare extension energy only after subtracting the current model tangent, then calibrate to finite/recovered gains. | LM/JVP, PUP, SteepGS, GradMax, OMP/column generation. | Small known-source fixtures, then `64x64`, `N=64` disjoint screen. | **Selected.** |
| Local max-entropy Gaussian compositor | N3-T correspondence | Exponentially tilt Gaussian responsibilities so their positive barycenter equals the query pixel. | LME and maximum-entropy coordinates are direct prior art. | No-ghost compact-support feasibility map plus exact affine reproduction. | Strongest post-auction method if affine wins. |
| Compact luma/affine jet | N1/N2-T grammar | Add the smallest local polynomial appearance statistic rather than a texture. | SVGS, Billboards, Textured Gaussians; CORE-006. | BENCH-009 affine arm. | Wait. |
| Fixed-envelope carrier grammar | N1 transfer | Add local phase/frequency degrees under an existing Gaussian envelope. | WIPES directly occupies the mechanism. | BENCH-009 carrier arm. | Wait. |
| Reduced-manifold appearance fitting | N1/N2 transfer | Eliminate constant/affine appearance coefficients exactly while optimizing geometry. | Variable projection and Gaussian LM. | Current-tangent win plus exact-vs-Adam convergence screen. | Wait. |
| Exact finite birth statistic | N1 diagnostic | Solve the finite normalized denominator/color change exactly for fixed new geometry. | Matching pursuit/exact line search; existing birth code. | Formula/direct-render parity and recovery. | Included as native control. |
| Decoder-synchronized Fisher companding | N2 relationship | Recompute curvature at the decoder and transmit only a global companding rule. | Sensitivity-aware Gaussian compression and RDO. | Real base-plus-correction cold stream. | Interesting but more expensive; wait. |
| Owner-computes fused exact update | N0/N1 systems | One Gaussian-owned block reduces, applies chain rule, and updates parameters. | Faster-GS; live exact kernel already owns a Gaussian. | Wall-time profile then block reduction. | Independent performance lane. |
| Quotient observability atlas | N2 evidence | Measure identifiable overlap-group Jacobian subspaces rather than parameter rows. | Fisher/observability analysis, FIT-019/020. | Explicit `N<=64` SVD/finite perturbations. | Partly absorbed by BENCH-009 scaling/gauge audit. |
| Support-conflict-coloured local LM | N1 transfer | Update non-overlapping support-graph colors independently with local second-order solves. | 3DGS-LM, matrix-free LM, domain decomposition. | Conflict density plus explicit-Jacobian smoke. | Low novelty and likely SSIM coupling; wait. |

### Idea card A — residual tangent-space auction

**Central claim.** A recovered, rate/DOF-labelled projection assay can determine whether remaining
StructSplat error is optimization-, affine-, frequency-, or allocation-limited before a new grammar
is implemented.

**Known foundation.** Gauss--Newton/LM, variable projection, matching pursuit/OMP, gradient-based
growth, Fisher sensitivity, and MDL/RDO.

**Irreducible delta.** One common residual currency for heterogeneous current-grammar and extension
actions, explicitly residualized against existing capability and calibrated by finite/recovery
behavior under StructSplat's normalized compositor.

**Why not A+B.** Projection alone is known and richer atoms are known. The test matters only if the
same frozen assay separates those causal explanations and predicts finite recovered outcomes; that
recipient-specific discrimination is not supplied by either component.

**New prediction.** Underfit parents will show current-tangent capacity, whereas near-plateau
parents will expose a stable family-specific extension only if a representation deficit remains.

**Cheapest kill.** Known-source Stage 0 plus the preregistered disjoint `64x64` screen. Kill if
rankings are damping/trust sensitive or fail predicted-versus-realized/recovery gates.

**Novelty confidence.** `0.55--0.75` that the complete evidence protocol is unexplored under the
stated search; low confidence for any component-level novelty. Search cutoff 2026-07-15.

**Scientific value if negative.** Closes expensive affine/carrier implementation or shows local
linear capacity is the wrong abstraction.

### Idea card B — positive linearly precise Gaussian compositor

**Central claim.** Enforcing a first-moment constraint on positive Gaussian responsibilities lets
constant per-center RGB reproduce any affine image without per-Gaussian slopes.

**Known foundation.** Local maximum-entropy meshfree approximation and maximum-entropy barycentric
coordinates with Gaussian priors.

**Irreducible delta.** Application to a moving, anisotropic, compact-support Gaussian image codec,
including its boundary grammar, implicit backward, and rate/work evidence.

**Why not A+B.** Merely replacing softmax with “max entropy” is known. A contribution exists only if
the compact-support/boundary incompatibility is solved without hidden state and the no-extra-
appearance-payload prediction survives a real stream/work comparison.

**New prediction.** On feasible interiors, a constant-color field sampled at Gaussian centers has
machine-precision ramp reconstruction; any remaining error concentrates at convex-hull/support
failures rather than throughout the ramp.

**Cheapest kill.** With exactly current positive contributors and no ghosts, map convex-hull
feasibility, moment residual, conditioning, Newton iterations, and boundary-band failures. Any
infeasible scored pixel kills vanilla LME as a universal renderer.

**Novelty confidence.** `0.25--0.50`; donor method is direct and old, while the recipient
correspondence appeared absent in bounded paper/code search. Patent/non-English search incomplete.

**Scientific value if negative.** Establishes why compact Gaussian image supports and positive
linear precision conflict at boundaries.

### Idea card C — decoder-synchronized Fisher companding

**Central claim.** A decoder-recomputable curvature field can allocate correction precision without
transmitting a per-row importance map.

**Known foundation.** Reverse water-filling, transform-coder RDO, Fisher/Hessian sensitivity, and
sensitivity-aware Gaussian compression.

**Irreducible delta.** Both encoder and decoder deterministically recover the same normalized-
renderer curvature from the decoded base and one global rule.

**New prediction.** It beats COMP-006's best global mix only where curvature heterogeneity is high,
while spending no importance-map bytes.

**Cheapest kill.** A real base-plus-correction stream on new targets; kill if curvature
recomputation is unstable after quantization or the global precision box already matches it.

**Verdict.** More expensive and prior-art-threatened than BENCH-009; do not start yet.

## 8. Cross-domain transfer audit

| Transfer | Preserved mechanism | Broken correspondences requiring invention | Recipient-specific prediction | Native competitor |
|---|---|---|---|---|
| Separable nonlinear LS -> residual assay | Linear appearance variables can be eliminated conditional on geometry. | Compact support changes discretely; SSIM is not least squares; quantization and color bounds break exact linear solves. | Tangent-explainable residual should realize under a bounded current-grammar update. | Existing constant-color CG solve and Adam. |
| Matching pursuit/column generation -> action auction | Compare residual decrease from candidate atoms/actions. | Normalized births change the denominator; atom families have unequal searches/bytes/work; immediate decrease can reverse in recovery. | Exact finite birth rankings differ most in high overlap, but must persist after recovery. | Residual/support births and COMP-006 controls. |
| MDL/RDO -> family pricing | Additional fit must pay for model description. | No rich-atom stream exists; container deltas are non-additive; decoder work is not bits. | A family winning only by unpriced frequency/index choices disappears under cross-fitting/packet pricing. | SSPL1 cold-stream oracle. |
| LME meshfree coordinates -> compositor | Positive partition and first-moment consistency give affine precision. | Queries may leave active-center convex hull; exact-zero support contradicts global positive priors; centers move; per-pixel Newton/implicit backward is costly. | Interior ramps become exact with constant colors, while failures localize to support/boundary infeasibility. | Current normalized and affine renderers. |
| Owner-computes GPU reductions -> exact CUDA | One work owner reduces repeated atomic writes and fuses nearby operations. | StructSplat is a normalized 2-D sum, already Gaussian-owned in exact backward, and has different support/count regimes. | Benefit scales with active threads/final atomic contention, not simply pixel--Gaussian overlap. | Existing exact kernel and tiled path. |

The LME mapping is the rarest and most conceptually attractive transfer. It is not selected because
its main donor assumption—query inside the convex hull of positive active nodes—is exactly what
compact image-boundary supports often violate.

## 9. Pareto decision

Scores are `0` (poor) to `5` (strong). First-test cost is scored high when cheap. No weighted mean
was used.

| Candidate | Apparent novelty | Falsifiability | Importance | Feasibility | Cheap first test | Interpretability | Strong native control | Informative failure | Publication path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Residual tangent auction | 3 | 5 | 5 | 4 | 4 | 5 | 5 | 5 | 4 |
| LME compositor | 3 | 5 | 5 | 2 | 5 | 4 | 5 | 5 | 4 |
| Compact affine jet | 2 | 5 | 4 | 3 | 4 | 5 | 5 | 3 | 3 |
| Local carrier | 1 | 5 | 4 | 2 | 3 | 5 | 5 | 3 | 2 |
| Reduced-manifold fitting | 1 | 5 | 4 | 4 | 4 | 4 | 5 | 4 | 2 |
| Fisher companding | 3 | 4 | 5 | 2 | 2 | 4 | 5 | 4 | 4 |
| Fused exact update | 1 | 5 | 3 | 4 | 5 | 5 | 5 | 3 | 2 |

The auction is selected because it dominates on uncertainty reduction and negative-result value.
LME has the most elegant method core but assumes the answer—affine precision is binding—before the
repository has established it.

## 10. Independent adversarial kill memo

The adversary rejected the following proposed shortcuts:

1. **Zero-opacity birth tangent:** invalid because color and opacity are bilinear/degenerate at
   zero. BENCH-009 instead solves the exact finite normalized birth.
2. **One blended price:** DOF, stream bytes, fit work, and decode work are different resources and
   are reported separately.
3. **Projected gain as result:** prohibited; every ranking needs a finite trust render and matched
   recovery.
4. **Free carrier search:** frequency/orientation/type/index choices are recorded, cross-fitted,
   and provisionally priced.
5. **Full `J` versus six scalars as equal-rate:** prohibited; full tangent is an unpriced
   optimization-capability upper bound, not a codec competitor.
6. **LME first:** rejected until affine residual capacity wins and compact no-ghost feasibility is
   established.
7. **Performance priority based on Faster-GS:** downgraded because exact backward is already
   Gaussian-owned in live code.

The adversary would kill the auction itself if rankings change materially with parameter units,
damping, trust radius, row gauge, or candidate-bank size; if predictions do not realize; or if
synthetic identity trivially succeeds while natural/mixed targets disagree without reproducible
strata.

## 11. Selected killing experiment

The full preregistration is in BENCH-009. Its essential sequence is:

1. Create known-source current-tangent, affine, carrier, finite-birth, and null fixtures.
2. Validate centered finite-difference/JVP relative error `<=1e-3`.
3. Scale parameter units, solve/project the current tangent, and residualize each extension.
4. Compare one six-RGB-slope affine block, one six-RGB sine/cosine block, and two three-RGB finite
   births; report the full current tangent only as an optimization upper bound.
5. For births, use the exact normalized sufficient statistic

   ```text
   a = u / (D + u + eps)
   y_new = (1-a)y + a c
   c* = sum a[t-(1-a)y] / sum a^2
   ```

   per channel, or the joint two-column solve for two births.
6. Cross-fit candidate selection on frozen spatial folds and report bank-size sensitivity.
7. Apply two frozen dimensionless trust radii; require predicted/realized Spearman `>=0.8` and
   median gain ratio in `[0.5,2]`.
8. Run matched `20`- and `100`-step recovery.
9. On disjoint `64x64`, `N=64` underfit and near-plateau parents, require a richer family to beat
   the strongest tangent/birth control by `>=25%` residual energy and `>=0.15 dB`, with a family-
   bootstrap lower bound above zero, positive in at least four of six families, and persistent
   after recovery.
10. If the assay is unstable or no family survives, stop. Do not tune its target bank, damping,
    trust radii, horizons, or recovery steps.

### Interpretation contract

| Winner | Authorized next work |
|---|---|
| Existing tangent | Reduced-manifold/variable-projection optimizer experiment. |
| Affine | No-ghost LME feasibility pre-gate, then smallest versioned luma/affine grammar if needed. |
| Carrier | WIPES-controlled versioned carrier grammar. |
| Finite birth | New discrete allocation/search hypothesis, not a retune of failed birth scores. |
| No stable prediction | Stop local linear action selection; investigate topology/nonlocal dynamics. |
| Nothing | Stop richer-atom work on this frontier. |

## 12. Current empirical status

Stage 0 passed its five self-generated controls, JVP comparison, and packet-realization checks.
Stage 1 then completed all frozen immediate and recovery ledgers: `4,608` immediate cells, `816`
recovery trajectories, `2,448` logical recovery checkpoints, and the complete matched-evidence
union. The independent audit confirms exact bindings, parent hashes, shard unions, manifests, and
step-0 joins.

The scientific result is negative/unavailable. Global causal calibration is `0.268549 < 0.8`, all
causal action-by-horizon strata fail, and the independently truncated base/joint projectors yield
negative incremental energies. Affine and carrier both lose immediately and at step 20 against
their stronger matched control; carrier's positive step-100 mean is radius-sensitive and cannot
pass the frozen survival decision. See
[`2026-07-16-bench009-results-audit.md`](2026-07-16-bench009-results-audit.md) for the exact claim
audit and provenance limitations.

BENCH-011 v1 repaired the algebra and completed `96/96` native rows, but did not reproduce the
exact BENCH-009 factorization seed and is retained only as an invalid-run diagnostic. Corrected v2
binds every exact unit ID/base rank and replays every prediction/render bit-exactly. Its four
calibration strata all fail: correlations are at most `0.400`, and every radius-`0.75` row loses.

## 13. Requested-axis conclusion

| Axis | Status from this pass | Reason |
|---|---|---|
| Quality | **No promoted improvement.** | Both richer candidates fail the frozen development utility screen. |
| Convergence | **No demonstrated improvement.** | Carrier recovers late descriptively, but fails the immediate/step-20 and both-radius requirements. |
| Performance | **Unavailable.** | Search/range-factorization work is outside the persisted action timing. |
| Compression | **Unavailable.** | No rich-atom complete stream exists; provisional packet pricing is not actual rate. |
| Expressiveness | **Unavailable; current local-linear branch closed.** | BENCH-009's projector invariant fails; corrected BENCH-011 v2 is coherent but uncalibrated to native held-out gain. |

## 14. Audit limitations

- The search used primary papers, official proceedings/project pages, official repositories, and
  original mathematical sources through 2026-07-15. Patent, exhaustive thesis, non-English, and
  closed IEEE full-text coverage is incomplete.
- The apparent novelty of BENCH-009 is only the full evidence protocol/common currency. Its linear
  algebra, sensitivity tools, atom selection, and pricing principles are known.
- The LME correspondence may exist under terminology such as reproducing kernels, corrected
  Shepard interpolation, moving least squares, entropy coordinates, or meshfree shape functions.
- Results across 2-D image fitting, 3-D novel-view synthesis, raw parameter BPP, and complete
  entropy-coded streams remain protocol-specific.
- A successful oracle is necessary but not sufficient for a useful codec or renderer. Syntax,
  quantization, decoder work, backward parity, and broad confirmation remain separate gates.

## 15. Recommended action

**Keep every production/default path unchanged and do not retune BENCH-009/011.** The next
scientific question, if pursued, must use finite count-neutral topology changes under one aligned
objective and carried optimizer state. It must be preregistered on disjoint synthetic structure
before any natural-image or codec claim.

### Independent systems-lane update

The source-bound RTX-3050 microprofile in
[`benchmarks/exact_backward_profile.py`](../../benchmarks/exact_backward_profile.py) measured the
existing untiled exact backward at `1.120256 ms`, or `33.1719%` of a `3.377120 ms` representative
device-side micro-fit step at the frozen `256x256`, `N=2048`, overlap-16 cell. This passed the
preregistered `>25%` actionability gate. It authorizes PORT-004's opt-in within-block reduction
experiment; it does not demonstrate a speedup. Hardware-counter attribution remains unavailable
because Nsight Compute returned `ERR_NVGPUCTRPERM`.
