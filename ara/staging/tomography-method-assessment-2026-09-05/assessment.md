# Assessment of realtime-gs tomography from compact Gaussian observations

Date: 2026-09-05. Scope: the user-authorized methodological assessment, including the instruction to continue. Status: provisional, self-reviewed code/literature assessment. No new reconstruction benchmark, calibrated-scene run, implementation change or method selection.

Inspected sibling revision: `2ebe52cc28a1b0e812a6c436d49225fac4ee943f`, clean during inspection. The accompanying CPU worksheet records hashes of the imported operators. Historical claims were read, not independently replayed.

## Assessment

Retain the compact-observation pipeline and Beam Fusion as an initialization candidate. The first correction to prepare is the refitter's representation-dependent RGB weighting. The first substantive research comparison should then isolate the source-footprint constraint under a clearly declared projection/appearance objective. Another optimizer or a correspondence tracker would not answer these questions.

The repository already implements whole-field analytic matching, covariance observability/repair, and a separate compact-only sampled rendering objective. These must not be proposed as missing features. The open issue is how their assumptions fit together, and which assumptions cost downstream quality.

## Findings and evidence scope

| ID | Finding | Evidence and disposition |
|---|---|---|
| A1 | The observation field's normalized image weights are not identified physical attenuation or emitted-density measurements. | Confirmed code semantics; physical interpretation remains an assumption. `observation2d.py:134,506`, StructSplat `render.py:3,234`. |
| A2 | Beam covariance intersection is an initialization/fusion rule, not exact covariance inversion. | Derived orthographic counterexample, checked using `_ci_fuse`; not a measured dome defect. |
| A3 | The analytic field loss is exact for its additive whole-plane proxy. | Existing gradient/split tests pass; does not make it an exact normalized or visibility-composited RGB loss. |
| A4 | The refitter's RGB normalization breaks exact target-field representation invariance. | Confirmed isolated CPU counterexample against `_variable_field_objective`; current tests do not cover this layer. |
| A5 | The continuous refitter fixes each primitive to one source footprint and source color. | Confirmed parameterization; whether relaxing it helps real captures is untested here. |
| A6 | Visibility and gains make the refitter a block-updated surrogate, not a fixed linear tomographic inverse. | Confirmed implementation; do not infer a single globally decreasing physical objective from accepted local steps. |
| A7 | The compact trainer already compares sampled student rendering with native compact-teacher colors. | Confirmed code path; a useful existing objective for evaluating proxy transfer, not new quality or memory evidence. |

## 1. Specify the observation and forward operator together

For the maintained normalized observation model,

`F_v(u) = N_v(u) / (D_v(u) + epsilon_v)`.

Its denominator is a fitted basis-weight sum. Where it dominates epsilon, scaling numerator and denominator together changes color little while changing the denominator substantially. At fixed positive epsilon this is approximate, not an exact global gauge symmetry. A successful RGB fit therefore does not establish that `D_v` is a physical projection of one common 3D density. Additive fitting removes the quotient but does not by itself remove occlusion from the captured RGB.

Source: [native observation semantics](/home/alex/Documents/realtime-gs/src/rtgs/core/observation2d.py:134), [query](/home/alex/Documents/realtime-gs/src/rtgs/core/observation2d.py:506), [StructSplat renderer](/home/alex/Documents/structsplat/src/structsplat/render.py:234).

For a true parallel-ray line-integral control, let a unit-peak 3D Gaussian be

`rho(x) = a exp[-(x-mu)^T Sigma^-1 (x-mu)/2]`.

With unit ray direction `d` and an orthonormal detector-plane basis `P`, its projected covariance is `C = P Sigma P^T`, but its peak is

`a_projected = a sqrt(2 pi / (d^T Sigma^-1 d))`.

A mass-normalized Gaussian has a different coefficient convention: its total mass survives marginalization and the 2D peak includes `1 / (2 pi sqrt(det(C)))`. Neither convention permits silently treating a peak coefficient as integrated mass.

The current [predicted field](/home/alex/Documents/realtime-gs/src/rtgs/lift/field_refit.py:245) instead uses `field_masses * center_visibility * view_gain` as a peak coefficient with EWA-projected geometry. This is a declared appearance/density proxy, not the line-integral operator above. A single scalar view gain cannot generally supply a different missing covariance factor for every component. Adding that factor blindly would also be wrong if the observations retain their present RGB-weight semantics.

[R2-Gaussian](https://arxiv.org/html/2405.20693v2), section 4.2.1, derives the covariance-dependent amplitude correction and additive composition for CT. Its cone-beam projection still uses a local affine approximation. [Radioactive 3D Gaussian Ray Tracing](https://arxiv.org/html/2602.01057v1) supplies analytical ray integrals. These are useful projection references, not permission to reinterpret ordinary RGB as attenuation data. For dome radiance, the observation model must account for visibility and appearance; source images can still remain absent by querying the retained fields.

## 2. Beam fusion localizes evidence but does not solve covariance inversion

[Beam Fusion](/home/alex/Documents/realtime-gs/src/rtgs/lift/beam_fusion.py:1) lifts a transverse footprint at an implied depth and adds a finite along-ray variance. That is a local Gaussian beam approximation; a perspective footprint's world width varies with depth. Pair gates, per-view nearest-contributor fold-in and reduction determine which hypotheses survive. Consequently the implementation is not a global optimization over anonymous whole projection fields, even though it does not use an RGB optical-flow matcher.

The implemented [CI rule](/home/alex/Documents/realtime-gs/src/rtgs/lift/beam_fusion.py:283) averages precision matrices. Consider three perpendicular orthographic views of a centered isotropic Gaussian with true covariance `s^2 I`. Each beam has transverse variance `s^2` and longitudinal variance `L^2`. Equal-weight CI returns

`Sigma_CI = [3 / (2/s^2 + 1/L^2)] I`.

As `L` grows, this approaches `1.5 s^2 I`, despite the three exact projected covariances identifying `s^2 I`. A naive precision product returns the opposite error in this control, approaching `0.5 s^2 I`. Neither operation is exact inversion.

The worksheet checks the actual `_ci_fuse` function at `s=1, L=1000`: each returned diagonal is `1.499999250000375`. This probes the fusion rule only, with no perspective cameras, clipping or full Beam pipeline. It establishes neither a 50% error on dome scenes nor a benefit from replacing the initializer.

CI's usual covariance-consistency interpretation concerns uncertainty in estimates. A physical splat footprint is a different object; that theorem should not be imported without proving its premises. The repository's existing CI test is an isotropic fixture with loose eigenvalue bounds, not a general anisotropic consistency proof.

Existing remedies already include [covariance repair](/home/alex/Documents/realtime-gs/src/rtgs/lift/carrier_refinement.py:321), `beam_covariance_refit.py`, and rank-aware geometry. Sibling C23 limits two-view EWA covariance identifiability; C35 records a scoped positive exact-synthetic rank-aware result. Neither needs rediscovery as a new method.

## 3. Confirmed representation-invariance defect in the refit objective

[field_loss.py](/home/alex/Documents/realtime-gs/src/rtgs/lift/field_loss.py:246) correctly evaluates additive field L2 through Gaussian inner products. For identical means and covariances, replacing a component with two copies at half its coefficient leaves that field exactly unchanged.

The [refitter](/home/alex/Documents/realtime-gs/src/rtgs/lift/field_refit.py:326) subsequently divides its RGB term by

`sum_j ||rgb_amplitudes_j||^2`.

For a one-component target with coefficient vector `b`, that denominator changes from `||b||^2` to `2 ||b/2||^2 = ||b||^2 / 2`. The field-dependent numerator stays unchanged. Thus its RGB contribution and gradient double. Density weighting is unchanged in this exact split example, so the relative density/RGB objective changes too.

CPU check against the imported functions:

| Quantity | One target component | Two identical half-weight copies |
|---|---:|---:|
| Unweighted density L2 | 0.16622148882085042 | 0.16622148882085042 |
| Unweighted RGB-numerator L2 | 0.11500432472717392 | 0.11500432472717392 |
| Current RGB mean-gradient x | 0.151614960972703 | 0.303229921945406 |
| Current RGB mean-gradient y | 0.056314128361289684 | 0.11262825672257937 |
| Target coefficient energy | 0.35840000000000005 | 0.17920000000000003 |
| Target field energy | 1.125946807046582 | 1.125946807046582 |

This contradicts representation invariance at the composite refit-objective layer. It does not show that Adam takes exactly double the parameter step, or establish a downstream quality loss. Existing colocated-split tests cover the underlying L2 helper and therefore pass.

Recommended correction to prepare: use explicit fixed per-view RGB weights independent of componentization, or a fixed normalization derived from target-field energy. Cache any target-only calculation; computing exact target energy adds pairwise work that must be charged. Choose and freeze the normalization semantics before changing a benchmark objective. The regression check must exercise `_variable_field_objective` and its gradients, not only `field_l2`. No patch was applied during this assessment.

## 4. Whole-field matching still has a source-component constraint

[InverseProjectionFiber](/home/alex/Documents/realtime-gs/src/rtgs/lift/inverse_projection_fiber.py:95) fixes one source 2D mean and covariance per 3D row, leaving depth and the remaining covariance coordinates trainable. `fit_field_fibers` also uses source-anchored appearance and fixed field coefficients/render opacity within its continuous stage. The whole-field objective avoids assigning a target component to every projection, but the feasible 3D models are still tied to selected source components.

That equality is appropriate for an ideal projected primitive. For an independently fitted image basis, it may force one 3D primitive to explain a patch chosen for texture reconstruction, even when several scene elements or an occlusion boundary produced it. Whether that restriction is currently the limiting factor is unresolved.

The nearby [field lifter](/home/alex/Documents/realtime-gs/src/rtgs/lift/field_lifter.py:1) is a separate placement/fiber/refit pipeline. Do not imply the Beam-to-compact carrier path necessarily invokes `_variable_field_objective`; the identified defect is scoped to callers of the field-refit objective.

## 5. Visibility changes the inverse problem

[Center visibility](/home/alex/Documents/realtime-gs/src/rtgs/lift/field_visibility.py:48) samples incoming transmittance at each splat center, freezes it for a block, and can force valid source entries visible. This can approximate visibility that is nearly constant across a footprint. At an occlusion boundary, multiplying a full Gaussian by its center visibility does not equal spatially varying alpha compositing.

`_gain_for_view` solves a density-only gain problem. When the RGB term is active, it is not the exact variable-projection solution of the combined density-plus-RGB objective. Visibility/gain refresh, appearance activation and active-view changes also alter the function represented in `objective_history`. Accepted-step monotonicity is local to the current block and settings. These are surrogate-design properties, not evidence that the entire implementation is incorrect.

The separate [compact trainer](/home/alex/Documents/realtime-gs/src/rtgs/optim/compact_trainer.py:1635) already queries native teacher colors and compares them to student rendering at explicit points. This gives an existing way to evaluate whether tomographic proxies are helping the intended visible-color task without returning source RGB to the training worker. Sampling weights, masks, support and the evaluation measure still have to match the declared experiment.

## 6. Literature transfer and next assessment boundary

[Zickert, Oktem and Yarman (2022)](https://doi.org/10.1088/1361-6420/ac8bee) formulate joint Gaussian-dictionary learning and tomographic reconstruction and use a Gaussian-specific filtered-backprojection initializer. Its useful lesson is to derive an inverse and data-space objective from the same declared forward operator. Its parallel-beam simulated setting does not establish the RGB extension or exact inversion by beam products. Publisher PDF access was blocked; the institutional indexed abstract and indexed original manuscript were available.

[Panaretos (2009)](https://arxiv.org/abs/0909.0349) and [Panaretos/Konis (2011)](https://arxiv.org/abs/1202.6475) bound broad mixture-tomography novelty under radial-kernel and labeling assumptions. [FaCT-GS](https://papieta.github.io/fact-gs/) contributes scalable CT machinery and warm starts. Their practical gains cannot be transferred to this RGB pipeline without measurement.

Recommended order, not a selected experiment protocol:

1. Prepare the narrow RGB-weighting correction and its function-level invariant test. Keep the historical objective and its results identifiable.
2. Retain Beam Fusion and the existing covariance repairs as controls. Establish an explicit additive ray-integral control separately from the current EWA/visibility proxy; do not mix their amplitude laws.
3. On a shared fixed initialization and frozen observations, compare hard source-footprint equality with a controlled relaxation, using the same declared objective, capacity, optimizer and evaluation. Begin with known-generator diagnostics, acknowledge existing ideal/split-field tests, and then require calibrated development evidence for practical conclusions. Use a separate source-RGB control/evaluator for attribution, with no source access in the compact worker.
4. Use native compact-rendered color error alongside proxy-field error to detect improvement in a surrogate without improvement in the desired reconstruction. Charge conversion, initialization, any cached target energies, training and all retained state.

The smallest new research question is whether a source-anchored feasible set loses useful solutions when the observations are fitted image fields, after removing the known representation-dependent weighting defect. This is more specific than re-running another generic reverse-projection success test. Sibling C22's failed raw-fragment transport release and C37's negative association result remain intact; they neither authorize a retry nor answer the proposed matched constraint comparison.

## Verification and limits

- 37 existing CPU tests passed across beam fusion, additive field losses, field refit, covariance repair and the two-/three-view rank check. Exact command and raw output are in `verification.json`.
- The retained `algebra_checks.py` verifies the split counterexample, a representation-invariant target-energy quantity, the orthographic CI formula, the line-integral amplitude convention and the qualified normalized-color gauge example. Its JSON records Python/torch versions and source hashes.
- No scene assets, source photographs, held-out cameras or GPU were used. The algebra worksheet is a correctness audit, not a formal reconstruction experiment or new performance evidence.
- Self-review is provisional. No implementation, default, task ownership, historical result, scientific claim status, or sibling repository file was changed.
