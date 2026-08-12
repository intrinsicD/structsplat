# Research Portfolio: Pixel-gradient anatomy and operator choice inside HIER

**Repository/domain:** StructSplat HIER, especially the masked, constant-color, pure-additive
four-array field selected by HIER-031 and carried into HIER-032.

**Literature cutoff:** 2026-08-12.

**Sources searched:** the live StructSplat renderer, fitter, Gaussian parameterization, HIER and
FIT task/evidence history; primary papers or official proceedings pages for
[AbsGS](https://arxiv.org/abs/2404.10484),
[Revising Densification](https://arxiv.org/abs/2404.06109),
[GDAGS](https://arxiv.org/abs/2508.09239),
[SteepGS](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.html),
[Splitting Steepest Descent](https://proceedings.neurips.cc/paper/2019/hash/3a01fc0853ebeba94fde4d1cc6fb842a-Abstract.html),
[COB-GS](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_COB-GS_Clear_Object_Boundaries_in_3DGS_Segmentation_Based_on_Boundary-Adaptive_CVPR_2025_paper.html),
[Faster 3DGS Convergence via Structure-Aware Densification](https://arxiv.org/abs/2604.28016),
[LocoADC](https://arxiv.org/abs/2607.17896),
[REFINE](https://arxiv.org/abs/2606.09074), and
[Improving Densification](https://arxiv.org/abs/2508.12313); plus functional searches around
matching pursuit, adaptive approximation, split/merge mixture models, influence, and leverage.

**Status:** mathematical diagnosis and experiment proposal only. No production method, public
API, default, or quality claim is selected here.

**Key unresolved assumptions:** compact streamed summaries retain enough information to predict a
discrete edit; an immediate action oracle remains informative after recovery; L2 diagnostic
structure transfers to HIER's masked L1-plus-SSIM objective; and a useful operator label is stable
enough to learn or threshold.

## Executive answer

Yes: the pixels covered by a Gaussian contain substantially more topology information than the
ordinary summed gradient exposes. The mathematically useful object is the set of
**pixel–Gaussian gradient contributions before reduction**. For parameter family \(z_i\),

\[
g^{z}_{ip}
  =
  \left(\frac{\partial C_p}{\partial z_i}\right)^{\mathsf T}
  \frac{\partial L}{\partial C_p},
\qquad
g^{z}_{i}=\sum_p g^{z}_{ip}.
\]

The final gradient \(g_i^z\) is only the vector sum. Its small magnitude has several possible
causes:

1. large contributions point in opposing directions and cancel;
2. the pixels are outside every active support, so no contribution exists;
3. the current Gaussian color cannot affect the erroneous color direction;
4. the residual is orthogonal to the span or tangent space of the current basis;
5. the raw error and optimized loss have different sensitivities, especially under L1 and SSIM;
6. image and regularization gradients cancel;
7. overlapping Gaussians create an ill-conditioned or non-identifiable parameterization; or
8. the field is genuinely stationary and the proposed edit has no value.

Therefore, **high error plus a small aggregate gradient is not by itself a split rule**.
Cancellation becomes split evidence only when its spatial pattern is compatible with a bipolar
refinement and either the additive split matrix has a negative eigenvalue or an exact finite split
probe wins. Symmetric width error can produce the same small mean gradient but asks for a scale
change; an orientation mismatch asks for rotation; an unsupported pixel asks for birth; and a
residual orthogonal to all current basis columns asks for dictionary expansion rather than moving
an existing row.

For HIER, the most promising controller is consequently a two-stage system:

1. stream a compact contribution packet containing signed, absolute, directional, and
   parameter-family moments; then
2. compare legal move, reshape, split, birth, merge, prune, and teleport transactions in the
   actual rendered-loss currency, with exact trials for the finalists.

This is a task-specific synthesis of known ingredients, not a novelty claim. AbsGS already exposes
pixelwise cancellation, GDAGS already uses coherence to route clone versus split, SteepGS already
uses split curvature, LocoADC already combines regional 2D error allocation with merging, and
REFINE already uses second-order removal importance. The open question is whether their signals
can identify the right operation under HIER's additive, compact-support, exact-count semantics.

## 1. Frontier map

| Work or local component | Signal retained | Decision made | What it establishes | Gap for HIER |
|---|---|---|---|---|
| Original 3DGS ADC | aggregated view-space mean-gradient norm and scale | clone or split | gradient magnitude can trigger growth | pixel cancellation and scale-based routing are ambiguous |
| AbsGS | absolute pixelwise positional contributions before summation | split high-score large rows | cancellation is real and actionable | loses direction, family structure, and alternative operators |
| GDAGS | net-to-absolute gradient coherence ratio | low coherence promotes split; high coherence promotes clone | direction adds information beyond absolute mass | directly occupies simple coherence routing and does not certify the edit |
| Revising Densification | auxiliary per-pixel error assigned to contributors | error-driven densification | error sees failures that row gradients miss | error is not an operator value |
| COB-GS | sign consistency of semantic pixel gradients | split boundary-ambiguous rows | foreground/background pixels can cancel on one Gaussian | semantic boundary case, not a general reconstruction router |
| SteepGS | per-Gaussian splitting matrix and minimum eigenpair | split only along negative curvature | symmetric splitting is a second-order action | known split-only result; HIER has different renderer and fixed count |
| Structure-aware densification | footprint versus per-axis local frequency | anisotropic split | position gradient confounds misplacement and aliasing | external frequency signal, not a full operator grammar |
| LocoADC | regional distortion magnitude/coherence and local Gaussian similarity | regional births and merges | locality matters in 2D Gaussian image allocation | does not use pixel–Gaussian parameter contributions or exact donor cost |
| REFINE | approximate rendering-aware Hessian importance | prune | stationary gradient is not removal importance | approximate 3D removal only; no destination transaction |
| StructSplat current absgrad | absolute value of the already-reduced mean gradient | duplicate selected rows | existing integration point | cancellation has already occurred before the absolute value |
| StructSplat residual/support/responsibility selectors | pixel error, support, or normalized ownership | configured birth/split wave | several complementary diagnostics already exist | selector and operation remain coupled heuristically |
| HIER-032 | explicit coverage deficit plus exact donor/placement tests | funded coverage closure | support holes need explicit candidates and donor accounting | closure succeeded geometrically but did not preserve every quality gate |

The closest prior-art threat is GDAGS: its gradient coherence ratio is exactly the scalar
\(\|\sum_p g_{ip}\|/\sum_p\|g_{ip}\|\), and it explicitly maps conflict to split and alignment to
clone. The HIER research space is narrower:

- derive the full packet for an additive 2D Gaussian rather than only mean coherence;
- distinguish mean, scale, rotation, and color moment patterns;
- use the exact additive split matrix where applicable;
- detect tangent-space or basis-span deficiency for birth;
- price donor damage for exact-count topology changes; and
- measure whether the packet predicts exact actions after recovery.

## 2. Functional problem signature

### 2.1 Five different objects are often all called “the gradient”

They should be kept separate:

| Object | Definition | Meaning |
|---|---|---|
| RGB residual | \(r_p=T_p-C_p\) | reconstruction discrepancy, independent of the chosen loss derivative |
| spatial error gradient | \(\nabla_x \|r_p\|\) | how error changes across neighboring image coordinates; LocoADC uses this kind |
| image adjoint | \(u_p=\partial L/\partial C_p\) | what infinitesimal RGB output change lowers the actual objective |
| pixel–Gaussian contribution | \(g^z_{ip}=J_{ip}^{z\mathsf T}u_p\) | pixel \(p\)'s contribution to parameter family \(z_i\) |
| parameter gradient | \(g_i^z=\sum_p g^z_{ip}\) | the reduced update signal exposed by ordinary autograd |

“Per-pixel gradient contribution” is the safest term. AbsGS and GDAGS use “pixelwise
sub-gradient” for the same decomposition. “Subgradient” is also technically appropriate for the
L1 term at its nondifferentiable zero, where PyTorch selects one admissible value. It is less exact
for the mixed objective because SSIM couples neighborhoods: there may be no independent scalar
loss \(L_p\) whose derivative is \(g_{ip}\). The output adjoint \(u_p\), however, is well defined,
and decomposing the renderer's vector–Jacobian product by pixel is exact.

### 2.2 The actual selected HIER renderer is additive

HIER-031 and HIER-032 persist four arrays—mean, log-scale, rotation, and RGB coefficient—and use
the compact-support additive renderer, not the repository's normalized default. Let

\[
\delta_{ip}=x_p-\mu_i,\quad
A_i=R_i
\begin{bmatrix}s_{ix}^{-2}&0\\0&s_{iy}^{-2}\end{bmatrix}
R_i^{\mathsf T},
\]

\[
G_{ip}=\exp\left(-\tfrac12\delta_{ip}^{\mathsf T}A_i\delta_{ip}\right),
\qquad
\kappa=\exp(-\tfrac12\tau^2),
\qquad
f_{ip}=[G_{ip}-\kappa]_+.
\]

With constant RGB coefficient \(c_i\),

\[
C_p=\sum_i f_{ip}c_i.
\]

The fade is important. Inside active support, derivatives of \(f\) are derivatives of the raw
\(G\), not \(f\); outside support they are exactly zero. At the clamp boundary the derivative is a
chosen one-sided/subgradient value. The renderer also detaches the integer support rectangle used
for tile enumeration, so changing which pixels enter a tile is not differentiated. A coverage
hole is therefore invisible to every existing row gradient.

Define local coordinates

\[
\xi_{ip}=R_i^{\mathsf T}\delta_{ip}
\]

and scalar geometric pressure

\[
h_{ip}=u_p^{\mathsf T}c_i.
\]

For an active-support pixel, the exact contribution formulas are

\[
g^c_{ip}=f_{ip}u_p,
\]

\[
g^\mu_{ip}
  =G_{ip}h_{ip}A_i\delta_{ip},
\]

\[
g^{\log s_x}_{ip}
  =G_{ip}h_{ip}\frac{\xi_{ip,x}^2}{s_{ix}^2},
\qquad
g^{\log s_y}_{ip}
  =G_{ip}h_{ip}\frac{\xi_{ip,y}^2}{s_{iy}^2},
\]

\[
g^\theta_{ip}
  =
  G_{ip}h_{ip}\xi_{ip,x}\xi_{ip,y}
  \left(\frac{1}{s_{iy}^2}-\frac{1}{s_{ix}^2}\right).
\]

These equations immediately expose several blind spots:

- A large RGB error can exert no geometry pressure when \(u_p\) is orthogonal to \(c_i\).
  Moving a red basis function cannot directly repair a green-only output demand.
- Symmetric pixels can cancel the first spatial moment \(g^\mu_i\) while reinforcing the
  log-scale gradient.
- Rotation sees an off-diagonal second moment and vanishes for an isotropic Gaussian.
- A pixel outside support has \(g_{ip}=0\) for every family, no matter how large its error.
- Color and geometry answer different questions. A high color gradient with a weak geometry
  gradient asks first for appearance adjustment, not densification.

The formulas change when affine colors or opacity are present. Those are valid future variants,
but they are not the selected HIER four-array field and should not contaminate the first
diagnostic.

### 2.3 Parameter gradients are low-order moments of one pressure field

Let

\[
q_{ip}=G_{ip}h_{ip}
\]

inside active support. In the Gaussian's local frame, define

\[
m_{1,i}=\sum_p q_{ip}\xi_{ip},
\qquad
M_{2,i}=\sum_p q_{ip}\xi_{ip}\xi_{ip}^{\mathsf T}.
\]

Then the local mean gradient is a scaled first moment,

\[
R_i^{\mathsf T}g^\mu_i
=
\begin{bmatrix}s_{ix}^{-2}&0\\0&s_{iy}^{-2}\end{bmatrix}m_{1,i},
\]

the two log-scale gradients are the diagonal entries of a scaled \(M_{2,i}\), and the rotation
gradient is its off-diagonal entry multiplied by the anisotropy factor. This gives a useful
“multipole” reading:

- appearance/amplitude is a zeroth-order response;
- translation is a dipole or first moment;
- scale and rotation are diagonal and off-diagonal quadrupole moments;
- residual structure invisible to all of these lies outside the current local tangent family.

This is a diagnostic transfer from multipole expansions, not a claim that Gaussian fitting is a
physical field theory. Its value is that cancellation in one moment does not imply cancellation in
the others.

### 2.4 The normalized renderer has an additional geometry blindness

For comparison, the normalized renderer has

\[
Z_p=\sum_j w_{jp}+\varepsilon,\qquad
C_p=\frac{\sum_j w_{jp}c_j}{Z_p},\qquad
\rho_{ip}=\frac{w_{ip}}{Z_p}.
\]

Its local derivatives include

\[
\frac{\partial C_p}{\partial c_i}=\rho_{ip}I,
\qquad
\frac{\partial C_p}{\partial w_{ip}}=\frac{c_i-C_p}{Z_p}.
\]

Thus normalized geometry pressure is

\[
h^{\mathrm{norm}}_{ip}
=
\frac{u_p^{\mathsf T}(c_i-C_p)}{Z_p}.
\]

When \(c_i\) is close to the current composite, changing its weight or geometry barely changes the
render even if the pixel error is high. This compositor-neutrality failure does not occur in the
same form in additive HIER. It matters for any future transfer of the proposed packet back into
the normalized pipeline and is a reason not to reuse thresholds across renderers.

### 2.5 Error magnitude and image adjoint magnitude are not interchangeable

The selected HIER objective uses masked L1 pixel loss plus SSIM weight 0.3.

- Under L2, \(u_p\) is proportional to \(C_p-T_p\), so its magnitude tracks residual magnitude.
- Under L1, each nonzero channel contributes essentially its sign, independent of whether the
  channel error is small or large. Large error does not create proportionally large pressure.
- Charbonnier shrinks near zero and saturates toward an L1-like response for large errors.
- SSIM makes \(u_p\) depend on a neighborhood and can produce an adjoint direction different from
  the local raw RGB residual.
- Mask weighting can make a visually large raw error irrelevant to the optimized objective.
- Coverage, containment, rate, and geometry penalties may act through weights or parameters
  separately from \(u_p\).

Consequently, a diagnostic must log both residual/error and the image adjoint. It should also
separate objective channels:

\[
u_p
=
(1-\alpha)u^{\mathrm{pixel}}_p
+\alpha u^{\mathrm{SSIM}}_p,
\]

with coverage, containment, rate, and explicit geometry contributions recorded separately at the
parameter level. Otherwise “cancellation” may only mean two objectives disagree.

### 2.6 At an additive color optimum, zero gradient has a stronger interpretation

With geometry fixed, the pure-additive renderer is a linear model. Let \(A\in\mathbb{R}^{P\times N}\)
contain the sampled Gaussian columns \(A_{pi}=f_{ip}\), and let \(B\in\mathbb{R}^{N\times3}\)
contain RGB coefficients:

\[
C=AB.
\]

For L2 fitting, the color optimum obeys the normal equations

\[
A^{\mathsf T}(C-T)=0.
\]

Nonzero residual with zero color gradient is therefore not merely accidental cancellation. The
residual is orthogonal to the span of the current basis columns. Recoloring those same Gaussians
cannot reduce L2 error; the dictionary must change through geometry, birth, split, or a richer
appearance basis.

This interpretation must be scoped carefully. HIER uses L1 plus SSIM, colors can be bounded or
projected, and geometry remains nonlinear. Still, an L2 color-projected diagnostic is valuable
because it distinguishes “optimizer has not recolored yet” from “the current basis span cannot
express this residual.”

### 2.7 Symmetric splitting is second-order, and HIER has an analytic split matrix

Consider a function-preserving additive split of row \(i\): replace coefficient \(c_i\) at
\(\mu_i\) with two half-coefficient children at \(\mu_i+\eta\) and \(\mu_i-\eta\). The first
order terms cancel:

\[
\tfrac12 f(\mu_i+\eta)+\tfrac12 f(\mu_i-\eta)
=
f(\mu_i)+\tfrac12\eta^{\mathsf T}
\nabla_{\mu_i}^2 f\,\eta+O(\|\eta\|^4).
\]

Inside unchanged active support,

\[
\nabla_{\mu_i}^2G_{ip}
=
G_{ip}
\left[
(A_i\delta_{ip})(A_i\delta_{ip})^{\mathsf T}-A_i
\right].
\]

The additive HIER splitting matrix is therefore

\[
S_i
=
\sum_p h_{ip}G_{ip}
\left[
(A_i\delta_{ip})(A_i\delta_{ip})^{\mathsf T}-A_i
\right].
\]

The loss change for an infinitesimal symmetric split is

\[
\Delta L
=
\tfrac12\eta^{\mathsf T}S_i\eta+O(\|\eta\|^4).
\]

If \(\lambda_{\min}(S_i)<0\), a symmetric split along its minimum eigenvector is locally
descent-producing even when the ordinary mean gradient is zero. If \(S_i\) is positive
semidefinite, cancellation alone does not certify a mean split.

This is the direct additive specialization of the known Splitting Steepest Descent/SteepGS
construction. It is not a new theorem. HIER-specific cautions are:

- compact-support crossings and station-ball containment are nonsmooth finite effects;
- a practical split also changes scale and perhaps support;
- the pure four-array field should halve RGB coefficients, not silently introduce an opacity
  array; and
- every exact-N split needs donor funding.

The eigenpair should therefore rank a finite, certified split proposal; the existing exact render
and short recovery gate remains authoritative.

### 2.8 Birth is a gradient with respect to a missing dictionary column

An unsupported location cannot be found from existing row gradients. Introduce a temporary
candidate column \(\phi_b\) with a linear RGB coefficient \(\beta_b\):

\[
C'=C+\phi_b\beta_b.
\]

The gradient at zero coefficient is

\[
\frac{\partial L}{\partial\beta_b}\bigg|_{\beta_b=0}
=
\phi_b^{\mathsf T}u.
\]

This is a valid “ghost” or column-generation gradient. It requires candidate geometry to be
proposed independently; the nonexistent Gaussian has no geometry gradient at zero coefficient.

For L2 and fixed existing coefficients, the optimal candidate coefficient and exact SSE gain are

\[
\beta_b^\star
=
\frac{\phi_b^{\mathsf T}(T-C)}{\phi_b^{\mathsf T}\phi_b},
\]

\[
\operatorname{gain}(b)
=
\sum_{k\in\{R,G,B\}}
\frac{(\phi_b^{\mathsf T}r_k)^2}{\phi_b^{\mathsf T}\phi_b}.
\]

After the existing colors have first been re-solved under L2, a joint refit with the candidate
uses its component orthogonal to their span:

\[
\phi_b^\perp=(I-P_A)\phi_b,
\qquad
\operatorname{gain}_{\mathrm{refit}}(b)
=
\sum_k
\frac{((\phi_b^\perp)^{\mathsf T}r_k)^2}
     {\|\phi_b^\perp\|_2^2}.
\]

This is matching pursuit/column generation and is adjacent to StructSplat's existing orthogonal
detail pursuit. It should not be presented as a novel birth mechanism. Its value here is as an
operator oracle: it detects basis-span deficiency and scores a birth in the same output currency.
For masked L1 plus SSIM, use the first-order action score
\(-\langle u,\phi_b\beta_b\rangle\), a bounded local coefficient solve, and an exact trial.

### 2.9 Small stationary gradients cannot rank pruning or merging

At a fitted point, important and unimportant rows can both have nearly zero gradients. For an
additive L2 field, let \(v_i=\phi_i c_i\) be row \(i\)'s rendered contribution and
\(r=T-C\). Removing the row without refitting changes SSE by

\[
\Delta_{\mathrm{remove},i}
=
2\langle r,v_i\rangle+\|v_i\|_2^2.
\]

At a color least-squares optimum, the cross term vanishes but the positive
\(\|v_i\|^2\) term remains. Zero gradient therefore does not mean safe pruning.

With all surviving colors refit, the useful cost is the component of \(v_i\) that the remaining
span cannot reproduce:

\[
\Delta_{\mathrm{LOO},i}
\approx
\|(I-P_{A_{-i}})v_i\|_2^2.
\]

Merge two rows by fitting the best certified replacement column and measuring the residual after a
local or global coefficient re-solve. This connects to leverage, influence, Schur complements,
LocoADC's similarity merge, REFINE's Hessian importance, and HIER-032's exact local additive merge
ranking. Again, the HIER opportunity is the joint donor-plus-destination transaction, not a new
standalone merge score.

## 3. Error/gradient phase atlas

### 3.1 Compact statistics to retain

For each row \(i\) and parameter family \(z\), stream:

\[
G_i^z=\sum_p g_{ip}^z
\quad\text{(net pressure)},
\]

\[
A_i^z=\sum_p\|g_{ip}^z\|_2
\quad\text{(cancellation-resistant activity)},
\]

\[
\chi_i^z
=
\frac{\|G_i^z\|_2}{A_i^z+\epsilon}
\quad\text{(directional coherence)},
\]

\[
Q_i^z=\sum_p g_{ip}^z(g_{ip}^z)^{\mathsf T}
\quad\text{(directional second moment)}.
\]

For a dominant eigenvector \(v\) of \(Q_i^z\), define positive and negative projected masses

\[
a^+=\sum_p\max(v^{\mathsf T}g_{ip}^z,0),\qquad
a^-=\sum_p\max(-v^{\mathsf T}g_{ip}^z,0),
\]

\[
b_i^z
=
1-\frac{|a^+-a^-|}{a^++a^-+\epsilon}.
\]

Here \(b\) near one means balanced opposing pressure and \(b\) near zero means one-sided pressure.
For the 2D mean family, the eigenvalue ratio of \(Q^\mu\) distinguishes a one-axis conflict from an
isotropic or multi-directional conflict.

Also retain:

- raw residual energy and high-pass error in the row's active support;
- active-support pixel count, sole-owner/coverage count, and minimum local unit coverage;
- rendered contribution energy and a local Jacobian-column norm;
- neighbor column similarity and approximate leverage;
- split minimum eigenvalue/eigenvector for shortlisted rows;
- spatial sign-cluster count or a small signed contribution thumbnail for diagnostics;
- objective-channel versions of \(G\) and \(A\); and
- temporal stability across several steps or recovery checkpoints.

Coordinatewise absolute accumulation, as used by AbsGS, and sum-of-vector-norm activity should both
be measured in the first oracle. They are not identical. The former is cheap and kernel-friendly;
the latter gives the exact triangle-inequality coherence ratio.

Raw gradients across parameter families are not comparable: mean is measured per pixel, log-scale
is dimensionless, rotation is radians, and color is RGB coefficient. Normalize them through a
common bounded action or preconditioner, for example

\[
V_z
\propto
(G_i^z)^{\mathsf T}D_z^{-1}G_i^z,
\]

or, preferably, render the finite trust-region proposal and score its actual loss secant.

### 3.2 Main phase table

| Error | Absolute activity | Coherence / other structure | Diagnosis | First action to test |
|---|---|---|---|---|
| low | low | any | locally settled or objective-irrelevant | no-op; prune only after leave-one-out test |
| high | high | high mean coherence | coherent translation pressure | move; clone only if keeping the parent plus a child beats moving |
| high | high | low mean coherence, strong coherent log-scale | symmetric width/extent mismatch | shrink or grow, not split by default |
| high | high | low mean coherence, strong rotation moment | orientation mismatch | rotate or anisotropically reshape |
| high | high | low mean coherence, anisotropic balanced \(Q^\mu\), negative split eigenvalue | two-sided/multi-lobe demand at a split saddle | certified split along minimum eigenvector |
| high | high | low mean coherence, isotropic or multi-cluster pressure, no negative binary split | more than a two-child structure or high-frequency aliasing | multiple ghost births, regional retessellation, or richer appearance |
| high | low | zero active support / low coverage | no current parameter can see the pixel | birth; at fixed N, teleport from a cheap donor |
| high | low geometry, high color activity | current footprint is useful but appearance is wrong | recolor, exact coefficient solve, or richer color basis |
| high | low in every current family after color solve | residual outside current tangent/basis span | ghost-column birth or basis enrichment |
| high raw error | low image adjoint | L1 saturation, mask exclusion, SSIM interaction, or objective mismatch | inspect \(u\); do not densify from raw error alone |
| low | high | high | sensitive but already accurate region, possibly L1/SSIM or regularizer pressure | continue optimization or protect; topology is not justified |
| any | image and regularizer packets oppose | aggregate is small only after objective mixing | objective conflict, not pixel cancellation | resolve weighting or constraints; log channels separately |
| low/moderate | low individual row gradients | two rows have nearly collinear render/Jacobian columns and low leave-one-out cost | redundant or gauge-like pair | merge/refit |
| low/moderate | low gradient | high leverage, unique coverage, or sole-owner pixels | important stationary row | protect from prune/merge |

### 3.3 Seven canonical visual patterns

These fixtures are more informative than a single “high error / low gradient” example:

1. **Shifted blob.** Target mass lies mainly to the right of one Gaussian. Mean contributions
   align, \(G^\mu\) and \(\chi^\mu\) are high. Move is the correct first-order action.
2. **Wrong width.** Target is a narrower or wider symmetric blob. Left and right mean
   contributions oppose and cancel, but log-scale contributions align. Resize is correct; an
   AbsGS-only rule can incorrectly split.
3. **Wrong orientation.** An elongated target crosses the Gaussian at a different angle. Mean may
   cancel while the off-diagonal second moment and rotation gradient are coherent. Rotate.
4. **Two separated lobes.** One broad Gaussian covers two target peaks. Mean cancels, opposing
   activity is anisotropic and balanced, and the split matrix can have a negative eigenvalue.
   Split is now justified.
5. **Checkerboard or hair bundle.** Many alternating demands occur inside one footprint. Low
   coherence is isotropic or multi-cluster; a binary split may not help. Use several births,
   frequency-aware retessellation, or a richer appearance model.
6. **Support hole.** A high-error pixel has no active Gaussian. Every existing contribution is
   exactly zero. Only a proposed new column can see it; birth or teleport is required.
7. **Duplicate rows.** Two nearly identical Gaussians reconstruct well and sit at a stationary
   point. Both gradients are small, yet one can be removed or merged after refit at low cost.
   Gradient magnitude alone cannot discover the donor.

### 3.4 Multiple overlapping Gaussians require group diagnostics

The full reduced gradient is

\[
\nabla_\theta L=J^{\mathsf T}u,
\]

where \(J\) contains every row's renderer Jacobian. If two row blocks have nearly collinear
columns, individual parameter gradients can be unstable or gauge-dependent even while their
combined rendered function is stable. Under additive L2, the Gauss–Newton cross block is
\(J_i^{\mathsf T}J_j\); large normalized cross blocks identify strong coupling.

Consequences:

- aggregate statistics by exact duplicate group, lineage, or high-column-similarity neighborhood
  before interpreting row scores;
- treat equal-and-opposite row updates as possible reparameterization, not automatically useful
  scene structure;
- use leverage or leave-one-out refit to find redundant donors;
- score merge and split on the entire group transaction; and
- keep sole-owner coverage as a hard protection even when a row's gradient is small.

FIT-019 already showed an analogous representation-gauge problem for normalized opacity splitting.
The additive field has a simpler but still real coefficient gauge when columns are identical or
nearly dependent.

## 4. Operator-specific decision rules

### 4.1 Move

Use a bounded, optimizer-preconditioned mean proposal when:

- \(G_i^\mu\) is large in action-normalized units;
- coherence is high enough that one direction represents most covered pixels; and
- the move preserves containment and coverage.

The exact finite move secant should beat no-op. High absolute activity alone is insufficient.

### 4.2 Resize and rotate

Test log-scale and rotation actions before split whenever their family-specific predicted decrease
is larger than the mean action. Symmetric cancellation in mean is expected for a width error. The
moment packet supplies the missing distinction:

- diagonal \(M_2\) pressure routes to anisotropic scale;
- off-diagonal \(M_2\) pressure routes to rotation;
- frequency violation can supplement this when pixel structure exceeds the footprint bandwidth.

### 4.3 Split

A strong split proposal requires all of:

1. meaningful error and absolute activity;
2. a spatially multi-sided contribution pattern rather than mere loss-channel conflict;
3. no cheaper continuous family action with comparable predicted gain;
4. negative split curvature or a positive exact finite split trial;
5. a legal containment-preserving child geometry; and
6. a donor whose removal/merge cost is smaller than the split gain at exact count.

Low coherence is a screening feature, not the decision. This is stricter than AbsGS and GDAGS and
is deliberately compatible with SteepGS.

### 4.4 Clone

An exact co-located, coefficient-halved additive clone is functionally identical and gives both
children identical gradients. It adds no usable degree of freedom until symmetry is broken.

Test an asymmetric clone only when:

- pressure is coherent toward an adjacent underrepresented area;
- moving the parent would damage pixels it uniquely covers;
- retaining the parent and inserting a nearby child beats an independent ghost birth; and
- a donor funds the extra row.

In many 2D HIER cases, “clone” is better understood as a parent-anchored birth proposal. Scale alone
should not choose it.

### 4.5 Birth

Birth sites must come from a candidate bank because nonexistent geometry has no gradient. Candidate
sources can include:

- connected high-error or high-adjoint regions;
- coverage-debt components and certified station-ball sites;
- high-pass or Laplacian residual extrema with NMS;
- structure-tensor or error-gradient tangent proposals;
- residual matching-pursuit kernels over several scales and orientations; and
- finite ghost gates around rows whose pressure is multi-cluster but not binary-splittable.

Rank by orthogonalized L2 gain in the diagnostic and by actual mixed-objective action score plus
exact render in the real protocol.

### 4.6 Prune

Never use “small current gradient” as the main prune signal. Use:

- exact or approximate leave-one-out reconstruction cost after coefficient refit;
- leverage or Hessian/Fisher removal sensitivity;
- rendered contribution energy;
- sole-owner coverage and boundary protection;
- recovery-aware loss after a short block; and
- rate saving if a complete codec is in scope.

### 4.7 Merge

Propose pairs or small groups with:

- overlapping support;
- similar render and Jacobian columns;
- compatible color after local consistency/refit;
- low joint replacement error; and
- no unique coverage loss.

Fit the merged row's coefficient by local least squares, then use the existing global projection
and exact containment/render gate. HIER-032's contribution-aware donor merge is already the closest
local mechanism.

### 4.8 Teleport

At exact N, teleport is a complete transaction:

\[
\text{remove donor }d + \text{insert candidate }b.
\]

The approximate value is birth gain minus donor removal cost, but interactions mean the final
score should come from a joint local/global refit and exact render. Coverage-debt destinations may
be mandatory, yet HIER-032 shows that geometrically successful closure can still lose too much
interior quality through its donors.

### 4.9 No-op and continuous optimization

No-op is a real operator and must participate in every oracle. If an ordinary bounded parameter
step gives the largest recovered gain, topology should wait. Without no-op and continuous-family
controls, an experiment can only compare different ways of making unnecessary edits.

## 5. Fixation anti-library

The following are controls, occupied prior art, or known failure modes:

- take the absolute value of the already-summed gradient and call it AbsGS;
- use high error plus low aggregate gradient as an unconditional split rule;
- use coherence alone to claim a new split-versus-clone controller;
- compare raw mean, scale, rotation, and color gradient magnitudes without unit normalization;
- birth only at the largest raw residual pixel;
- infer a safe prune from stationary gradient magnitude;
- merge nearest means without fitting appearance and measuring exact local error;
- ignore pixels outside current support because their gradients are zero;
- let the selected operation be predetermined by the selector name;
- use immediate gain as the sole label despite FIT-017/018/019 sign reversals after recovery;
- mix image and regularization contributions before diagnosing cancellation; or
- claim a general method from one exposed HIER-031/032 field.

## 6. Productive recombinations

### Candidate method P1 — True pre-reduction HIER contribution packet

**Central claim:** streamed pre-reduction \(G,A,Q\) statistics for every HIER parameter family
contain operator-relevant information that the current post-reduction mean-gradient proxy loses.

**Novelty class:** N1.

**Known foundation:** AbsGS pixelwise absolute gradients, GDAGS coherence, standard renderer
backpropagation, and StructSplat's existing fit statistics.

**Irreducible delta:** exact specialization to additive compact-support RS Gaussians, including
color, log-scale, rotation, objective-channel, coverage, and split-moment telemetry.

**Why this is not merely A + B:** the contribution packet itself is an engineering synthesis, not
a new scientific principle. Its value must come from demonstrated operator identification rather
than the packet definition.

**Changed grammar or transfer mechanism:** replace one scalar selector score with a renderer-level
sufficient-statistic interface shared by several action proposers.

**New prediction:** width and orientation fixtures with low mean coherence will be routed to
scale/rotation rather than over-split, reducing action regret relative to AbsGS/GDAGS-like scalar
rules.

**Cheapest killing test:** analytic one-Gaussian fixtures plus a reference-renderer recomputation
whose signed sums match autograd. **Null hypothesis:** family packets do not reduce exact-action
regret beyond residual, scale, and the current aggregate mean gradient.

**Prior-art threats:** GDAGS, AbsGS, COB-GS, any unpublished per-attribute gradient-statistic
controller, and standard mixture sufficient statistics.

**Novelty confidence:** 5–15% as a method; literature and local-code cutoff 2026-08-12.

**Scientific value:** a correct measurement substrate, even if no controller wins.

**Publishable if successful:** only as part of a broader operator-identification result.

**Publishable if partially successful:** a precise negative map of which families are informative.

**Publishable if it fails informatively:** demonstrates that compact moments are insufficient and
full spatial/action probes are required.

### Candidate method P2 — Tangent-family operator router

**Central claim:** action-normalized first- and second-moment responses can choose continuous
move, resize, rotate, or recolor before topology is considered.

**Novelty class:** N2.

**Known foundation:** Gaussian derivatives, natural/trust-region gradient scaling, structure-aware
anisotropic splitting, and HIER's RS parameterization.

**Irreducible delta:** interpret parameter-family gradients as competing finite rendered actions,
with no topology edit unless every legal continuous action is insufficient.

**Why this is not merely A + B:** the method changes the decision space from “densify or not” to
“which local tangent family explains the residual,” but its mathematical components are standard.

**Changed grammar or transfer mechanism:** continuous parameter families become explicit operators
in the same auction as topology edits.

**New prediction:** many rows selected by cancellation-resistant mean activity will prefer scale
or rotation and recover equally or better with no count growth.

**Cheapest killing test:** exact finite action enumeration on the seven canonical fixtures and
shortlisted HIER rows. **Null hypothesis:** continuous-family controls do not change the best
operator or recovered loss.

**Prior-art threats:** structure-aware densification, second-order optimization, and unpublished
Gaussian attribute schedulers.

**Novelty confidence:** 10–25%, cutoff 2026-08-12.

**Scientific value:** separates optimization failure from representation-capacity failure.

**Publishable if successful:** a general action-routing result with substantial count or quality
benefit.

**Publishable if partially successful:** a diagnostic showing when densification is premature.

**Publishable if it fails informatively:** establishes that topology dominates local tangent
repair in this regime.

### Candidate method P3 — Additive split-matrix-certified funded split

**Central claim:** negative additive HIER split curvature plus exact donor funding predicts useful
splits more reliably than gradient magnitude, coherence, or footprint scale.

**Novelty class:** N1/N2.

**Known foundation:** Splitting Steepest Descent and SteepGS; HIER station-ball containment and
exact-count transactions.

**Irreducible delta:** the task-specific relationship between four-array coefficient-halved
compact-support splitting, containment certification, and exact donor cost.

**Why this is not merely A + B:** it largely is a known split certificate combined with HIER
constraints; no standalone novelty should be claimed.

**Changed grammar or transfer mechanism:** split becomes a second-order certified transaction
rather than a consequence of a high row score.

**New prediction:** positive-semidefinite low-coherence rows will often prefer reshape or birth,
while negative-eigenvalue rows will concentrate the actual split wins.

**Cheapest killing test:** compute the 2-by-2 matrix for only top-activity rows and compare its
eigenvalue ranking with exact coefficient-halved split trials. **Null hypothesis:** the minimum
eigenvalue adds no rank correlation or regret reduction after coherence and scale.

**Prior-art threats:** SteepGS is direct and severe.

**Novelty confidence:** 0–10% for split scoring, 10–20% for the exact-count HIER relationship;
cutoff 2026-08-12.

**Scientific value:** a principled rejection filter may be useful even with no novelty.

**Publishable if successful:** only through a broader HIER allocation system or renderer-specific
theory.

**Publishable if partially successful:** a renderer-specific validation or boundary failure map.

**Publishable if it fails informatively:** compact support/finite edits invalidate the
infinitesimal certificate in the relevant regime.

### Candidate method P4 — Orthogonal ghost birth with donor-aware teleport

**Central claim:** an orthogonalized missing-column score finds useful HIER births that every
existing row gradient must miss, and joint donor pricing prevents their quality damage.

**Novelty class:** N2.

**Known foundation:** matching pursuit, column generation, Revising Densification, FIT-040
orthogonal detail pursuit, HIER-032 coverage candidates, and influence/leave-one-out removal.

**Irreducible delta:** compare the complete exact-N donor-plus-ghost transaction after local
coefficient projection, rather than selecting destination and donor independently.

**Why this is not merely A + B:** the components are known; the hypothesis is that their joint
transaction value solves the failure exposed by HIER-032.

**Changed grammar or transfer mechanism:** “birth” is represented as a candidate dictionary column
and “teleport” as a count-balanced exchange in rendered function space.

**New prediction:** orthogonal gain will reject residual peaks already expressible by current
columns, while joint donor scoring will preserve more interior quality than coverage-only funding.

**Cheapest killing test:** reuse the frozen HIER-031 field, generate a small certified bank, and
compare predicted transaction gain with exact no-recovery and 20-step outcomes. **Null hypothesis:**
orthogonalization or joint pricing does not improve top-k transaction regret.

**Prior-art threats:** matching pursuit, LocoADC, HIER-032 itself, and sparse approximation methods.

**Novelty confidence:** 10–25% for the relationship, cutoff 2026-08-12.

**Scientific value:** directly attacks unsupported pixels and exact-count funding.

**Publishable if successful:** as a general fixed-budget Gaussian column-exchange mechanism.

**Publishable if partially successful:** a clear donor/destination interaction model.

**Publishable if it fails informatively:** shows that local linear scores cannot predict recovered
fixed-budget exchanges.

### Candidate method P5 — Leverage-protected merge and prune

**Central claim:** local column similarity plus leave-one-out/refit cost identifies safe HIER
donors more reliably than contribution, opacity, or gradient magnitude.

**Novelty class:** N1.

**Known foundation:** regression leverage, influence, REFINE, LocoADC merging, and HIER-032 local
merge error.

**Irreducible delta:** enforce sole-owner coverage and exact containment while selecting donors for
another operation.

**Why this is not merely A + B:** it is mostly a careful integration of known removal diagnostics.

**Changed grammar or transfer mechanism:** pruning and merging become funding operators whose
value is measured after refit.

**New prediction:** low-leverage redundant pairs will fund births with smaller interior loss than
mutual-nearest contribution-only pairs.

**Cheapest killing test:** exact local leave-one-out and pair-merge trials on a few hundred
shortlisted donors. **Null hypothesis:** leverage/refit features do not improve donor ranking over
the current exact local additive merge error.

**Prior-art threats:** REFINE, PUP/Meson-style importance, LocoADC, classical regression deletion
diagnostics.

**Novelty confidence:** 5–15%, cutoff 2026-08-12.

**Scientific value:** improves falsifiability of any fixed-count densification claim.

**Publishable if successful:** only in combination with a better allocation controller.

**Publishable if partially successful:** donor-safety benchmark and failure taxonomy.

**Publishable if it fails informatively:** confirms that only exact global refit can price donors.

## 7. Exploratory candidates

### Exploratory candidate E1 — Error–adjoint–gradient phase atlas

**Central claim:** a synthetic atlas separating raw residual, spatial error gradient, image
adjoint, pixel contributions, and action outcomes reveals repeatable operator signatures.

**Novelty class:** N2.

**Known foundation:** system identification, analytic fixtures, and gradient-debug suites.

**Irreducible delta:** a Gaussian-topology operator oracle rather than another end-quality sweep.

**New prediction:** fixtures with identical residual energy but different spatial moment structure
will prefer different operations.

**Cheapest killing test:** seven canonical fixtures under L2 and masked L1-plus-SSIM.

**Null hypothesis:** operator labels are unstable under harmless scale, color, or loss changes.

**Prior-art threats:** diagnostic figures in AbsGS, GDAGS, SteepGS, and related densification work.

**Novelty confidence:** 10–25%, cutoff 2026-08-12.

### Exploratory candidate E2 — Functional-equivalence group packets

**Central claim:** aggregating contribution statistics over duplicate/near-collinear row groups
produces more stable operator decisions than rowwise signals.

**Novelty class:** N2.

**Known foundation:** quotient parameterizations, FIT-019 gauge evidence, mixture-component
identifiability, and Jacobian clustering.

**Irreducible delta:** define groups by rendered/Jacobian equivalence under the active HIER
objective, not parameter proximity alone.

**New prediction:** group packets remain stable under exact coefficient splits that alter raw row
statistics.

**Cheapest killing test:** create equivalent duplicate parameterizations and test action-ranking
invariance.

**Null hypothesis:** grouping does not improve invariance or donor ranking.

**Prior-art threats:** gauge-invariant allocation, mixture grouping, and canonicalization methods.

**Novelty confidence:** 15–30%, cutoff 2026-08-12.

### Exploratory candidate E3 — Horizon-stable action spectroscopy

**Central claim:** a small sequence of exact immediate and recovery action responses can identify
which gradient statistics remain predictive after optimization.

**Novelty class:** N1/N2.

**Known foundation:** perturb-and-recover analysis and the FIT-017–020 evidence history.

**Irreducible delta:** use discrete operator interventions as labels, not an early scalar bend.

**New prediction:** split curvature and orthogonal birth gain retain sign more often than simple
immediate PSNR gain, but no claim is made before measurement.

**Cheapest killing test:** recovery horizons 0, 5, and 20 for a small balanced action set.

**Null hypothesis:** action ranking changes too often across horizons to support a router.

**Prior-art threats:** learning-to-densify and response-prediction systems.

**Novelty confidence:** 10–20%, cutoff 2026-08-12.

### Exploratory candidate E4 — Objective-channel conflict detector

**Central claim:** keeping L1, SSIM, coverage, containment, and rate contribution packets separate
prevents objective cancellation from being misclassified as representation cancellation.

**Novelty class:** N2.

**Known foundation:** multi-objective optimization and gradient-conflict diagnostics.

**Irreducible delta:** route topology only from conflicts that persist in the primary
reconstruction channel and pass protected gates.

**New prediction:** some low aggregate gradients near boundaries are caused by protected-loss
opposition rather than a need to split.

**Cheapest killing test:** recompute packets with one VJP per objective channel on frozen boundary
states.

**Null hypothesis:** channel separation does not change any shortlisted action.

**Prior-art threats:** PCGrad-like diagnostics and multi-task gradient surgery.

**Novelty confidence:** 10–25%, cutoff 2026-08-12.

## 8. Transformational candidates

### Transformational candidate T1 — Gradient multipole spectrum

**Central claim:** treat each Gaussian as a local sensor of residual-pressure moments, and grow the
representation only when pressure energy lies beyond the moments exposed by its current parameter
families.

**Novelty class:** N3-T.

**Known foundation:** multipole expansions, moment methods, Gaussian derivatives, and adaptive
basis refinement.

**Irreducible delta:** topology becomes an order-selection problem: optimize captured zeroth/first/
second moments; add components only for unresolved higher-order structure.

**Why this is not merely A + B:** it changes the formulation from thresholded densification to
local model-order diagnosis, although its mathematics is assembled from known moments.

**Changed grammar or transfer mechanism:** primitives expose a moment spectrum and topology edits
raise local representational order.

**New prediction:** high-order residual energy predicts when binary splitting fails and multiple
births or affine appearance are superior.

**Cheapest killing test:** compare moment spectra with exact action oracle labels on the canonical
atlas.

**Prior-art threats:** scale-space theory, Gaussian derivative filters, adaptive mixture order,
wavelets, and unpublished moment-based densification.

**Novelty confidence:** 20–40%, cutoff 2026-08-12; terminology overlap is a major uncertainty.

### Transformational candidate T2 — Tangent-deficiency quotient

**Central claim:** density control should operate on the residual quotient left after projecting
the desired image change onto the current field's legal tangent space.

**Novelty class:** N3-T.

**Known foundation:** variable projection, tangent-space methods, orthogonal matching pursuit, and
manifold model reduction.

**Irreducible delta:** define topology debt as

\[
u_\perp=(I-P_{\mathcal{T}_\theta})u,
\]

where \(\mathcal{T}_\theta\) spans legal bounded color, move, scale, and rotation render
perturbations. Birth and split candidates compete only against \(u_\perp\).

**Why this is not merely A + B:** it changes the allocator's target from pixel error to the part of
the desired functional descent that current continuous parameters cannot realize.

**Changed grammar or transfer mechanism:** topology edits expand a renderer tangent space rather
than merely add high-error primitives.

**New prediction:** many high-error regions vanish from the densification queue after tangent
projection, while true support holes remain.

**Cheapest killing test:** local Jacobian projection on small tiles, compared with exact action
regret.

**Prior-art threats:** Gauss–Newton, variable projection, neural tangent features, column
generation, and adaptive finite elements.

**Novelty confidence:** 20–35%, cutoff 2026-08-12.

### Transformational candidate T3 — Topology transaction compiler

**Central claim:** compile every legal discrete edit into a rendered secant plus invariants, so
heterogeneous operations can be optimized in one action language.

**Novelty class:** N3-T.

**Known foundation:** compiler rewrite systems, model editing, action-value control, trust-region
optimization, and StructSplat's transactional safe schedule.

**Irreducible delta:** an operation emits

\[
\mathcal{A}
=
\{\Delta C,\Delta\text{count},\Delta\text{bytes},
\text{containment receipt},\text{coverage receipt},
\text{state-migration plan}\},
\]

and the selector compares complete transactions rather than operation-specific thresholds.

**Why this is not merely A + B:** it changes the permissible interface between proposal and
selection; the selector no longer knows whether a secant came from split, birth, merge, or
teleport.

**Changed grammar or transfer mechanism:** topology is an intermediate representation of legal
function-space rewrites.

**New prediction:** common scoring exposes when a sophisticated edit is dominated by no-op or a
simple move and makes exact-count donor interactions explicit.

**Cheapest killing test:** implement the representation only in an offline oracle for a few
actions; no production compiler is needed.

**Prior-art threats:** generic model-edit auctions, transactional architecture search, and
rewriting-based optimizers.

**Novelty confidence:** 20–40%, cutoff 2026-08-12.

## 9. Cross-domain transfers

### Transfer 1 — Adaptive finite elements: residual versus jump estimators

Finite-element methods distinguish unresolved cell residuals, discontinuity/jump errors, mesh
refinement, and higher-order basis enrichment. The useful transfer is to separate representable
tangent error from support/boundary error and to choose split-like refinement versus richer local
basis.

**Adoption barrier:** Gaussian supports overlap and move continuously rather than partitioning a
mesh.

**Broken correspondences:** there is no conserved cell ownership, no exact element boundary, and
SSIM is nonlocal. FEM convergence guarantees do not transfer.

### Transfer 2 — Matching pursuit and column generation

A missing Gaussian is a missing dictionary column. Reduced cost or residual correlation proposes
birth; orthogonalization prevents adding a column already spanned by existing rows.

**Adoption barrier:** the candidate dictionary is continuous in position, scale, and rotation, and
the real objective is not pure unconstrained L2.

**Broken correspondences:** Gaussian geometry is optimized after insertion, coefficients may be
bounded, compact support changes nonsmoothly, and exact-count donor removal couples the exchange.

### Transfer 3 — Multipole and moment expansions

Mean, scale, and rotation gradients are first and second spatial moments of signed pressure. The
transfer supplies a language for cancellation order and unresolved higher-order content.

**Adoption barrier:** RGB loss adjoints are signed, data-dependent, and not physical source
densities.

**Broken correspondences:** no conservation law, far-field expansion, or harmonic truncation
guarantee exists. “Monopole/dipole/quadrupole” is an interpretive diagnostic only.

### Transfer 4 — Mixture-model split/merge EM

Mixture models use component sufficient statistics, responsibility, split tests, and merge
likelihood. The useful transfer is group-level state and exact local refit.

**Adoption barrier:** additive Gaussian image bases are not normalized probability components.

**Broken correspondences:** colors can be signed coefficients, responsibilities need not sum as
probabilities, and reconstruction loss replaces log likelihood.

### Transfer 5 — Regression leverage and experimental design

Leverage measures whether a basis column is unique; leave-one-out and Schur complements price its
removal. This naturally protects sole-capacity rows and locates merge/prune donors.

**Adoption barrier:** the full nonlinear, mixed-loss Jacobian is too large for exact leverage.

**Broken correspondences:** geometry changes columns, protected coverage is a hard constraint, and
rate/containment are not captured by ordinary least squares.

### Transfer 6 — Compiler intermediate representations

Compilers lower many source constructs into a common intermediate language before optimization.
The transfer is an action IR containing rendered secants, count, bytes, constraints, and state
migration.

**Adoption barrier:** compiling an edit still requires rendering and possibly recovery, so the IR
does not remove scientific cost.

**Broken correspondences:** topology rewrites are not generally semantics-preserving, ordering can
change outcomes, and there is no confluence guarantee.

## 10. New-evidence discovery programs

### Program D1 — Analytic HIER operator atlas

Build deterministic low-resolution fixtures for shifted blob, wrong width, wrong orientation, two
lobes, checkerboard/hair, support hole, and duplicate rows. Run both clean L2 and the frozen masked
L1-plus-SSIM objective.

For every state:

1. retain raw residual, spatial error gradient, image adjoint, and exact pixel–Gaussian
   contributions;
2. verify signed contribution sums against autograd for color, mean, scale, and rotation;
3. compute \(G,A,\chi,Q,b\), family moments, split eigenpair, candidate birth gain, and
   removal/merge cost;
4. enumerate bounded recolor, move, scale, rotate, split, clone, birth, merge, prune, teleport,
   and no-op actions;
5. exact-render each action immediately and after fixed 5- and 20-step recovery; and
6. report top-1 accuracy, top-k recall, regret, sign calibration, and runtime/memory overhead.

This program survives the audit because it tests operator identifiability rather than claiming
end quality. A negative result would still establish which statistics are insufficient.

### Program D2 — Frozen HIER-031/032 contribution audit

Use the immutable HIER-031 selected field and HIER-032 protocol state without changing the result.
Stratify rows/sites into hair, boundary, interior, coverage debt, high error/high coherence, high
error/low coherence, and candidate donors.

Compare:

- current post-sum mean-gradient magnitude;
- true pre-reduction absolute activity;
- GDAGS-style coherence;
- full family/moment packet;
- additive split minimum eigenvalue;
- orthogonal ghost birth gain;
- exact local donor/merge cost; and
- complete transaction outcomes.

The audit must preserve exact N=7,000, outside-zero/containment receipts, and the existing quality
floor. It is a diagnostic continuation, not a reopening or rescue of HIER-032.

### Program D3 — Cross-image operator-identification benchmark

Only after the mechanism survives D1/D2, freeze a small multi-image development/held-out suite.
Sample comparable operator opportunities rather than only high-error rows. Balance actions and
include hard negatives.

Primary endpoints:

- recovered regret relative to the exact finite-action oracle;
- stability across loss, scale, image, and seed;
- false split rate for width/orientation cases;
- false prune rate for high-leverage/sole-owner rows;
- unsupported-site birth recall;
- exact-count transaction quality; and
- packet overhead versus saved recovery work.

Do not train a learned selector until the labels themselves are stable; FIT-020 already refuted one
early-response predictor in its frozen scope.

## 11. Pareto frontier

| Candidate | Apparent novelty | Falsifiability | Importance | Feasibility | First-test cost | Informative failure | Publication potential |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 contribution packet | low | very high | high | high | low | high | low alone |
| P2 tangent-family router | medium-low | high | high | high | low-medium | high | medium |
| P3 split-matrix funded split | low | very high | medium-high | high | low | high | low-medium |
| P4 ghost birth + donor teleport | medium-low | high | very high | medium | medium | very high | medium-high |
| P5 leverage-protected donors | low | high | high | medium | medium | high | medium in system |
| T1 gradient multipole spectrum | medium | high | high | medium-high | low | high | medium-high |
| T2 tangent-deficiency quotient | medium | high | very high | medium-low | medium | very high | high |
| T3 topology transaction compiler | medium | high | very high | medium | medium | high | high as a system |

## 12. Recommended first experiment

Run **HIER-033: pixel-gradient operator oracle**, a diagnostic with no production integration.

### Frozen question

Does a true pre-reduction, parameter-family contribution packet predict the best local operation
better than raw residual, the current post-sum mean-gradient proxy, AbsGS-style absolute activity,
and GDAGS-style coherence?

### Arms/signals

1. residual plus footprint/scale;
2. current post-sum mean-gradient magnitude;
3. true pre-reduction coordinatewise absolute mean activity;
4. net-to-absolute mean coherence;
5. full color/mean/scale/rotation \(G,A,Q\) packet;
6. packet plus additive split eigenpair;
7. packet plus orthogonal ghost birth and leave-one-out donor cost.

### Oracle actions

- no-op and ordinary optimizer continuation;
- bounded recolor or exact local coefficient solve;
- bounded move;
- anisotropic resize;
- rotate;
- function-preserving coefficient-halved split;
- asymmetric parent-anchored clone;
- independent ghost birth;
- merge/refit;
- prune/refit; and
- exact-count teleport or funded split.

### Data order

1. analytic fixtures;
2. a frozen sample of HIER-031/032 rows and sites;
3. only then a multi-image development/held-out assay.

### Decisive observation

The full packet must reduce recovered top-k action regret and specifically reduce false splits on
wrong-width/wrong-orientation fixtures without losing two-lobe split recall or support-hole birth
recall. Exact thresholds should be preregistered after measuring oracle noise, not guessed here.

### Null hypothesis

After residual, scale, support, and current aggregate gradient are known, pre-reduction direction,
family moments, split curvature, and ghost/removal scores do not improve recovered action ranking.

### Abandonment criteria

Stop the controller lineage if any of the following holds:

- analytic contribution sums do not match autograd;
- best-action labels are unstable across small harmless perturbations;
- immediate and 20-step labels reverse too often for useful prediction;
- the full packet cannot beat residual/current-gradient controls on held-out regret;
- runtime or memory overhead erases the saved optimization work; or
- the apparent gain comes only from changing count, containment, renderer, or objective.

### Implementation route

Start with a slow exact reference diagnostic:

1. obtain \(u=\partial L/\partial C\) with one VJP;
2. recompute active pixel–Gaussian intersections chunkwise using the reference equations;
3. accumulate signed sums, absolute activity, and second moments without storing the full
   pixel-by-row tensor;
4. verify \(\sum_p g_{ip}\) against ordinary autograd for every parameter family; and
5. enumerate finite actions offline.

Only if the oracle is positive should the exact CUDA backward stream \(G,A,Q\) before its reduction.
At N=7,000 the per-row packet is small; the expensive object is the pixel–Gaussian intersection
stream, which the renderer already visits. Determinism and block-reduction overhead must be
measured rather than assumed.

## 13. Audit limitations

- This document derives the primary formulas for the selected constant-color additive HIER field.
  Affine color, learned opacity, filter-variance derivatives, alpha compositing, and downstream 3D
  projection need separate derivations.
- The compact C0 fade is not differentiable at its cutoff, and tile membership is detached. The
  analytic split matrix is local to unchanged active support; finite certified trials remain
  authoritative near boundaries.
- The literature search used accessible primary papers and official proceedings/code surfaces. It
  cannot exclude unpublished, differently named, or inaccessible work. GDAGS, SteepGS, LocoADC,
  and REFINE substantially narrow any novelty claim.
- “Gradient multipole spectrum,” “tangent-deficiency quotient,” and “topology transaction
  compiler” are working formulations, not established new contributions.
- The L2 projection and gain formulas are exact only for fixed geometry, unconstrained additive
  coefficients, and squared error. HIER's actual L1-plus-SSIM decision requires exact trials.
- Current HIER evidence contains exposed, single-subject and task-specific diagnostics. It cannot
  establish a general image-representation method.
- FIT-017–020 show that immediate action benefit and early dynamics can reverse after recovery.
  Any router needs recovered labels and a no-op control.
- HIER-032 shows that destination repair can be real while donor damage breaks the total quality
  gate. No destination-only score is acceptable at exact count.

## Bottom line

The user's cancellation intuition is correct, but the actionable object is richer than the
ordinary gradient magnitude:

\[
\boxed{
\text{residual}
\;+\;
\text{image adjoint}
\;+\;
\{G,A,Q\}_{\text{color,mean,scale,rotation}}
\;+\;
\text{split curvature}
\;+\;
\text{ghost/removal action values}
}
\]

That packet can tell several stories:

- coherent first moment: move;
- coherent diagonal second moment: resize;
- coherent off-diagonal second moment: rotate;
- balanced bipolar pressure plus negative split curvature: split;
- unsupported or tangent-orthogonal residual: birth;
- low leave-one-out cost and similar columns: merge or prune;
- valuable destination plus cheap donor: teleport; and
- no winning finite action: no-op and continue optimization.

The scientifically defensible next step is not to implement another heuristic threshold. It is to
measure whether these signatures identify the exact winning operation.
