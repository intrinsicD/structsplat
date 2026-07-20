# Renderer-aware color quantization frontier

## Status and decision

**Search cutoff:** 2026-07-16
**Status:** literature/mechanics contingency; no development result and no authorization to tune
COMP-011.

COMP-011 remains the selected frozen experiment. The literature update below does not change its
menu, data, thresholds, or implementation. It narrows the experiment that may follow by making the
failure mechanism decisive:

- if quality-qualified SSP2V streams exist but lose on complete bytes, test an **implicit** compact
  color lattice/transform on a new development split rather than another explicit codebook;
- if SSP2V has useful rate but fails quality, test the already-proved normalized-renderer Gram
  distortion against Euclidean and diagonal controls on a new development split;
- if neither rate nor quality is close, close post-hoc color VQ instead of retuning it;
- if COMP-011 passes, run only its separately bound confirmation.

No COMP-011 development stream or target was opened to reach this decision.

## New primary-source findings

### Scene-adaptive lattice VQ attacks explicit-codebook overhead

[Xu, Wu, and Zhang's scene-adaptive lattice vector quantizer
(SALVQ)](https://arxiv.org/html/2509.13482) replaces uniform scalar quantization in per-scene
anchor-based 3DGS compressors. It parameterizes a learned invertible lattice basis with an SVD-like
orthogonal/diagonal/orthogonal factorization, uses Babai rounding for practical assignment, and
entropy-codes the integer lattice coordinates. The codebook is implicit: only the compact basis is
stored. In the paper's native high-dimensional anchor-latent regime, reported same-distortion
memory savings range from 4.55% to 16.16%, with small encode/decode overhead.

This is the strongest new threat to an “explicit RGB codebook is necessary” story. It is not direct
evidence for StructSplat: SALVQ is jointly trained with a Scaffold-GS representation and entropy
model, primarily quantizes a 50-dimensional latent, and does not test post-hoc three-dimensional
RGB symbols under StructSplat's normalized renderer or complete SSPL framing.

### Generic clustered VQ and renderer-aware quantization are already occupied

- [CGVQ](https://arxiv.org/abs/2607.05667) explicitly clusters Gaussian attributes. It removes any
  defensible generic novelty claim for applying clustered VQ to Gaussian parameters.
- [RDO-Gaussian](https://arxiv.org/html/2406.01597) combines entropy-constrained VQ with an
  end-to-end rendering rate-distortion objective. “Add rendering loss to VQ” is therefore prior
  art, not a new StructSplat mechanism.
- [Compressed 3D Gaussian Splatting](https://openaccess.thecvf.com/content/CVPR2024/html/Niedermayr_Compressed_3D_Gaussian_Splatting_for_Accelerated_Novel_View_Synthesis_CVPR_2024_paper.html)
  uses sensitivity-aware vector clustering and quantization-aware training. First-order
  renderer-sensitivity weighting is also occupied.
- [AQLM](https://arxiv.org/html/2401.06118) and the broader GPTQ/QuIP family establish that
  quadratic output-error objectives can materially change additive quantization. Transferring a
  quadratic Gram metric is known optimization structure; a StructSplat result would be a recipient-
  specific evidence/mechanics contribution.

### Cross-color correlation is an established lossless coding lever

[Wang, Ding, and Ma](https://arxiv.org/abs/2303.12917) exploit cross-scale, cross-group, and
cross-color prediction for lossless point-cloud attributes. Their cross-color stage processes
Y/Co/Cg or RGB components conditionally and shows content-dependent incremental savings on top of
the stronger spatial/group model. [MPEG G-PCC](https://www.mpeg.org/standards/MPEG-I/9/) already
contains color-space, prediction, lifting, and hierarchical attribute-transform machinery.

Therefore a reversible RGB lift can be a useful small StructSplat codec experiment, but not a new
color-coding method. Its scientific value would be exact complete-stream evidence about whether
the present independent SSP2 color channels leave enough correlation to matter.

## Repository-specific mathematical remainder

For fixed geometry, opacity absence, renderer settings, and unclamped normalized splatting, let
`A[p,i]` be Gaussian `i`'s normalized contribution to pixel `p`, `C` the `N x 3` decoded color
matrix, and `Y=A C`. A color perturbation `E=Ĉ-C` has exact source-render distortion

```text
D_source(Ĉ) = ||A(Ĉ-C)||_F^2
             = sum_channel e_channel^T (A^T A) e_channel.
```

For target-relative fitting, the exact quadratic contains the missing linear term:

```text
D_target(Ĉ)-D_target(C)
  = 2 <A C - T, A E> + ||A E||_F^2.
```

The full Gram matrix `G=A^T A` differs from per-Gaussian sensitivity weights because its
off-diagonal entries price cancellation and reinforcement between overlapping splats. That is the
narrow recipient-specific mechanism. It is exact only while renderer geometry and normalization
remain fixed and before output clamping changes the active set.

Two isolated pre-data implementations now establish mechanics, not benefit:

- `benchmarks/renderer_gram_aq_proof.py` checks exact sparse/dense source and target move deltas,
  monotone local updates, a global one-move terminal certificate, empirical rate deltas, and a
  matrix-free fixed-assignment codebook solve;
- `benchmarks/renderer_gram_structsplat_bridge.py` reconstructs the normalized StructSplat design
  matrix from geometry and checks it against the CPU renderer and color transpose/Jacobian action.

These proofs do not show a quality or rate improvement and do not authorize use of COMP-011 data.

## Adversarial kill memo

### Candidate A: another explicit flat/RVQ menu

**Verdict:** kill after COMP-011. CGVQ and prior Gaussian VQ already occupy the method, while
COMP-011 is the exact complete-stream killing test. A larger `K`, more stages, a different start,
or a softer assignment after seeing the cells would be a rescue sweep.

### Candidate B: scene-adaptive lattice VQ as method novelty

**Verdict:** kill the novelty claim; retain as a transfer only if explicit codebook bytes are the
measured bottleneck. SALVQ supplies the central implicit-codebook mechanism. The only legitimate
StructSplat question is whether a tiny basis/transform plus actual coordinate streams beats every
complete exact baseline under this three-dimensional, post-hoc setting.

### Candidate C: generic renderer-aware or Hessian-aware VQ

**Verdict:** kill the generic claim. RDO-Gaussian, sensitivity-aware Gaussian compression, and
quadratic model quantization supply the ingredients. Retain only the exact **off-diagonal normalized-
splat overlap** hypothesis, and require it to beat joint Euclidean VQ and diagonal-`G` weighting at
equal menu, bytes, and compute.

### Candidate D: reversible cross-color transform

**Verdict:** engineering/evidence experiment only. Point-cloud and image/video codecs already
exploit cross-component prediction and lifting. It survives as a low-cost exact control because it
could determine whether independent SSP2 color models, rather than lossy reconstruction, are the
remaining bottleneck.

## Conditional next experiments

### Branch 1 — explicit model cost is binding

**Trigger:** COMP-011 produces definitely quality-qualified variants whose index payload is
competitive, but codebook/model/framing bytes make the complete stream miss the rate gate.

**Simple core:** replace an explicit per-scene RGB codebook with an invertible, implicitly generated
three-dimensional lattice or a small frozen family of reversible modular color lifts. Store every
basis/transform choice and entropy-code every coordinate; select only by exact target-blind complete
bytes.

**Strong controls:** identity RGB, best existing exact SSP2 format, a fixed standard reversible
color lift, and the same entropy coder with shuffled channel correspondence. Decoder work, basis
precision, determinant/invertibility certificate, coordinate range, and transform ID are charged.

**Kill condition:** no new development screen if the fitted basis needs an unpriced codebook,
target access, or per-row side information. Close the fixed transform family if it does not improve
complete bytes consistently; do not enlarge the family on that screen.

### Branch 2 — rendering distortion is binding

**Trigger:** at least one COMP-011 arm has useful complete-rate headroom but fails conservative
quality, or selected Euclidean assignments exhibit large full- versus diagonal-Gram disagreement.

**Simple core:** minimize actual `R + lambda D` with `D` defined by the fixed normalized renderer's
sparse overlap Gram, using exact local move deltas and actual empirical index-rate deltas. The
decoder and stream grammar remain unchanged.

**Strong controls:** joint Euclidean ECVQ, diagonal-`G` ECVQ, full-`G` with its off-diagonal entries
permuted or removed, and the same assignments with codebooks re-solved. Compare source-render and
target-relative objectives separately; do not let target metrics choose starts or syntax.

**Distinct prediction:** full-`G` should help specifically in high-overlap cells where correlated
color errors cancel or reinforce, while matching diagonal weighting in near-disjoint cells. A
global gain with no interaction with measured off-diagonal energy would favor generic extra search
rather than the proposed mechanism.

**Kill condition:** the full Gram must improve realized clamped rendering, not only its unclamped
quadratic proxy, and must repay its encoder compute without adding decoder state. Failure against
joint Euclidean and diagonal controls closes this fixed-geometry formulation.

## Axis claims at this point

| Axis | Current conclusion |
|---|---|
| Quality | no new evidence; only an exact distortion identity and mechanics proof |
| Convergence | no new method evidence; local monotonicity is not fitter convergence |
| Performance | no new method evidence; sparse local updates are unbenchmarked on real cells |
| Compression | no COMP-011 result yet; literature supports two conditional hypotheses only |
| Expressiveness | unchanged in both contingencies; they alter color coding/selection, not the decoded primitive class |

The next scientific action remains: finish and audit COMP-011, then follow exactly one measured
failure branch on disjoint development data.
