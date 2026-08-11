# Pure-Gaussian normalized-to-additive continuation audit

**Scope.** Research-ideation front end for HIER-022. This note selects a cheapest killing test; it
does not contain experimental outcomes, choose Field V2 semantics, change a default, or claim
novelty. Literature cutoff: 2026-08-11. Repository state was checked against `CLAUDE.md`,
`docs/architecture.md`, `docs/additive_field_v2.md`, HIER-008/014/015/017/021, FIT-022,
BENCH-020, and the live code before task creation.

## Frontier and functional signature

The recipient problem is to infer a finite set of localized anisotropic kernels and coefficients
whose forward synthesis approximates a bounded image, remains stable under finite precision and
row removal, and supports a cheap random-access decoder. Geometry controls both approximation
support and numerical conditioning. Appearance is linear at fixed geometry for direct additive
coefficients, while normalized rendering divides by a geometry-dependent coverage field and is
therefore rational in the kernels.

The unresolved anomaly is not additive expressivity alone. HIER-008 has a stable nearly exact
full-lattice additive solve, whereas reduced HIER fields develop holes; HIER-013/014 map large
cancelling coefficients and show that a minimum-norm restart cannot rescue fixed geometry;
HIER-015's direct normalized fit is much stronger than that contracted line; HIER-017 maps a real
low-coverage denominator-floor failure; and HIER-021 repairs the known local tail only by leaving
the Gaussian representation. The useful question is which invariant must replace normalization,
not whether a sum of Gaussians is mathematically capable of image approximation.

Domain-neutral signature:

- **state:** localized kernels, geometry, appearance coefficients, and optional training gauges;
- **observation:** target samples on a finite rectangular domain;
- **operator:** local kernel accumulation, optionally divided by a coverage field;
- **conserved quantity sought:** near-uniform ownership/coverage without persistent denominator;
- **continuous variables:** position, covariance, coefficient, temporary mass;
- **discrete variables:** row count and future topology events;
- **failure boundary:** uncovered samples at one extreme, near-dependent overlapping columns and
  cancelling coefficients at the other;
- **evidence gap:** no matched trajectory observes coverage defect, conditioning, coefficient
  amplification, and quality while continuously moving between the two renderer equations.

## Fixation anti-library

The following are useful controls but not sufficient research claims: another loss-weight sweep;
lowering epsilon; adding broad background rows; running more Adam steps; applying coefficient L2
or a minimum-norm solve to unchanged geometry; copying source pixels; adding an unpriced neural
decoder; or calling generic coarse-to-fine fitting a new hierarchy. HIER-014 and HIER-017--021
already make several of these direct negative or non-pure controls.

## Candidate portfolio

| Candidate | Donor mechanism | Distinct prediction | First-test value | Disposition |
|---|---|---|---|---|
| Cold direct additive + matrix-free appearance solve | sparse least squares | solver helps only when the fixed basis is adequate | cheap negative control | retain control; HIER-014 threatens it |
| Coverage-constrained additive frame | RBF partition of unity / moving least squares | bounded coverage defect removes holes without a decode denominator | high | combine only after isolated coverage arm |
| Local frame-bound geometry selection | stable RBF bases / optimal experimental design | local Gram floor and bounded coherence predict coefficient parity | high, higher implementation cost | successor if conditioning remains causal |
| Normalized-to-additive homotopy | numerical continuation / system identification | a successful anneal distinguishes an optimizer basin from representation efficiency | highest information per cost | selected for HIER-022 |
| Parent-to-child zero-moment lifting | wavelets / multigrid | split transitions preserve reconstruction and avoid HIER-007 hotspot creation | strong hierarchy follow-up | defer until a stable additive endpoint exists |
| Normalized Gaussian base + additive Gaussian residual | multiresolution base/detail decomposition | coherent Gaussian residual groups repair HIER-021 neighborhoods | practical, but not one homogeneous equation | fallback if strict endpoint fails |
| Coarse Gaussian confidence prior | Bayesian shrinkage / confidence-weighted prediction | a learned coarse Gaussian prior replaces the black epsilon prior only at low support | cheap but strongly threatened | deprioritize after HIER-018/019 |

The selected candidate is deliberately a recombination rather than a novelty claim. Its
scientific value is diagnostic: the trajectory can map whether the additive gap comes from
optimization path, coverage conservation, or the finite-count representation itself.

## Selected idea card

**Central falsifiable claim.** A shared anisotropic Gaussian field trained through a normalized to
direct-additive continuation, with a training-only coverage gauge driven toward one, can end at a
bounded pure-additive field that closes at least half of the current matched normalized/additive
quality gap without source-derived residuals.

**Novelty class.** N2-T / known components, possibly new recipient-specific relationship. The
classification must be downgraded if prior art contains the same shared-field homotopy and
discarded coverage gauge for 2D Gaussian image representation.

**Known foundation.** Normalized Gaussian RBF networks; RBF partition-of-unity and least-squares
stabilization; numerical continuation; variable projection; GaussianImage's accumulated-sum image
representation; Image-GS's normalized top-K field; and the local StructSplat evidence above.

**Irreducible delta.** Measure one shared finite-support anisotropic field while the denominator is
continuously removed, require the coverage gauge and coefficient range to remain observable, and
make the exact lambda-zero field—not a rational checkpoint—the only admissible saved result.

**New prediction.** If cold additive fitting is primarily trapped by the optimization path,
continuation succeeds even before explicit frame conditioning. If coverage becomes near-uniform
but quality collapses as lambda approaches zero, normalized rational functions are more
parameter-efficient at the measured count. If coefficient amplification tracks a local condition
diagnostic instead, basis design—not renderer semantics—is the next causal intervention.

**Cheapest killing test.** Programmatic endpoint/gradient fixtures followed by COCO4 at
max-side 160, N=640, two seeds, and four matched arms. HIER-022 freezes the exact schedule,
coverage-weight selection rule, metrics, and abandonment gate.

**Prior-art threats.** Normalized RBF networks already interpret normalization as partition-like
ownership; RBF-PU methods already combine local approximation with a unity constraint; stable RBF
bases and continuation are mature; GaussianImage and Image-GS already occupy additive and
normalized endpoints. A publication contribution cannot be “normalization annealing” alone.

**Novelty confidence.** Low-to-moderate, roughly 0.25--0.45 that the exact recipient formulation is
unpublished under the searched terminology. Searches covered arXiv, official project/paper pages,
SIAM/ScienceDirect landing pages, repository documentation, and terms around additive/normalized
Gaussian image rendering, normalized RBFs, partition-of-unity RBF approximation, stable Gaussian
RBF bases, continuation, and variable projection. Patents, closed reviews, non-English venues,
and every 2025--2026 compact-GS repository were not exhaustively searched.

## Cross-domain transfer map

| Structural role | Donor systems | StructSplat recipient |
|---|---|---|
| state | basis coefficients plus continuation parameter | Gaussian geometry, RGB coefficients, temporary masses, lambda |
| observation | residual of a parameterized equation | reconstruction error and coverage field |
| action | follow a solution branch while changing the equation | anneal normalized denominator to one |
| invariant | partition of unity / bounded frame | `D` near one and bounded coefficients |
| failure mode | branch loss, singular basis, ill-conditioned interpolation | quality cliff, holes, cancelling RGB coefficients |
| boundary condition | bounded approximation domain | finite image canvas and clipped Gaussian support |

Preserved mechanisms are path following and explicit conservation of coverage. Broken
correspondences are important: image fitting is nonconvex in geometry; finite support makes the
coverage field nonsmooth at tile-boundary changes; and a low image loss does not certify a frame
bound. Those are correctable for a diagnostic but prevent importing donor convergence theorems.

Donor fields represented in the portfolio are numerical continuation/control, approximation
theory and RBF frames, optimal experimental design, multigrid/wavelet lifting, Bayesian
confidence estimation, and sparse numerical linear algebra. The measurement transfer—jointly
plotting coverage defect, coefficient amplification, and quality against lambda—is at least as
important as the candidate optimizer.

## Prior-art anchors

- GaussianImage, accumulated-sum 2D Gaussian image representation:
  https://arxiv.org/abs/2403.08551
- Image-GS, content-adaptive top-K normalized Gaussian representation:
  https://arxiv.org/abs/2407.01866
- GaussianImage++, distortion-driven densification at high reconstruction error:
  https://arxiv.org/abs/2512.19108
- AbsGS, gradient-collision-aware Gaussian densification:
  https://arxiv.org/abs/2404.10484
- Normalized Gaussian radial-basis-function networks:
  https://doi.org/10.1016/S0925-2312(98)00027-7
- A stable RBF basis:
  https://arxiv.org/abs/1210.1682
- Least-squares RBF partition-of-unity approximation:
  https://doi.org/10.1137/17M1118087

## Evidence program and decision boundary

HIER-022 first asks whether the exact additive endpoint is reachable at all under a matched small
screen. Success justifies a separate, disjoint conditioning-aware and larger-budget task; it does
not select Field V2 semantics. Failure remains useful when the lambda trajectory identifies a
coverage, conditioning, or representation boundary. No failed natural-image gate may be rescued
by retuning its images, and no result changes the maintained renderer without BENCH-020 and the
downstream production gates.

## HIER-022 outcome and updated causal model

The frozen diagnostic selected coverage weight `0.05` and completed all 32 natural-image cells.
Every selected endpoint is an exact finite `lambda=0`, `N=640`, mass-free additive field with cold
parity at most `4.77e-7` and coefficient maximum `2.831`. Endpoint reachability is therefore not
the failure.

The central claim is rejected. Mean normalized, additive, no-coverage continuation, and coverage
continuation PSNR are `26.840`, `26.291`, `26.045`, and `25.837 dB`. Explicit coverage succeeds at
its own objective—mean `(D-1)^2` falls from `0.51250` to `0.01405`, or 97.3%—but the candidate is
`0.454 dB` below plain additive, raises LPIPS by `0.0120`, and worsens displayed pixel and complete
7x7 maxima. The no-coverage path is less harmful but still `0.246 dB` below plain additive after
500 steps. Thus neither support coverage nor exact endpoint feasibility explains the gap on this
screen.

Trajectory telemetry identifies a more specific confound: independently learned numerator and
mass variables do not reproduce the ordinary normalized optimizer path even while `lambda=1`.
At the end of the 35% hold, no-coverage continuation averages about `0.58 dB` below the ordinary
normalized trajectory. The learned gauge also spends only 15% of the horizon at the true additive
equation. The cheapest new discriminating test is consequently a **unit-gauge continuation**:
`A=sum(c_i G_i)`, `D=sum(G_i)`, and
`I_lambda=A/[lambda(D+eps)+(1-lambda)]`. This is exactly the ordinary normalized renderer at
`lambda=1` and exactly direct additive at `lambda=0`, requires no masses or coverage loss, and can
devote 50% of training to the endpoint. A separate optimizer-state-reset arm isolates stale Adam
moments. This is a new HIER task and disjoint development selection, not an in-place rescue.

## HIER-023 outcome and next discriminator

The unit-gauge implementation removes HIER-022's trajectory confound. The maximum step-175 PSNR
difference from the ordinary normalized control is `0.0344 dB`, within the frozen CUDA tolerance,
and all 16 candidate endpoints are finite, exact-additive, mass/auxiliary-free fields. Resetting
Adam at the first endpoint step is negative (`-0.0700 dB` versus continuous moments), so no-reset
is selected.

The result separates convergence from representation efficiency. No-reset averages `29.0524 dB`,
only `0.0326 dB` below a 500-step additive fit despite spending just 250 steps at that equation. It
also improves mean LPIPS by `0.00357`, pixel maximum by `0.00406`, 7x7 maximum by `0.00703`, and
PSNR-AUC by `1.205`. Yet normalized averages `29.7498 dB`: the candidate retains none of the
`0.6648 dB` normalized/additive gap, and one `0343` cell raises LPIPS `0.01226`. Native review finds
no new structural artifact, only the common fixed-count blur. The frozen mechanism claim is
therefore rejected even though the path is an efficient warm start for additive fitting.

Seven of eight selected fields choose step 500 and remain visibly rising in their endpoint tail.
That does not authorize a schedule retune: a longer path could merely spend more work. The sharper
next test holds geometry fixed and applies the already implemented safeguarded matrix-free all-row
RGB solve to both cold additive and unit-gauge endpoints. If both move equally, normalization's
fixed-count advantage is geometric/functional and basis redesign is required. If gauge geometry
moves materially more, the remaining failure was coefficient optimization and a pure additive
composition can be retained without a denominator.

## HIER-024 outcome and basis decision

The fixed-geometry solve is not the missing mechanism. On a third hash-selected DIV2K4x2 bank,
the safeguarded all-row PCG improves ordinary additive by `0.12996 dB` and unit-gauge geometry by
`0.17191 dB`; both select seven of eight proposals and fail closed on the remaining cell. The
gauge-specific gain is only `0.04195 dB`, below the frozen `0.05 dB` discriminator, and its final
field is only `0.01046 dB` better than projected additive. It closes 1.91% rather than half of the
positive normalized gap. Mean LPIPS/MS-SSIM improve, but mean and per-cell local-error gates fail.

All projected endpoints preserve exact N=640 geometry, remain bounded pure additive fields, and
cold-render within `3.19e-6`; the negative result is not a safety or persistence failure. Visual
review instead exposes seed-sensitive support placement, especially a broad misplaced lobe on
`0571`, that coefficient changes cannot move. This closes the coefficient-optimization branch.

The next admissible branch is **basis surgery without renderer semantics**: reserve counted broad
Gaussian carriers for low frequencies and use the remaining rows for anisotropic residual/detail
support, then jointly fit and optionally apply the same safeguarded RGB solve. The endpoint remains
one ordinary Gaussian sum—no denominator, mass, pixel patch, or auxiliary residual stream. A new
task must freeze carrier count, geometry, initialization, controls, and a new selection before
opening pixels; HIER-024 data and thresholds cannot be reused for tuning.

## HIER-025 outcome: disconnected multiscale fitting is not the basis repair

HIER-025 executes that counted basis on the four remaining repository DIV2K files, two seeds, and
the unchanged N=640/500-update controls. The endpoint contract succeeds exactly: 16 coarse plus
624 detail rows, one direct additive pass, no opacity/mass/denominator/optimizer/scaler/residual/
level payload, exact coarse-geometry freeze, removed training mask, bounded coefficients, and
sub-`2e-6` cold parity.

The quality hypothesis fails broadly. Folded versus direct additive is `-1.55421 dB`; after both
receive the same safeguarded all-row solve, folded is still `-1.40831 dB`. Its mean MS-SSIM,
LPIPS, pixel maximum, 7x7 maximum, and full-target PSNR-AUC are all worse, and per-cell guards fail.
The proxy trajectory reaches only `30.334 dB` by the end of step 400 before the mixed-loss joint
stage and finishes at `30.873 dB`, versus `32.428 dB` for ordinary full-target additive fitting.
Native review confirms diffuse loss of thin skyline, insect, and aircraft detail rather than a
serialization artifact.

This rejects the literal “fit low pass, fit residual, then briefly polish” transfer from LIG under
the strict one-sum/equal-update contract. It does not reject multiscale Gaussian sums in general:
LIG keeps separately rendered/scaled levels, while the strict fold removes that machinery. The
next admissible test preserves a fully optimized additive base and changes capacity progressively,
with fresh output/data binding and explicit row/work accounting. HIER-025's official validation
files remain unopened.

## HIER-026 outcome: normalization buys row efficiency, not exclusive representation

HIER-026 opens those four official validation files only after binding archive/member hashes,
seven arms, two seeds, counts, schedules, work units, and a composite quality gate. Its progressive
candidate preserves a fully fitted additive N=640 base, inserts 256 signed residual Gaussians, and
jointly fits all N=896 rows for 200 further updates. Both base and candidate persist as exactly four
arrays and render in one ordinary additive pass. Cold projected N=896 and N=960 controls separate
staging from raw capacity.

The denominator is not uniquely responsible for low distortion. Projected progressive N=896 beats
normalized N=640 by `+0.75388 dB` mean PSNR and in every cell (minimum `+0.04411 dB`); cold N=960
is `+0.94493 dB` on mean with minimum `+0.35273 dB`. Both improve mean MS-SSIM and pixel/7x7
maxima; progressive also improves mean LPIPS. Same-count projected additive remains `-0.84193 dB`,
so this is a capacity exchange, not evidence that additive semantics are better at N=640.

The frozen all-metric gate nevertheless rejects both tested rungs. The dense-forest `0860` seed-0
cell raises progressive/cold-N=960 LPIPS by `0.05447/0.02910`, and native review finds diffuse
directional foliage smear where normalized rendering instead has polygonal coverage breakup. Cold
N=960 also breaches one other LPIPS cell and one pixel-maximum guard. Normalization is therefore
not necessary to exceed PSNR/MS-SSIM/local fidelity, but N<=960 is not yet a robust perceptual
substitute under the predeclared rule.

Post-decision probes on the now-consumed HIER-026 bank locate a simpler successor. Ordinary cold
projected additive N=1088 passes every numeric clause across all eight cells with
`+1.68200/+0.98761 dB` mean/minimum PSNR deltas and worst LPIPS delta `+0.00335`; N=1152 also
passes, while N=1024 still fails the forest killing cell. Those probes choose counts only. A new
untouched selection must confirm the ordinary additive N=1088 candidate and N=1152 fallback before
describing normalization as unnecessary for tested max-side-160 fidelity.

## HIER-027 outcome: capacity alone leaves sparse extrema

The frozen eight-image/two-seed confirmation rejects both ordinary cold-capacity rungs under the
unchanged local gate. Projected N=1088 and N=1152 are globally compelling: they beat normalized
N=640 by `+1.84883/+2.19555 dB` mean PSNR, their minimum paired gains are
`+1.34241/+1.55374 dB`, and mean MS-SSIM, LPIPS, pixel maximum, and 7x7 maximum all improve. Every
per-cell PSNR and LPIPS guard passes. Yet each rung has two isolated pixel-maximum regressions above
the allowed `+0.02`, while the corresponding 7x7 maxima pass. Native review finds no broad new
artifact, so count is not the remaining robust-control mechanism.

This is stronger than a generic “more Gaussians help” result and narrower than a representation
claim. It identifies a sparse allocation defect: hundreds of additional cold rows improve the
distribution but can miss the few pixels that define the hard maximum. The bank is sealed without
threshold relaxation. Consistent with GaussianImage++'s distortion-driven densification principle,
the next prospective method spends a small explicit tail at current worst residuals.

## HIER-028 outcome: a bounded pure-additive solution

HIER-028 confirms the N=960+64 residual-pursuit recipe on eight further untouched official DIV2K
images and two seeds. The projected cold N=960 prefix is preserved bit-exactly. Each of 64 appended
rows uses a fixed 0.35-pixel isotropic Gaussian at the row-major highest raw-RGB-MSE pixel and takes
the current signed residual as its coefficient; no optimizer, adaptive count, residual raster, or
source-derived state survives. The final N=1024 field contains only means, log-scales, rotations,
and signed RGB and cold-renders in one ordinary additive pass.

The candidate reaches `31.08179 dB`, `+1.62037 dB` over normalized N=640 on mean with a
`+1.14979 dB` minimum paired gain. Mean MS-SSIM improves from `0.983110` to `0.989308`, LPIPS falls
from `0.079350` to `0.057393`, pixel maximum falls from `0.34432` to `0.14257`, and 7x7 maximum
falls from `0.12038` to `0.07428`. Every frozen aggregate and per-cell clause passes. Both local
maxima also improve over the exact N=960 base in all 16 cells. A separately fitted same-count cold
N=1024 control reaches `+1.39705 dB` but fails its per-cell local clause, isolating residual-aware
allocation rather than count alone. Native audit finds no material lattice, ringing, holes, wash,
color lobes, blur, or tail speckle; magnified suspicious points align with real source texture.

The answer to the bounded research question is therefore **yes**: normalization is not required
for this max-side-160 fidelity gate if the encoder pays 1.60x Gaussian rows, 1.50x base row-update
work, and 64 full target-known residual scans. It is not “better than normalization” without that
qualification. Normalization remains more row-efficient and stays the maintained default;
same-count additive N=640 is worse, complete bytes are unmeasured, and full-resolution, broad-
corpus, downstream, and actual-rate behavior remain open. The residual-pursuit method is default-
off research evidence, not a production-pipeline or novelty promotion.
