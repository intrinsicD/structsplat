# Ideation audit: feature-targeted coverage matching (fit-time Gaussian blue noise)

**Scope.** Single-candidate pass through the `structsplat-research-ideation` skill (idea card,
transformation tests, adversarial prior-art audit, value scores, cheapest killing experiment).
This is *not* a full portfolio run; the generation lanes were not exercised because the request
was to audit one already-stated candidate. No production code is modified by this document.
Literature cutoff: **2026-07-23**, live web search (arXiv, ACM/Wiley/SIAM landing pages,
Semantic Scholar, project pages). Origin: design discussion of 2026-07-22/23 following FIT-021.

## Candidate formulation

Add to the fitter a differentiable **coverage-matching energy**

```
E = ∫ w(p) · ( S(p) − c(p) )² dp,      S(p) = Σ_i o_i · G_{Σ_i}(p − μ_i)
```

where `S` is exactly the normalized compositor's denominator (the raw opacity-weighted weight
sum the repo already computes in `raw_weight_map`, CORE-010/011), `c(p)` is a feature-weighted
target profile (structure-tensor energy + mask-boundary band boost; optionally an error-adaptive
blend), `w(p)` the masked domain weight, and `∫c` is normalized to the current detached total
kernel mass so the term only redistributes. Expanding the square gives closed-form
attraction-to-features plus pairwise anisotropic Gaussian repulsion — the "push" and "pull" are
one term, evaluated in field form at O(N·support) through the existing ragged-tile accumulator
rather than O(N²) pairwise. Opacities are detached inside `S` so the term transports geometry
instead of dimming rows. Scheduled to decay toward zero over the fit. Interacts with FIT-021
triage as continuous transport between discrete topology events.

## Idea card

**Central claim.** In a *normalized-compositor* Gaussian image representation, a two-sided
coverage-matching energy on the compositor's own denominator acts predominantly along the data
term's null directions (the redundancy gauge), and therefore improves equal-budget convergence
(PSNR-AUC) and low-budget quality over the same fitter with blue-noise *initialization only*,
without hurting terminal PSNR.

**Novelty class.** N2-T (unexplored transfer of a mature mechanism; recipient formulation
otherwise unchanged). The mechanism itself is *not* novel.

**Known foundation.**
- Kernel-variance blue noise: Gaussian Blue Noise (Ahmed, Ren, Wonka, ToG 2022); Fattal's
  kernel-density blue noise (ToG 2011).
- Attraction–repulsion point placement on images: Electrostatic Halftoning (Schmaltz, Gwosdek,
  Bruhn, Weickert, CGF 2010); attraction–repulsion dithering functionals (Teuber et al., SIAM
  J. Imaging Sci. 2011); halftoning-as-measure-approximation survey (Krahmer et al., GAMM 2025).
- Discrepancy flows: MMD gradient flow (Arbel et al., NeurIPS 2019) — `E` is a quadratic kernel
  discrepancy, so its particle gradient is exactly an attraction + pairwise-repulsion flow, with
  known convergence caveats; SVGD's attraction/repulsion decomposition is the same structure.
- In-repo: WSE blue-noise init (ADR-0005), one-sided band-limited under-coverage hinge
  (CORE-011 — "acts on the gauge the normalized compositor cancels"), out-of-mask coverage
  penalty (CORE-010), tensor loss weighting (FIT-012), responsibility machinery (FIT-018/021).

**Irreducible delta.** (1) The identification of the normalized compositor's *denominator* as
the discrepancy field — the regularizer reuses the representation's own normalization sum, so
anisotropy, opacity weighting, and the O(N·support) evaluation come from the renderer rather
than from an auxiliary kernel; and (2) the gauge claim: under a *normalized* compositor,
redundant capacity is loss-invariant, so a coverage term is not fighting the data term (as it
would under additive/alpha compositing) but selecting within its null space. Both are
recipient-specific; neither adds a new mathematical object.

**Why this is not merely A+B.** It admits to being close to A+B (GBN + Gaussian image fitting).
What survives subtraction is the compositor-specific mechanism claim: the same term under an
additive compositor is predicted to be *harmful* (there `S` is the image itself, so flattening
`S` toward a smooth `c` directly biases reconstruction), while under the normalized compositor
it is predicted to be ~orthogonal to the data term. That differential prediction is not implied
by any of the foundation works, and it is testable inside this one repo because both compositors
exist behind one flag (ADR-0003/0006).

**Changed grammar / preserved mechanism.** No grammar change (fitter regularizer; searchable
knob per ADR-0010). Preserved donor mechanism: quadratic kernel-discrepancy minimization by
gradient flow on kernel parameters.

**New predictions.**
1. *Gauge orthogonality*: pure-`E` gradient steps of matched parameter-norm change the pixel
   loss far less under the normalized compositor than under the additive one (ratio ≫ 1), and
   less than data-term steps of the same norm.
2. *Redundancy drain*: mean max-responsibility rises and FIT-021 merge-candidate counts fall
   during fitting, without a terminal PSNR penalty.
3. *Marginal value over blue-noise init*: a measurable AUC/low-budget-PSNR gain remains when the
   baseline already uses WSE blue-noise init (the native-baseline test most external papers
   would skip by comparing against random init).

**Cheapest killing test.** See below.

**Prior-art threats (strongest first).**
1. **Electrostatic halftoning** (CGF 2010) — literally attraction-to-image + pairwise repulsion
   with blue-noise behavior; differs in producing *dot placements* (no fitted covariance /
   color / opacity, no reconstruction compositor, no joint data loss), but a reviewer can
   frame the candidate as "electrostatic halftoning as a regularizer".
2. **Mini-Splatting / 3DGS-MCMC / Gaussian-herding OT reduction** — the recipient field already
   pursues Gaussian uniformity/redistribution, via discrete reorganization, stochastic
   relocation, or OT reduction; a differentiable coverage energy is a different mechanism but
   the same *motivation*, so novelty rests on the mechanism + compositor claim only.
3. **Instant-GI-style coverage-aware initialization** (and the Dec-2025 structure-guided
   allocation line) — coverage/overlap control at init; the candidate must show fit-time value
   *beyond* good init or it collapses into this bucket.
4. **MMD flow literature** — the math is standard; any theory claim beyond "we apply it" is
   likely already proved there (including failure modes: mode collapse of the flow, need for
   noise injection).

**Audit label.** Mechanism: *likely known*. Recipient formulation (denominator-as-discrepancy
+ compositor-differential prediction, jointly optimized with reconstruction in a Gaussian image
codec): *apparently unexplored under the stated search*. Overall: **known components, possibly
new relationship** — publishable only with the differential-compositor evidence, not as
"blue noise for Gaussians" per se.

**Novelty confidence.** Moderate-low (0.4–0.6 that the recipient formulation is unpublished as
of cutoff). Searched: arXiv (incl. 2512.24018, 2512.19108, 2403.08551, 2403.14166,
2206.07798, 1906.04370), ACM DL, Wiley/CGF, SIAM, Semantic Scholar, CVPR/WACV pages, a daily
GS paper index. Not searched: patents, non-English venues, closed reviews, code of every 2026
GS repo. The 3DGS literature's volume makes a miss plausible; a dedicated pass over 2025–26
"efficient/compact GS" papers is warranted before any submission-grade claim.

**Scientific value scores (0–5).** Apparent novelty 2; falsifiability 5; explanatory value 3
(the gauge story, if confirmed, explains *why* uniformity ops help normalized compositors);
importance 2–3; feasibility 5 (plumbing exists: `raw_weight_map`, tensor energy, masked
domain); cost of first test 5 (cheap); interpretable-result probability 4; strong baselines
available 5 (WSE init, undercoverage hinge, FIT-021 triage, all in-repo); robustness to
negative result 4 (informative either way given FIT-012/013/016 history); publication
potential 2 alone, 3–4 as a mechanism section inside a larger allocation paper.

## Transformation-test outcomes

- A-plus-B: fails as stated ("GBN + Gaussian fitting") → N1/N2 unless the compositor claim
  survives; it does, as a distinct falsifiable prediction → N2-T retained.
- Subtraction: delta reduces to two sentences (denominator-as-discrepancy; gauge orthogonality
  under normalization) — thin but nonempty.
- Grammar: unchanged (downgrades from any N3 aspiration).
- Prediction: passes (the additive-vs-normalized differential is not implied by prior work).
- Necessity: partial — the renderer-native field form is convenient, not necessary; the gauge
  argument is the necessary part.
- Compression: "Regularize the normalized compositor's weight-sum toward a feature-weighted
  target; under normalization this is a null-space transport, under additive compositing it is
  a bias." Survives without buzzwords.

## Cheapest killing experiment

Fixed-N transport isolation on the difficult-four proxy slice (CPU/GPU, no triage, no topology
events, so only the transport claim is tested):

- **Arms** (equal budget, iters, seeds ×3, WSE init everywhere): (a) baseline data term;
  (b) + coverage-matching `E` at three weights × decay on/off; (c) + one-sided undercoverage
  hinge only (existing machinery — is two-sided worth anything?); (d) same as (b) under the
  **additive** renderer (the differential prediction).
- **Null hypothesis.** (b) does not improve PSNR-AUC or 1k-budget PSNR over (a) by more than
  seed noise, or hurts terminal PSNR by > 0.05 dB.
- **Signature if correct.** Var[S−c] falls; prediction 1 ratio ≫ 1 in normalized and ≈ 1 in
  additive mode; (b) > (a) and (b) > (c) on AUC at ≥ 2 of 3 weights; (d) degrades.
- **Strongest conventional explanation.** "It only re-spreads a bad init" — excluded by using
  the blue-noise WSE init as the baseline; "it duplicates the hinge" — excluded by arm (c).
- **Confounders.** Weight/schedule tuning (cap at 3×2 grid, preregistered); opacity leakage
  (opacities detached by construction; verify with an opacity-frozen control if ambiguous);
  shared-seed coupling (paired comparisons per BENCH-002 conventions).
- **Abandonment criterion.** All (b) cells lose AUC *and* terminal PSNR on the proxy slice →
  close as an informative negative consistent with the FIT-012/013/016 regularizer record; no
  retuning beyond the preregistered grid, per repo no-rescue discipline.
- **Cost.** Hours on the proxy regime; one new loss term + logging, no architectural change.

## Entry path if pursued

New task `FIT-022-coverage-matching-regularizer.md` (knobs `coverage_match_weight`,
`coverage_match_target=tensor|tensor+boundary|error_blend`, `coverage_match_decay_frac`;
default off; shares the weight-sum accumulation with CORE-010/011 penalties), routed
`task-workflow` → `method` → `benchmark` → `structsplat-results-audit` → `review` → `docs-sync`.
Any external claim additionally requires a dedicated 2025–26 compact-GS prior-art sweep and the
repository's confirmation regime; the proxy screen above is a screen, not a promotion.

## Sources

- Gaussian Blue Noise — https://arxiv.org/abs/2206.07798 ; https://dl.acm.org/doi/abs/10.1145/3550454.3555519
- Fattal, Blue-noise point sampling using kernel density model — https://dl.acm.org/doi/10.1145/2010324.1964943
- Electrostatic Halftoning — https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2010.01716.x
- Dithering by Differences of Convex Functions — https://epubs.siam.org/doi/10.1137/100790197
- The Mathematics of Dots and Pixels (halftoning survey) — https://arxiv.org/html/2406.12760
- MMD Gradient Flow — https://arxiv.org/abs/1906.04370
- SVGD — https://arxiv.org/pdf/1608.04471
- Mini-Splatting — https://arxiv.org/html/2403.14166
- Gaussian Herding across Pens (OT reduction) — https://arxiv.org/pdf/2506.09534
- Effective-rank regularization for 3DGS — https://arxiv.org/html/2406.11672v2
- DropGaussian — https://arxiv.org/html/2504.00773v1
- GaussianImage — https://arxiv.org/abs/2403.08551 ; GaussianImage++ — https://arxiv.org/pdf/2512.19108
- Structure-Guided Allocation of 2D Gaussians — https://arxiv.org/pdf/2512.24018
- Instant GaussianImage — https://www.researchgate.net/publication/393184734
