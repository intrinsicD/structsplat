# Overnight findings and handoff — 2026-09-05

No maintained default change is justified by tonight's results. Four experimental routes now have
implementations, clean-source experiments and independent audits. The useful outcome is a clearer
map of where the simple methods work, where they fail, and what to test next.

## Results at a glance

| Route | Finding | Practical disposition |
| --- | --- | --- |
| Pixel-gradient action selector — HIER-033 | Joint low-regret gate passes in13/18 constructed cases, below the required15/18. Cancellation does not uniquely identify the useful edit family. | Keep the diagnostic packet and finite oracle; do not adopt the selector. |
| Diagonal/local-block curvature — HIER-035 | Helps translated/anisotropic fixtures near numerical precision; loses every overlap and texture condition to the strongest Adam control. | Numerical-polish evidence, not a generally better fitter. |
| Dense cross-Gaussian coupling — HIER-036 | Helps realizable overlap, but the texture coupling gain does not persist to additional conditions; full curvature loses every texture condition to strongest Adam. | Keep the small diagnostic oracle; no production dense solver or optimizer promotion. |
| Fixed-geometry cache — HIER-034 | Only irregular-mask workloads pass all interchangeability checks, and every passing call returns the unchanged input. Natural-image pixel parity fails even between streaming repeats. | Keep opt-in and experimental; no general interchangeable-backend or accepted-solve speed claim. |

These findings are bound to **C68–C72** in the [claim ledger](../../ara/logic/claims.md) and the
[complete independent audit receipt](../../ara/evidence/overnight-method-research-2026-09-05/run.md).
All five formal matrices completed:270 operator cells,60 local-optimizer cells,84 coupling cells,
and two180-cell cache profiles. Cells are not independent experiments, and the cache profiles
reuse the same planned workloads.

## What matters beyond the headline numbers

The large curvature gains on the easy fixtures are mostly precision-limit polishing, not visible
detail recovery. Conversely, the texture fits reject many attempted updates and pay for every
failed trial render. A fixed attempted-update budget is not equal computational work, and none of
these optimizer tests establishes isolated speed or natural-image transfer. The
[native comparison](../../ara/evidence/overnight-method-research-2026-09-05/hier036_visual_qa.png)
illustrates this distinction; it is not an additional selection gate.

The action oracle gives useful counterexamples. Low position-gradient coherence can accompany
either a useful continuous edit or a useful split. An uncovered residual can have zero gradients
for all existing rows. Immediate and recovered edit rankings can differ, so a recovery gain must
be compared with an equally recovered no-op. These are finite, donor-funded N=3 examples, not a
new splitting theorem or a result about downstream3D reconstruction. See
[all action decisions](../../ara/evidence/overnight-method-research-2026-09-05/archive/hier033/decision.json).

The original cache timing predicates nominally pass on six rollback workloads at1.691–2.009x
median paired ratios. Those calls execute the solver but accept no fitted update. The occupancy
log also records two foreign-GPU episodes, and unrelated CPU compilation was observed separately.
The ratios are therefore observed shared-workstation measurements of rollback calls, not useful
accepted-solve or isolated acceleration. The separate shared-correctness profile remains
categorically timing-ineligible. Its restrictions were not relaxed after seeing outcomes.

Streaming's own natural-image repeat variability prevents blaming every parity failure on the
cache. Small aggregate PSNR differences can hide large localized differences, so neither endpoint
PSNR near-equality nor matching selected iteration is a sufficient parity check here.

## Recommended next bounded test

The [portfolio's final sketch](2026-09-05-overnight-research-portfolio.md#12-unrun-follow-up-sketch-direction-rescue-versus-deeper-backtracking)
specifies a simple projected-gradient rescue after six failed GN trials, compared with twelve GN
backtracking trials and the same strong Adam controls. It has independent design critique, but is
**not implemented, executable-approved or run**.

Keep that test small: separate exposed and additional conditions, charge actual trial work, log
the projected direction, and require both the deeper-backtracking and Adam comparisons to pass.
This tests a safeguard; it does not presume the cause of the texture failure. A standalone
projected-gradient control would be needed before claiming hybrid synergy.

For caching, the next prerequisite is a controlled baseline-repeat/stability diagnosis and a
prospectively defined accepted-update utility gate. Do not reinterpret tonight's rollback passes
or tune the failed thresholds in place.

The broader [simpler-first portfolio](2026-09-05-overnight-research-portfolio.md) preserves scheduling,
support-event, representability-certificate, donor-recipient and image-space ideas with primary
literature links and killing tests. They remain hypotheses with bounded novelty confidence.

## Implementation and evidence

New tools are confined to the experimental additive path: owned fixed-basis scatter/CSR caches,
a pixelwise Jacobian/gradient/Gram/split-curvature reference, local optimizer controls and a
memory-bounded dense coupling oracle. The maintained normalized renderer and fitting defaults
are unchanged. Every cache must be rebuilt after geometry, support, mask or row changes.

The [tracked evidence archive](../../ara/evidence/overnight-method-research-2026-09-05/archive/index.html)
is explicitly partial, with every omission listed. It retains all decisions and scalar rows,
configurations, histories, native fields and selected displays. The full raw rasters, all
learning curves and native reports remain in the untouched local bundles:

- [Operator oracle](../../results/hier033_operator_oracle_2026-09-05/index.html)
- [Local convergence controls](../../results/hier035_convergence_2026-09-05/index.html)
- [Coupling/cap factorial](../../results/hier036_coupling_2026-09-05/index.html)
- [Shared cache correctness](../../results/hier034_shared_correctness_2026-09-05/index.html)
- [Original cache timing](../../results/hier034_basis_cache_2026-09-05/index.html)

Each original manifest records the exact source commit, approved protocol digest, command, inputs
and environment. Reproduction uses that frozen commit, not an outcome-tuned current checkout;
the natural cache assay also requires its recorded exposed HIER-031 base bundle. The partial
archive is not advertised as a complete stand-alone replay package.

All five full reports pass the maintained bundle checker. The final packager/source gate passes
2,163 portable tests,26 skips and514 deselections, with all structural checks clean. Distinct
post-run audits recompute the decisive metrics, saved-field replays, gates and work accounting.
No failed formal cell was selectively repeated, no immutable bundle was repaired, and no foreign
process was stopped. Work is on local branch `research/overnight-2026-09-05`; no push or merge.

The distinct integrated review accepted all four tasks for retirement, not method promotion.
Its verdicts are recorded in the retired
[HIER-033](../../tasks/done/HIER-033-pixel-gradient-operator-oracle.md),
[HIER-034](../../tasks/done/HIER-034-fixed-geometry-basis-cache.md),
[HIER-035](../../tasks/done/HIER-035-additive-convergence-controls.md), and
[HIER-036](../../tasks/done/HIER-036-dense-coupling-oracle.md) task records.
