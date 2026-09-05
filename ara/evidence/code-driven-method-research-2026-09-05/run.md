# Code-driven method research — complete evidence and audit

Date: 2026-09-05. Driver: codex-root. Distinct prospective and outcome reviewer:
codex-code-research-reviewer (`/root/overnight_protocol_reviewer`).
User authority: inspect the code, create ideas, design, implement and run experiments; prefer
simpler improvements to convergence, quality, performance or efficiency. Method interpretations
are AI-derived, not user-affirmed. No default, sealed-data, cloud-spend or foreign-process action.

## Outcome

Three complete, immutable studies contain 214/214 prescribed cells. All maintained full-bundle
checks and independent raw CPU/GPU audits pass without allowances. All candidate frozen
utility/promotion gates are negative. Retain the experimental tools, not a promoted method.
The source-bound findings support ADR-0034 and ARA C73–C75; they do not establish novelty.

| Study | Cells | Clean source commit | Exact approved protocol digest |
|---|---:|---|---|
| FIT-050 | 48 | e2bf6ae6e06ca8050d9aa8a93d713679a0c9c150 | edad3bb041fb2e34697a101f729402bf12470c411a5abe2567a91be04a027153 |
| PORT-007 | 110 | e2bf6ae6e06ca8050d9aa8a93d713679a0c9c150 | 730b7704c78e61de6c8d213cc2a013b455516fc73d668b4899564b7b1dea507c |
| FIT-051 | 56 | 8de800406e608d7c7a47cc3dfc56217ed69bbb53 | e1c3a421cef3e546f8135d405ee1315b76c47e11f9286b8ea68170930ee32010 |

Full bundle root: `/home/alex/Documents/structsplat/results/code-driven-2026-09-05/`.
Subdirectories: `fit050-v1`, `port007-v1`, `fit051-v1`. Each has `index.html`,
`manifest.json`, full raw arrays, native fields, metrics, decisions and temporal evidence.
The exact original manifest hashes and complete source/artifact inventories are preserved by
[archive_manifest.json](archive/archive_manifest.json) and each archived original manifest.
FIT-050's parent manifest SHA256 is
`5c5629df090b7946f1ee85ab98d988921998096888261c40192e7cd3a7a4f427`.

## Code-derived design and limits

The [portfolio](../../../docs/research/2026-09-05-code-driven-portfolio.md) records independent
code-reading lanes, multiple productive/assumption/primitive/evidence routes, cross-domain
transfers and adversarial prior-art screening. Chosen ideas are simple known mechanisms:
fixed-geometry linear RGB proposals with safe fractional steps; same-call common-subexpression
reuse; and a distinct actual-render follow-up when compatibility prevents the first test.
The analytic CVaR counterexample and image-space projection obstruction are analytic only,
not executed Gaussian experiments. All transformational novelty labels were downgraded; no
global novelty assertion follows. The older HIER projected-GN rescue proposal is still unrun.

All formal fitting data are the same four exposed COCO development images (IDs9,25,30,34),
Pillow LANCZOS max-side512. Seeds0/1 are clustered within images, not eight independent images.
Parents use quadtree_wse N2000, fixed topology, 750 terminal normalized CUDA L2-only Adam steps,
no opacity, no regularizer/schedule/color solve, learning rates means0.05/scales0.03/rotations0.01/
colors0.03, render chunk512, history every25 steps. Complete target/config/initial/terminal/
optimizer/history payloads are hash-bound. The environment is Python3.12, torch2.9.0+cu128,
RTX3050 8GiB, one torch CPU thread. Full versions/configurations are in the original manifests.

## FIT-050: safeguarded normalized color rays

Each of eight parents receives independently charged noop, legacyCG32, interpolatedCG32,
streaming gradient, exact-diagonal Jacobi gradient and inherited-moment all-parameter Adam32.
Color arms freeze geometry, opacity, support and count; signed colors remain unclamped.
Ridge1e-4 and six fractions 1,1/2,1/4,1/8,1/16,1/32 are frozen. Candidate images interpolate
a streaming linear basis; selected fields require actual maintained-render replay and the full
unchanged safe gate. Invalid proposals or failed selected replay roll back exactly.

Utility requires median of four image-level seed-mean gains >=0.1dB, no image-seed PSNR loss
>0.01dB, MS-SSIM loss>0.001 or LPIPS increase>0.002, plus every native safe-gate clause.
Acceptance of a numerical tie is not useful improvement. Parent-only gains do not establish
preference over CG or Adam; Adam continuation has deliberately unequal full-parameter work.

| Arm | Accepted / 8 | Median image-level PSNR gain (dB) | Utility |
|---|---:|---:|---|
| legacyCG32 | 4 | 0.00421580323 | false |
| CG ray | 1 | 2.88374e-9 | false |
| gradient ray | 1 | 1.98389e-9 | false |
| Jacobi ray | 1 | 2.37943e-9 | false |
| Adam32 | 1 | 2.21482e-10 | false |

Twenty-one of 24 ray transactions abort `basis_parent_parity_failed` before constructing a
direction or testing a fraction. Seven parents exceed the fixed2e-5 compatibility threshold,
with max RGB errors2.0206e-5–6.7651e-5. Only COCO30/seed0 is eligible (~9.1344e-6);
its three full rays are accepted (+0.005304/+0.001663/+0.002146dB CG/gradient/Jacobi).
This is an implementation compatibility obstruction, not a general negative result for line
search or preconditioning. The exact numerical source is unresolved. Four legacy endpoints
improve MSE but are rejected by CVaR. Early-aborted ray costs are not successful-work speedups.

Independent audit checked648 artifact hashes,62 source hashes,48 raw MSE/PSNR values
(max PSNR discrepancy7.1e-15dB),40 bitwise-parent selected rollbacks,331 transaction renders,
96 report/replay renders,6000 parent updates and256 Adam continuation updates.
Cold GPU replay of all48 fields has max RGB error4.7683716e-7; original CUDA perceptual
endpoints reproduce exactly. All99 occupancy samples have no foreign process/query error.
These are point observations, not proof of continuous workstation exclusivity.

## PORT-007: same-call coverage and tail reuse

Five arms are legacyA, independent legacyB, coverage reuse, shared tail and both. Both is the
preregistered primary arm; single axes are explanatory, never outcome-selected. Eight original
FIT-050 parents transfer within the same source commit, with all initial and terminal states.

The80 same-state cells cover4images x2seeds x2states x5arms. Every cell retains ten fresh timed
full quality calls in five cyclic orders plus reversals, two warmups, and a separately untimed
instrumented probe with actual RGB, raw denominator and full quality vectors. Timed calls
themselves are uninstrumented. The coverage wrapper falls back conservatively for unsupported,
nonfinite, negative, near-threshold or nonzero-outside cases; shared top-k preserves frozen
CVaR and pinned torch quantile arithmetic, with explicit unsupported-input fallback.

The30 complete pipelines cover COCO9 full-frame and COCO25 fixed synthetic ellipse x3seeds x5arms.
Capacity1000, step_scale0.025, block_steps25, Pareto every25, all scheduled phases retained.
Observer CPU clones and full selected-trajectory snapshots are charged to instrumented runtime;
offline perceptual scoring is outside that timer. This is not the default11000-row workload or
a semantic-mask generalization test. The full raw/work/configuration matrix is retained.

All110 cells complete, but every candidate's frozen component correctness/speed and pipeline
quality/speed/trajectory gate is false. Same-state A/A timing is stable on all16 states.
The only same-state discrete failures are null-gain changes from `no_material_gain` to accept:
legacyB20/160, coverage15/160, tail18/160, both16/160. Changed-state decisions and every hole mask
agree. Max cross-arm RGB discrepancy5.9604645e-7 is also reached by within-legacyA repeats;
reuse-specific numerical causality is not isolated. Coverage/both fall back on exactly4/16 states.

Observed median of four image-level component ratios is10.2484x coverage,1.00283x tail,
10.4008x both. Coverage image ratios span5.0726–11.0670x and both5.1628–11.2692x.
These are component-only descriptive observations despite failed formal gates, not validated
production speedups. Pipeline A/A initial fields are bit-identical but final trajectories differ
on all6 pairs;4/6 timing and3/6 quality pairs fail. Across all30 pipelines there are51,892
attempted versus8,952 accepted steps. Equal final N1000 does not mean execution-equivalent work.
Both's observed per-image total-time ratios1.03486/1.14965 are causally ineligible; only2/6
primary quality pairs pass. No reuse-specific pipeline acceleration or degradation is established.

Independent audit checked2306 artifact hashes,60 source hashes,800 raw parity records,
3200 exact decisions,1600 same-state plus30 pipeline quality vectors with a separate NumPy/
SciPy implementation, and654 selected native snapshot/event/work bindings.
Cold GPU replay of110 endpoints has max RGB error5.3644180e-7; CUDA perceptual scores reproduce
exactly. All84 occupancy samples have no foreign process/query error, without continuous
exclusivity inference. Timing ineligibility is not attributed to nonexistent foreign activity.

The original unoptimized frozen checker completed PASS without allowances for both driver and
reviewer. A post-run private geometry-cache optimization also passes the complete checker in
both worktrees: only owned mask geometry keyed by exact content/dtype/shape and all support
settings is reused within one validation call. Raw images, metrics and decisions are not cached.
Nineteen new cache tests and thirteen independent artifact regressions pass; no formal source
or immutable bundle was altered, and no validator speed is counted as method performance.

## FIT-051: actual-render transaction follow-up

This is a new mechanism and source freeze, not a repair/selective rerun of FIT-050. All eight
original parents and48 transferred payloads retain the explicit original manifest/source hashes.
Seven arms are noop, legacyCG32, actualCG fractions, actual streaming-gradient fractions,
actual Jacobi fractions, native-gradient fractions and inherited-moment Adam32.

Every trial is an actual maintained render of a changed field, not image interpolation.
Streaming proposals use the actual parent residual but remain approximate cross-backend directions.
Native VJP requests a cloned color leaf with all protected parameters fixed; the existing
backward can compute unused buffers and its entire invocation is charged. Signed direction q
is actually rendered and alpha is (r*q).sum/(q.square().sum+ridge*v.square().sum).
Keep ridge1e-4, six fractions, complete gate, first safe nonzero trial and selected replay.
CG fraction1 is exactly its own independently solved endpoint. Failed replay rolls back without
trying a later fraction. All proposed operands, rejected endpoints, raw denominators, actual
trials, selection/replay distinctions and method/stage-derived work counters are retained.

Before formal outcomes, procedural checks falsified CPU/CUDA LPIPS equivalence at the frozen
report tolerance (CPU0.19611902535 vs CUDA0.19611391425; near-target1.38985e-7 vs2.77357e-7).
Reporting was prospectively changed to one detached CPU float32 scorer for all endpoints and
curve points, one thread, outside transaction timing; native CUDA safety decisions are unchanged.
No tolerance was widened. Revised GPU-input/canonical-CPU tests pass. Older FIT-050 perceptual
values are not directly comparable across this changed reporting backend. The failed pre-run
diagnostic and provisional stale-source portable test failures were retained, not formal outcomes.

| Arm | Accepted / 8 | Median image-level PSNR gain (dB) | Utility |
|---|---:|---:|---|
| legacyCG32 | 4 | 0.00421578335 | false |
| actualCG ray | 8 | 0.00529267289 | false |
| actual streaming gradient | 7 | 0.00121955267 | false |
| actual Jacobi gradient | 7 | 0.00165984408 | false |
| native gradient | 7 | 0.00121954854 | false |
| Adam32 | 1 | 1.72762e-9 | false |

Within actualCG, four full endpoints are rejected solely by CVaR; three half steps and one1/16
step are accepted and replayed. Four other parents accept full steps. This is real small
backtracking progress, not the earlier near-ceiling HIER polishing, but misses the unchanged
0.1dB utility floor. Separately executed legacy/actualCG coefficients differ by max0.000512–0.005282
and first-trial RGB by0.000190–0.000512, although all8 full-step decisions/reasons agree.
Do not claim exact shared-direction causal intervention. Accepted COCO9/seed1 and COCO34/seed0
CVaR relative increases1.378e-5 and3.638e-6 lie inside the existing2e-5 numerical slack:
safe does not mean strict improvement of every metric.

Native and streaming gradients select identical7/8 fractions with almost identical gains;
no native-gradient quality advantage is demonstrated. All three non-CG rays reject all six
trials on COCO34/seed0 solely for CVaR. There is no useful-scale or perceptually established win.

Independent audit checked967 artifact hashes,61 source hashes,48 imported payloads,all234
CPU-scored curve points exactly,290 separate NumPy protected vectors,122 exact gate predicates
and raw PSNR within7.1e-15dB. GPU replay covered56 selected endpoints,77 trials,16 control
candidates,29 selected replays and32 signed q images, max RGB error5.9604645e-7.
Eight native VJPs,16 streaming gradients and8 diagonals reproduce with maximum relativeL2
errors7.91e-7/2.29e-6/1.07e-6; alpha relative error<=1.683e-7. All RGB-only selections preserve
geometry/support/opacity;22 rollbacks are bitwise parent fields.

Work:77 ray trials,178 transaction quality/coverage evaluations,482 actual renders including
32 q renders,8 VJPs,8 diagonal constructions,528 basisA/560 AT/32 denominator calls,
256 Adam updates and16 independently executed32-iteration CG solves (512 iterations).
The row field legacy_cg_iterations counts ray solves only (256); it is not the full CG total.
Reporting adds112 quality/render/coverage calls. Native VJP phase2.49–4.40ms and entire native
transaction80.7–177.4ms are descriptive instrumented observations, not a speed claim.
All245 occupancy samples across8 workers have no foreign process/query error; sampling does
not establish continuous exclusivity.

## Independent review and verification

The distinct reviewer approved each exact executable digest before any corresponding formal
outcome, then audited every complete matrix, raw source binding, gate, selected field and work
ledger. Outcome verdict: Accepted for bounded negative findings and task retirement, no required
numerical implementation change and no default promotion. This accepts evidence integrity,
not the disproved utility hypotheses.

Clean source e2bf6ae passed the portable gate with2342 tests;8de8004 passed with2395 tests,
97 skips,515 deselections and all structural checkers. The follow-up also has44 focused method
CPU passes,79 driver/control CPU/integration passes and3 CUDA diagnostics. The final integrated
tree requires the same full verification before commit. The checker-only cache and optional
code-driven archive profile have independent focused reviews/tests (32 and33 respectively).

All three original native HTML reports were browser-opened, plotted at full width, and their
native reconstruction and raw-metric links clicked. PORT shows110 rows/500 images/1916 links,
FIT-05156 rows/357 images/1576 links, with zero broken images. The FIT-051 COCO25/seed0 actualCG
curve was visually checked: full-step MSE improvement coexists with CVaR rejection; half-step
acceptance and subsequent replay/reporting are explicitly distinct actual-evaluation points,
not mislabeled optimizer iterations. Browser receipts are preserved separately from the bundles.

## Archive, reproduction and protected state

### Final integration review and test receipt

Distinct reviewer codex-code-research-reviewer accepted exact staged tree
`048423f25fa1f9ead882698de889d4e4e5cbb2b8`, independently confirming its index identity.
The verdict permits task retirement and this disclosed review/verification metadata append;
no further code or scientific correction remains. Outcomes were accessed during the prior
independent audits, not through new experiments in this integration review. This is a scoped
epistemic review, not a repository-wide Level-2 seal. Historical task paths, partial-archive
omissions, all214 cells, C73–C75 and every numerical/causal qualification were accepted.
Invented decision alternatives were removed; failed primary promotion gates are now explicitly
distinguished from passing subordinate integrity/timing predicates.

Final portable command:
`CUDA_VISIBLE_DEVICES='' STRUCTSPLAT_PYTHON=/home/alex/miniconda3/bin/python ./scripts/verify.sh`.
Result: **2447 passed,97 skipped,515 deselected**,3 pre-existing warnings,439.65seconds;
lint and all5 structural checkers pass. The three new CUDA diagnostics passed separately.
After restoring historical protocol paths during closure, all3 original manifest/source/approval
checks pass and the74 report/cache/archive/QA focused tests pass on the final code in6.81seconds.
All structural checks were rerun after the final wording corrections. Byte-preserved CSV files
retain their original CRLF endings; whitespace review recognizes those endings rather than
normalizing hash-bound evidence. No original artifact is edited to satisfy a style check.

[Partial archive](archive/index.html) contains every root JSON/JSONL/CSV, every nested JSON/JSONL,
and native files named field.npz, initial_field.npz and candidate_field.npz. Exact omitted paths
are listed in its inventory: raw float rasters, trial operands, optimizer states, input duplicates,
most native snapshots, PNGs and original HTML are not copied. No scalar rows or decisions are
selected away. A complete cold replay requires the original bundle, not this portable subset.
The packager is copied/hash-bound and its original overnight default profile is unchanged.

Independent archive QA verifies1438 copied/source hashes,5 generated hashes,183 committed
source-file bindings, all214 JSON/JSONL/CSV/HTML rows and19 contained links across4 pages.
Final archive manifest SHA256:
`8b4bc5e09a9f48bff3225f86874ea723ac15f5e70caeafc6720202bbc5b9a2f2`.
The [browser inventory](browser/inventory.json) binds all3 original browser receipts,6 unchanged
PNGs and the review program; it preserves both the earlier and current archive-QA receipts.
Only derived omission wording changed (`per-event field_*.npz snapshots`); the earlier generated
package is recoverable at `/tmp/structsplat-code-driven-archive-prewording-20260905T1023`.
No numerical artifact or original report changed. The final partial archive was also browser-
opened at1440x1000: all3 tables have the expected48/110/56 rows without page overflow, and
each actual decision link opens parseable JSON. The dedicated temporary browser was closed.
The5 QA-helper tests plus33 archive-profile tests pass. After current task retirement, direct
identity checks of all3 original manifests still pass against their historical task/source paths.

Use the exact clean source and task protocol at that historical revision. Reproduction commands
are in the corresponding task files and original manifests; use fresh output paths. FIT-051
must receive the preserved FIT-050 parent bundle, not newly fitted or outcome-selected parents.
For local full-bundle validation:
`python scripts/check_report_bundle.py /home/alex/Documents/structsplat/results/code-driven-2026-09-05/fit051-v1`
(and the analogous other two directories). Recorded commit source bytes, not current edited
task descriptions, are authoritative. No threshold rescue, selective repeat, held-out image,
push, foreign-process termination or original-result rewrite occurred. Final integration is a
local fast-forward of the reviewed isolated branch after full verification, not a new experiment.
