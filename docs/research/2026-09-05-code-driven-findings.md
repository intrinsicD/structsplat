# Code-driven research findings — 2026-09-05

Implemented three bounded studies and completed all **214 cells** with distinct prospective
protocol approval, clean-source execution, full report checks and independent outcome audits.
There are useful component observations and small real quality gains, but **no candidate passes
its frozen promotion/utility gate**. Maintained defaults stay unchanged (ADR-0034; ARA C73–C75).

| Study | Implemented question | Measured outcome | Decision |
|---|---|---|---|
| FIT-050, 48 cells | Cheap normalized RGB directions, Jacobi scaling and safeguarded CG interpolation | 21/24 ray transactions stop at the cross-backend parent-compatibility check before a direction is tested; legacy CG median gain +0.004216 dB | Utility negative; cannot infer that line search or preconditioning generally fails |
| PORT-007, 110 cells | Reuse the CUDA forward denominator and share CVaR/p99 tail work | Median image-level component ratios: coverage 10.248×, tail 1.003×, both 10.401×; null-state gate flips also occur in the legacy repeat | Every primary component/pipeline promotion gate fails; no validated end-to-end speedup |
| FIT-051, 56 cells | Render every trial directly; compare CG fractions, streaming gradient/Jacobi and native color VJP | Actual-CG accepts 8/8, including four smaller-step rescues; median gain +0.005293 dB, below the preset +0.1 dB utility floor | Tiny tolerance-safe progress, not useful-scale or established perceptual improvement |

## What changed in the code

- `src/structsplat/color_ray.py`: opt-in fixed-geometry directions, exact streaming diagonal,
  analytic scalar step, bounded fractions and selected-field replay.
- `src/structsplat/actual_color_ray.py`: a separate actual-render transaction; all trial images
  come from changed fields. The native gradient requests color derivatives through the maintained
  backward, whose entire cost is charged; this is not a new specialized CUDA kernel.
- `safe_schedule.py` / `cuda_render.py`: opt-in same-call coverage and tail reuse with explicit
  unsupported/near-threshold/nonfinite/outside-mask fallbacks. Both defaults remain `reference`.
- Task-scoped drivers, raw-artifact report validators, counter/provenance tests and a partial
  evidence packager make these studies reproducible. The post-run validator cache reuses only
  immutable mask geometry within one check; it caches no image, metric or decision.

## Interpretation and next question

FIT-051 avoids FIT-050's interpolation compatibility obstruction without widening a tolerance.
Within each actual-CG transaction, four full steps are rejected only by CVaR; three half steps
and one 1/16 step are accepted and replayed. Separately executed legacy/actual CG solves are not
bit-identical, so their acceptance rates are not an exact shared-direction causal intervention.
Existing numerical slack can admit tiny CVaR increases. Native and streaming gradients select
the same 7/8 fractions and achieve essentially identical +0.001220 dB medians; Jacobi reaches
+0.001660 dB. None establishes perceptual superiority.

PORT-007 is promising as a component optimization, not ready for a default flip. Its independent
legacy pipeline repeats diverge in all six trajectories; four timing pairs and three quality
pairs fail. Across the 30 instrumented pipelines, 51,892 attempted versus 8,952 accepted steps
make clear why equal final count is not equal work. A next bounded question is whether null-gain
gate sensitivity and rejected-trial work can be reduced without weakening safety. The cause and
a remedy are **unproven**; no successor experiment is silently authorized by this result.

All images are exposed COCO development data, max-side512; the pipeline mask is a fixed synthetic
ellipse. Four-image/two-image aggregates are descriptive, not population confidence statements.
No held-out, full-resolution, production, downstream3D, actual-rate or global novelty claim follows.
Formal point-sampled GPU occupancy found no foreign process or query error, but cannot establish
continuous workstation exclusivity. The earlier unrun HIER projected-GN rescue remains distinct.

## Evidence and reproduction

[Detailed evidence and independent audit](../../ara/evidence/code-driven-method-research-2026-09-05/run.md)
and [portable partial archive](../../ara/evidence/code-driven-method-research-2026-09-05/archive/index.html)
retain every scalar row and decision, source/protocol hashes, histories and selected native fields.
The archive explicitly omits raw images, trial operands, optimizer states and most native snapshots;
it is **not** a complete replay bundle. Full immutable image-bearing reports remain locally at:

- `results/code-driven-2026-09-05/fit050-v1/index.html`
- `results/code-driven-2026-09-05/port007-v1/index.html`
- `results/code-driven-2026-09-05/fit051-v1/index.html`

Use each evidence note's exact source commit and digest when reproducing. All original complete
report checkers pass without allowances; independent CPU/GPU raw replay and native browser checks
also pass. No original result or frozen gate was repaired, overwritten or selectively rerun.
