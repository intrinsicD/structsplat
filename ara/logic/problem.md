# Problem

## Governing goal

Convert calibrated multi-image dome captures, with and without masks, into 2D Gaussian
observation fields, then infer and train a shared 3D Gaussian scene from those fields and
calibration alone. Source images are absent from reconstruction training.

The central research question is: **What must the 2D fields preserve, and how should reconstruction
exploit their spatial support and geometry, to recover a good shared 3D scene efficiently?**

Judge StructSplat improvements by downstream reconstruction quality, convergence, and complete
conversion-plus-training resource cost. Image-fit fidelity, field size, and query cost are
component diagnostics whose value depends on that reconstruction goal. The usefulness of Gaussian
geometry beyond field-query color supervision remains a hypothesis to test.

This is the user's original objective, explicitly reaffirmed on 2026-09-05 (N338/N339;
[session record](../trace/sessions/2026-09-05_001.yaml); [research priority H19](solution/heuristics.md#h19-prioritize-image-free-3d-reconstruction-utility)).
It records the research objective and selection criterion, not a demonstrated method advantage.

The user further specifies tomography as the focus of the current methodological and literature
research (N342): realtime-gs already uses Gaussian beam back-projection and fusion. Center this
work on reconstruction from projection fields, the forward operator, inversion, and projection
consistency. Explicit correspondence tracking is adjacent prior art, not an adopted replacement
for that focus. This clarification does not establish that ordinary RGB fields obey a linear
ray-integral model or select a new implementation or experiment.

## Component-level research questions

Single-image 2D Gaussian representations can fit images compactly, but they usually rely on random
or weakly structured initial placement plus iterative optimization to discover geometry. That makes
low-budget quality, convergence speed, and density-control behavior difficult to reason about.

StructSplat asks whether a deterministic, feature-aware initial field can make the representation
more predictable:

- use one structure tensor for density, orientation, and flat/edge/corner classification;
- place Gaussians with density-adaptive blue-noise sampling rather than clumped random samples;
- treat edge discontinuities explicitly by comparing on-edge and flanked placement;
- keep the renderer/loss/optimizer policy explicit enough that ablations can isolate each stage.

The component-level scientific question is not "can a fitter eventually overfit one image?" It is: under
matched budgets, metrics, renderers, and stopping rules, which placement, scale, renderer, and
densification stages actually improve quality, convergence, or storage?
