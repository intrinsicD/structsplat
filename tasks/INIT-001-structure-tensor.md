# INIT-001: Structure tensor

**Status: done (reference).** See `structure_tensor.py`, ADR-0004.

## Acceptance criteria
- [x] `J = G_rho * (grad I grad I^T)`; eigen `lam1,lam2`, coherence, energy.
- [x] `across_edge_angle` (gradient dir) and `along_edge_angle` (tangent).
- [x] flat/edge/corner labels (percentile-relative thresholds).
- [x] Validated: correct orientation on a vertical edge; all three labels appear.

## Follow-ups
Scharr/Sobel gradients vs central differences; multi-scale tensor (tie to HIER-001).
