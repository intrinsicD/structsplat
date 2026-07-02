# ADR-0004: The structure tensor is the single feature/density/orientation operator

## Context
Feature-aware init needs three things: *where* to put Gaussians (density), *how to orient* them
(anisotropy), and *what kind* of region each is (flat/edge/corner). Naively these are three separate
detectors (e.g. Canny + gradient + Harris) that can disagree.

## Decision
Derive all three from one smoothed structure tensor `J = G_rho * (grad I grad I^T)`:
eigen-energy `lam1+lam2` -> density; eigenvectors -> orientation (elongate along the edge tangent);
eigenvalue pattern -> flat (`lam1≈lam2≈0`) / edge (`lam1≫lam2`) / corner (`lam1≈lam2≫0`).

## Consequences
+ Internally consistent: the same field that says "edge here" says "oriented like this" and "this
  dense". Fewer knobs, fewer contradictions.
+ Reused verbatim on the residual for finer pyramid levels.
- Classification thresholds (`flat_frac`, `corner_frac`) are percentile-relative heuristics; they
  affect flanking vs on-edge assignment and are an ablation/tuning surface (`INIT-004`).
