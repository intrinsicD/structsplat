# INIT-002: Density field

**Status: done (reference).** See `density.py`.

## Acceptance criteria
- [x] `density_from_energy` → normalized pmf with a floor (`density_base`) and `density_power`.
- [x] `density_from_residual` for pyramid levels.
- [x] `sample_candidates` draws sub-pixel positions ∝ density.
- [x] Validated: mass concentrates ~4× near an edge.
