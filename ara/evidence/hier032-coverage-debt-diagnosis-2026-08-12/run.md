# HIER-032 frozen-input coverage-debt diagnosis

## Scope

This is a read-only derivation from HIER-031's immutable selected exact-N7,000 field. It freezes the
HIER-032 development question; it is not a HIER-032 method result, held-out evidence, or a default
claim.

Input bundle:
`results/hier031_janelle_c0001_s1200_exact7k_boundary_detail_s0_diagnostic_2026-08-12`.

- field SHA-256: `a0a080ccbd255ce51f11489cd504956a1c5181a495bbca2b4bf74ecb0995c1db`
- analysis SHA-256: `8c01298c307b1311041f68a8c007441c58f8be5901a9bec9354cae1f567921a4`
- row SHA-256: `a75c1cfc22f82e1f85db35df61a53552346dee7b6f5a72d29deb149449c17108`
- evaluation raster: exposed C0001, 1200x1038, seed 0, RTX 3050

## Reproduction

```bash
python - <<'PY'
import numpy as np

path = (
    "results/hier031_janelle_c0001_s1200_exact7k_boundary_detail_s0_diagnostic_"
    "2026-08-12/artifacts/deep_only_terminal_closure_n7000/analysis.npz"
)
with np.load(path) as data:
    coverage = data["unit_coverage"]
    mask = data["mask"]
    error = data["error_raw"].astype(np.float64)
    x0, y0, x1, y1 = map(int, data["hair_crop_bounds"])
weak = mask & (coverage < 0.05)
hair = np.zeros_like(mask)
hair[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
sse = np.square(error).sum(axis=2)
print(int(weak.sum()))
print(int((weak & hair).sum()))
print(float(sse[weak].sum() / sse[mask].sum()))
print(float(np.maximum(0.0, 0.05 - coverage[mask]).sum()))
PY
```

The deterministic HIER-032 8-connected detector reports 483 components with largest size 11. The
frozen candidate bank has 921 candidates and 949 sparse incidence edges; its greedy priority
selects 732 certified placements before donor recertification and later waves.

## Result

- weak foreground pixels `<0.05`: `743`
- deterministic 8-connected components: `483`
- weak pixels in fixed hair crop `(54,434,839,530)`: `461`
- foreground SSE at weak pixels: `38.81602283448657%`
- total coverage deficit mass: `22.806974`
- initial certified set-cover placements: `732` (`719` fallbacks, `13` inward-offset tangent rows)

The weak set is therefore dominated by isolated capacity demands. The set cover provides only
modest initial compression and must be evaluated with exact donor funding and coverage rerenders;
these facts do not imply that any HIER-032 arm will pass the regional quality guards.
