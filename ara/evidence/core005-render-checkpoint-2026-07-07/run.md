# CORE-005 render checkpointing

Date: 2026-07-07

Goal: finish the remaining CORE-005 acceptance item: opt-in gradient checkpointing for reference
renderer slices so backward does not retain every per-slice flat-tile intermediate.

## Implementation

- Added `checkpoint_chunks` to the Python reference renderers.
- Added `FitConfig.render_checkpoint` and `structsplat fit --render-checkpoint`.
- Wired the flag through single-stage fitting and pyramid residual/prefix renders.
- Default behavior is unchanged. The non-checkpoint path still uses the direct per-slice
  `index_add` accumulation.
- CUDA extension renderers are unchanged; this flag only affects `normalized` / `additive`
  reference rendering and CUDA modes that intentionally fall back to the reference path.

## Memory Smoke

One CUDA reference-render forward/backward on the same random field:

- 256x256 target, 3000 Gaussians, scales in `[3, 8]` px, `chunk=64`.
- Loss: square loss to a zero target.
- Metric: `torch.cuda.max_memory_allocated()` after resetting peak stats.

| render checkpoint | start MB | peak MB | delta MB | loss |
|---|---:|---:|---:|---:|
| false | 0.84 | 204.40 | 203.55 | 0.261176 |
| true | 0.84 | 30.50 | 29.65 | 0.261176 |

The checkpointed path cut peak allocated memory for this reference-render backward by about 85%
with identical loss. It trades memory for recomputation, so it remains opt-in.

## Verification

Focused tests:

```bash
python -m pytest \
  tests/test_render.py::test_checkpointed_reference_render_matches_pixels_and_gradients \
  tests/test_fit_dynamics.py::test_fit_runs_with_reference_render_checkpointing \
  tests/test_cli.py::test_fit_cli_accepts_qat_rate_flags -q
```

Result: 3 passed.

Artifact: `memory_smoke.json`.
