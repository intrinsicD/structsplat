# CORE-016 codec-native dual-plane exposed diagnostic

This note binds a default-off implementation/killing diagnostic, not a scientific confirmation or
claim. The full result bundle remains under the git-ignored path
`results/core016_prefiltered_sigma045_steps8_selected_janelle_native_2026-08-06/`; its custom
manifest was independently replayed for all 25 hashes and sizes.

## Scope

- Exposed development image: Janelle frame 00008, C0001 crop selected by its existing mask.
- Complete v2 `.sgdp` packet: 3,896,344 bytes.
- Appearance: lossless WebP effort 75, sigma 0.45, radius 3, eight Jacobi prefilter steps.
- Structure: 512 deterministic anisotropic Field V2 rows with independent nonnegative mass.
- No optimizer steps; no maintained StructSplat or realtime-gs defaults changed.

## Diagnostic outcome

The selected packet reproduces decoded crop pixel centers below display quantization and passes the
task's narrow source-size, contextual-speed, and realtime-gs query-compatibility killing checks. Its
source ratio (`3.662x`) is not a matched crop comparison; canonical crop PNG ratio is `1.139x`.
The component-summed encode estimate is 3.286 seconds and cold decode is 1.316 seconds. The paired
backend matches NumPy colors to `1.19e-7`, structural weights to `1.43e-6`, and drives the existing
synthetic two-view realtime-gs CompactCarve initializer to an exact deterministic four-row 3D
field.

The continuous guard remains unresolved. Against a bilinear decoded-raster control, off-grid PSNR
is 49.375 dB; 3.784% of sampled channels escape their local 2x2 envelope and 0.0244% escape global
`[0,1]`. This is not continuous-scene truth. Structural sampled coverage is not evidence that 512
rows suffice for real multiview lifting.

## Validation boundary

`scripts/check_report_bundle.py` rejects this task-local custom schema, so no maintained portable
report-gate claim is made. Packet/member accounting, cold deterministic resave, the custom manifest,
focused tests, and visual pixel-center outputs were checked independently. Full interpretations and
corrections are in
`docs/research/2026-08-06-codec-native-dual-plane-results-audit.md`.

## Commands

```bash
PYTHONPATH=/home/alex/Documents/structsplat/src:/home/alex/Documents/realtime-gs/src \
  pytest -q tests/test_codec_native_field.py

./scripts/verify.sh
```

The exact driver invocation and source/config hashes are retained in the ignored bundle's
`config.json` and `manifest.json`. No claim is promoted from this note.
