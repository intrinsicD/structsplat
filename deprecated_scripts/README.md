# Deprecated scripts

These task-specific runners were moved out of `scripts/` when the supported
workflow surface was consolidated.

They remain source-controlled for historical evidence reproduction, but they
are not the recommended entry points and receive no new interface guarantees.
Use:

- `scripts/convert.py`
- `scripts/benchmark.py`
- `scripts/ablation.py`
- `scripts/stage_search.py`

Historical evidence and trace files intentionally retain the command paths that
were executed at the time. For an archived command, replace its old `scripts/`
prefix with `deprecated_scripts/`.

The two native environment provisioners also live here for now:

```bash
deprecated_scripts/setup_native_gaussianimage_env.sh
deprecated_scripts/setup_native_image_gs_env.sh
```
