# COMP-003 PNG Stream Codec Smoke

Purpose: advance COMP-003 rung 4 by adding a self-describing optional PNG-backed stream payload
without claiming a rate-distortion win.

Implementation: `CodecConfig(stream_codec="png")` stores each already-quantized byte stream as a
padded grayscale PNG plane. The `SSPL1` header now records `stream_codec` and
`stream_raw_lengths`; older blobs default to zlib when the fields are absent.

Validation:

```bash
python -m ruff check src/structsplat/codec.py tests/test_codec.py
pytest -q tests/test_codec.py
```

Result: ruff passed; codec tests passed 14/14. Tiny local byte-size smoke on a 256-G toy field:
zlib 2,348 bytes / 4.5859 bpp, PNG 2,756 bytes / 5.3828 bpp at identical 20.3396 PSNR.

Decision: keep this as format infrastructure only. Do not mark rung 4 complete until a pinned
`benchmarks/rate_distortion.py` comparison and true per-attribute planes are evaluated.
