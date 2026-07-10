# BENCH-005 GaussianImage release/RD audit

Date: 2026-07-10

## Scope and source identity

Read-only audit of official GaussianImage commit
`d53393bee7c9fbb24e3510614e3ff2c85b8fbbc1` in the isolated BENCH-005 checkout. Relevant source
hashes:

- `gaussianimage_cholesky.py`: `e475ac9acaa97a83e5779e9a0e7a8fe8a919a22b712b70821525f3924cf8929d`
- `train.py`: `81b6b621916a95f884262ce5c6e973c0e945058b3528c3a7890bb1cd3a60d3e1`
- `train_quantize.py`: `b4a4370ecb268a1851d42dfacb64e0d5ac06f563c6b7db55ee7c88d2712a7460`
- `test_quantize.py`: `7e97e6775b9595537529894bdaf34cca700e5bf0b0126e263f0f78b7ed4bfd8e`
- Cholesky Kodak representation/QAT scripts: `598622c0395258f7d2c66ce1d3075e207650408820ab468ea6f72f1606ad377f`
  and `a4701dfeb1512837dd619db88fe09b39888e72858a958392a2e5d6bc575a07ca`.

## Findings

- The released Kodak scripts iterate N={800,1000,3000,5000,7000,9000} in one process invocation
  per count, with default seed 1. Images retain native 768x512 or 512x768 orientation.
- Representation fitting runs 50,000 steps. QAT initializes from that checkpoint, runs a second
  50,000-step trajectory, and saves/restores the best training-PSNR state for best evaluation.
- Cholesky QAT stores positions as two FP16 values per Gaussian, three 6-bit covariance symbols,
  and two 3-bit RGB residual-VQ indices. Decoder metadata is two 8x3 FP32 RGB codebooks plus three
  FP32 scales and three FP32 offsets.
- Therefore the fixed-width no-entropy rate is
  `32N + 18N + 6N + (2*8*3*32 + 2*3*32) = 56N + 1728` bits. At N=800 this is 46,528 bits,
  5,816 ideal bytes, or 0.118326823 bpp over 393,216 Kodak pixels.
- Upstream `analysis_wo_ec()` derives color-index width from the observed maximum index and can
  undercount unused codewords. `compress_wo_ec()` returns an in-memory tensor dictionary but omits
  the quantizer state needed by `decompress_wo_ec()`, so it is not a self-contained serialized
  stream. `actual_codec_bytes` and `actual_bpp` must remain null for this lane.

## Benchmark consequence

The existing `release_cholesky`/`release_rs` adapter labels select representation covariance forms;
they do not enforce the released Kodak/QAT protocol. The smallest faithful next profile is
`release_kodak_cholesky_qat_woec` on `kodim01`, N=800, seed1, native resolution, 50k+50k steps,
with representation and QAT trajectories separated, cold checkpoint reload checked, in-memory
quantized decode matched, central metrics recomputed, corrected/upstream analytical rates both
reported, and actual byte fields left null.
