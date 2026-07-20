# Renderer-aware RGB coordinate probe

**Date:** 2026-07-17
**Scope:** isolated exposed-development mechanics; not a source-bound method result
**Cell:** `jason-briscoe-149782/m12_s6_r6_c8`
**Result schema:** `structsplat.onsq-isolated-dev-proof.v1`

## Audited outcome

The independent step-3 SSP2F baseline is 48,755 bytes with unclamped source-teacher SSE
69.45812460707485. Two coordinate variants were independently inspected:

| Variant | Unclamped teacher SSE | Complete SSP2F bytes | Delta vs independent s3 | Headroom vs original Q |
|---|---:|---:|---:|---:|
| Independent `s3` | 69.45812460707485 | 48,755 | 0 | 2,794 |
| Exact-gain-ranked unrestricted coordinate pass | 34.657494870163426 | 48,806 | +51 | 2,743 |
| Entropy-monotone coordinate pass | 47.47566100662405 | 48,652 | -103 | 2,897 |

The ranked unrestricted pass reduces teacher SSE by 50.10% while remaining below the original
strongest exact stream, SSP2E at 51,549 bytes. The entropy-monotone pass reduces teacher SSE by
31.65%, saves 103 bytes relative to independent `s3`, and reduces ideal empirical RGB entropy by
906.2548476840602 bits.

The corresponding SSP2F blob hashes are:

- independent `s3`: `cae95dee96c46a682afea4a8bf73be249b72ca703e1ee43cba63c83091909ae8`
- ranked unrestricted: `487d091a5a436a12b4c2ff498f478289b5bafdbad0f6813c21d5106f0ef488db`
- entropy-monotone: `d124e008a4514a0585dbee6fd8dab18d2adabc3b19a074c75dc52d45669074e2`

The raw result self-seal is
`5bcc5174311bc948dd14167fa58f1c07578cec0b14ac8454a9449ec76ae4f658`; the physical JSON file
SHA-256 at audit time was
`03ad93910c2e1b8f4d7edd2b55f6c2cd2111e9943a31055fd916d9149e58e78b`.

## Claim boundary

This is a strong mechanics signal, not a StructSplat method improvement. It uses one already
exposed development cell, source-teacher rather than original-target distortion, only 4,087 of
8,192 rows as coordinate receivers, and an ad-hoc `/tmp` runner with `sandbox_attested=false`.
No target PSNR/MS-SSIM/LPIPS, multi-image statistics, held-out evidence, exact-Q fixed-point
certificate, or end-to-end speed gain exists. The full probe took 876.696 seconds.

The result justifies a new source-bound exact-complete-byte Q-cap assay with unchanged SSP2F; it
does not justify a default, production, compression, quality, convergence, or expressiveness
claim.

## Raw provenance

- Raw result at audit time:
  `/tmp/structsplat_onsq_dev_probe_20260717/adversarial_coordinate_output/m12_s6_r6_c8/result.json`
- Probe script SHA-256:
  `dea5973f0abeda624d19b9528f62fc1ba928417e9daccf9f66513af1442ffc2c`
- Original candidate cell record SHA-256:
  `01c87dcaf4192feb31c0960524ce4157a09a5504951a24d217f070ae9bfb3eab`
- Confirmation payloads accessed: `false`
- Target/quality files accessed: `false`
