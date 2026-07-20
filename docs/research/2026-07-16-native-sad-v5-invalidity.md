# BENCH-016 v5: saved-state replay nondeterminism

## Decision

BENCH-016 protocol v5 is **invalid / no decision**. It halted during native
`0057 / 0.5 bpp / repeat 1`, before any StructSplat control or scientific decision gate, because
prepared saved-TXT replay sample 0 did not have the same decoded pixel hash as that row's
canonical saved-TXT replay. One earlier native row completed successfully. Neither row is a SAD
pass/fail result, and neither may be imported into a repaired protocol.

The sealed artifact is
`results/bench016_native_sad_frontier_v5_2026-07-16`. Its binding SHA-256 is
`a87aeed30594bf12f7fe711df332ef794c13e589c38fe7ff103fa90fe8b1b6f3`, its target-manifest
SHA-256 is `a74b03d618eae7de0c0d0e3c5d14387169619337a827cd88426d6318bcec633f`, its invalidity-record
SHA-256 is `333bd6eeb13bc4931e46808f3dbb10e4fa63ddeafe03481378002b07abce39b4`, and its 165-file,
31,140,605-byte invalidity-bundle manifest SHA-256 is
`8eac6e6210de179c1a39251051f38102238d9a5634489a160709b106a5e664c7`.

## What failed

Both fresh processes loaded the same 1,530-site TXT and rendered the same `768 x 510` shape. The
canonical and timing images differed at two pixels and two channel codes: only green changed, by
one uint8 code. Their image-to-image difference PSNR was `105.821030 dB`; the central PSNR delta
was about `+0.0000038 dB`, and AlexNet LPIPS was unchanged at stored precision. These magnitudes
are diagnostic only. V5 required exact decoded-pixel equality, so any nonzero difference is an
integrity failure.

No numeric tolerance may be derived from the observed two-code discrepancy. That would retune a
failed integrity rule after target access.

## Mechanism

The failure is explained by the pinned upstream CUDA source. `jfaSeedKernel` launches one thread
per site, maps each site to an integer home pixel, reads that pixel's first candidate slot, writes
its own site ID, and stores the four-slot word back without an atomic operation or deterministic
tie-break. Repeat 1 has 1,505 unique home pixels for 1,530 sites: 25 pixels each receive two sites.
The winning write can therefore change across fresh processes and propagate through jump
flooding/candidate refinement.

This also corrects the interpretation of the pre-v5 plumbing calibration. Exact 10/10 and 20/20
COCO saved-state replays did not establish a density-bounded deterministic regime. Collision
geometry, scheduling, and whether a seed difference survives later candidate passes matter; site
count alone is not sufficient.

## Repair boundary

A repair must be frozen and independently audited under a new protocol. It must retain the same
images, rates, methods, horizons, repeats, quality/performance gates, and first-replay quality
selection, disclose both v4 and v5 access, start a fresh 144-row artifact, and avoid importing any
old scientific row. It may remove the impossible equality premise by treating variation among
official saved-state replays as a preserved, threshold-free decoder diagnostic. It may not use the
observed difference magnitude to introduce a tolerance, select a favorable replay, average an
outcome-responsive ensemble, or silently replace the published CUDA decoder.

Compression remains untested: SAD's TXT plus a pinned decoder/config is recipient-replayable only,
not a self-contained or compressed representation.
