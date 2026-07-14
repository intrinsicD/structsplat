# BENCH-005: Native external-reference pipelines

## Status
Partial. Isolated native GaussianImage++, Image-GS, and base GaussianImage adapters, official
environments, central metric harnesses, matched COCO4 500/5k proxy slices, same-pixel pairing, the
10k-step native GaussianImage fixed-storage slice, and a separately scoped native AIR inference
run are complete. Native-authentic full-resolution multi-rate curves, actual-codec RD where streams
exist, and the remaining 2026 reference methods remain open.

## Goal
Replace paper-name analogue rows as the sole external comparison with separately reported native
repository executions. Preserve two distinct questions:

1. **Matched axes:** same input, resolution, final Gaussian cap, requested optimizer steps, and
   seed; each repository retains its native renderer, loss, optimizer, and growth semantics.
2. **Native authentic:** the repository's published/default protocol at its intended resolution,
   iteration count, rate definition, and learned-checkpoint assumptions.

Neither track may be labeled “same hyperparameters.” Native rows must remain outside the
StructSplat matched-policy default-promotion gate.

## Required row contract

- Repository root/commit, Python/Torch/CUDA versions, exact compiled-extension path and hash.
- Input/resolution/count/step/seed axes plus requested and achieved Gaussian count.
- Raw float reconstruction and visualization PNG.
- Centrally recomputed PSNR/SSIM/proxy-MS-SSIM/LPIPS, while retaining native-reported metrics.
- Per-step PSNR trajectory, AUC, target hits, synchronized fit wall time, and synchronized render
  latency distribution/FPS.
- Actual codec bytes/bpp only when a real encoded stream exists; parameter-bit estimates and
  checkpoint sizes must use separate names.

## Completed 2026-07-10

- Added `benchmarks/native_reference_compare.py` and the isolated
  `benchmarks/native_runners/gaussianimage_plus.py` adapter.
- Shared benchmark configs now stamp the StructSplat commit, dirty state, tracked-diff SHA-256,
  and untracked file list; the external checkout commit and compiled-extension hash remain separate
  native-row fields. Fair rows additionally key resume on a canonical hash of all scientific axes,
  device/environment versions, input pixels, and critical source files; execution-only shard limits
  do not change that identity.
- Enforced that GaussianImage++ resolves `gsplat.csrc` from its own checkout and records SHA-256,
  preventing the editable-install cross-contamination found in the local reference repos.
- Added central float-reconstruction metrics, trajectory/AUC/target-hit capture, count/shape/
  finite-value validation, subprocess error rows, actual-vs-estimated rate separation,
  synchronized render timing, CSV/JSON/Markdown/HTML artifacts, and CPU tests. Adapter v2 also
  validates every requested axis, repository commit, exact extension path/hash, timing, and full
  per-step history. The hardened resume identity includes clean repo/gsplat tree and Python-source
  fingerprints, adapter/harness/metric revisions, source/target/decoded-pixel hashes, extension
  hash, growth/timing/LPIPS settings, and exact Python/Torch/CUDA/NumPy/metric environment. Cached
  manifests are revalidated, central metrics are recomputed, and stale journal rows are compacted
  before output. Source-specific artifact paths bind both canonical path and content hash, avoiding
  collisions between same-named inputs (including identical-byte copies).
- Native plumbing smoke: one COCO image at max-side 160, cap 64, 2 steps, seed 0 completed with
  61 achieved Gaussians and verified commit/extension provenance.
- Matched-proxy slice: all four pinned COCO images x seeds 0/1 at max-side 160, cap 640, and
  500 requested steps completed (8/8). The paired central evaluator reran the pinned StructSplat
  default with LPIPS on the identical cells and wrote image-clustered confidence intervals.
  GaussianImage++ native is a clear time/quality tradeoff at this short horizon: native-minus-
  StructSplat mean gains are -5.0678 dB PSNR (95% CI [-7.6699, -2.4657]), -0.05142 proxy
  MS-SSIM [-0.06340, -0.03944], -0.1886 LPIPS gain [-0.2175, -0.1466], -7.1638 AUC
  [-10.1297, -4.4784], but +0.4284 s fit-time gain [+0.2704, +0.6549]. Median native render
  throughput is roughly 4.0k-4.35k FPS. GaussianImage++ restores its best training-PSNR
  checkpoint while StructSplat exports its terminal field; the adapter records this asymmetry.
  All eight cells selected step 500/500 in this slice, so it did not change their endpoint here.
  This is matched-axis/time-constrained evidence, not a native-default or global method-ranking
  claim. Artifact:
  `results/native_gaussianimage_plus_matched_proxy/index.html`.
- Added the Image-GS-specific `benchmarks/native_image_gs_compare.py` harness and isolated
  `benchmarks/native_runners/image_gs.py` v2 adapter. Image-GS cannot safely share the base
  environment with other external references because its bundled package is also named `gsplat`.
  The preflight requires a clean official checkout at commit
  `03088368d42684fb54225c981cfd94b58cc0393a`, requires the installed `gsplat` source/build to come
  from that checkout's bundled `gsplat`, and pins `fused-ssim` to release-era commit
  `b4fd8324e81c48c9b2b9f62e1b9c6431fece6ab3`. Repository tree/diff, package direct-URL metadata,
  installed/repository Python-source hashes, both extension hashes, Python/Torch/CUDA/GPU state,
  and the `libstdc++` preload are recorded and checked again in each cell manifest.
- Added `scripts/setup_native_image_gs_env.sh` to construct the official-version environment from
  Image-GS's `environment.yml` while pinning the previously floating `fused-ssim` dependency. The
  completed short run used the already available isolated Python 3.12.9, Torch 2.9.0+cu128, CUDA
  12.8 environment rather than the official Python 3.11.10, Torch 2.4.1, CUDA 12.4 stack. Its
  algorithm and compiled-build provenance are pinned, but it is not an official-environment
  reproduction.
- Exposed four non-interchangeable Image-GS profiles. `matched_steps_fixed_n` disables progressive
  allocation, uses float32/constant LR, and starts at full N for an arbitrary requested step
  horizon. `siggraph25` is the paper-aligned 5k-step constant-LR, 16-bit analytical-payload
  algorithm profile with native progressive allocation. `release_quickstart` is the current 10k
  release behavior plus quantization, LR decay/early stop, and progressive allocation.
  `release_default_float` is the corresponding bare-config float32 profile. The latter three are
  not automatically native-authentic/full-resolution claims; the requested resolution/count must
  also match the intended protocol.
- The Image-GS harness keeps upstream-reported values separate and recomputes PSNR, SSIM,
  small-image proxy MS-SSIM, and LPIPS centrally from the terminal float reconstruction. The
  upstream five-scale MS-SSIM is not reported as if it worked at max-side 160. AUC and target hits
  use sparse native evaluation checkpoints and remain diagnostic across implementations because
  clamping/cadence semantics differ. Image-GS provides an analytical attribute-bit formula, not a
  packed codec stream; rows name it `analytical_payload_bytes`/`analytical_bpp`, while
  `actual_codec_bytes`/`actual_bpp` remain blank.
- Hardened Image-GS resume and pairing. The cache key covers source and resized-target hashes,
  every requested axis, repository/build/environment provenance, adapter/harness/metric source
  hashes, timing settings, device, and LPIPS state. Cached manifests and reconstruction hashes are
  revalidated before reuse. Paired Image-GS/StructSplat rows additionally require identical
  run-recorded decoded-pixel hashes; progressive profiles require matching recorded start counts,
  while the fixed-N profile explicitly preserves and reports its intended start-count mismatch.
- Completed the Image-GS fixed-N proxy slice: COCO4 x seeds 0/1 at max-side 160, cap 640, 500
  requested steps (8/8 successful). Native-minus-StructSplat mean gains, with positive always
  better, are -3.6011 dB PSNR (95% CI [-4.3059, -2.7527]), -0.01879 proxy MS-SSIM
  [-0.02937, -0.00822], -0.1842 LPIPS [-0.2658, -0.1135], and diagnostic AUC -2.6909
  [-3.2317, -1.9494]. The final-quality familywise test supports StructSplat on this slice. Timing
  is inconclusive and diagnostic: fit-time gain is +0.0487 s [-0.0020, +0.0889], Image-GS's wall
  timer includes terminal image/checkpoint I/O, and StructSplat's does not. This also is not a
  strict implementation-dominance result because Image-GS starts with all 640 Gaussians while the
  pinned StructSplat row starts at half N and grows. Artifact:
  `results/native_image_gs_matched_fixedn_proxy/index.html`.
- Completed the small-image `siggraph25` algorithm-profile lane at the same four decoded targets,
  cap 640/start 320, and 5,000 requested steps for seed 0. A fresh 5,000-step StructSplat default
  supplied hash-verified pairs. Native-minus-StructSplat gains were -0.3840 dB PSNR (95% CI
  [-2.3698, +1.1997]), +0.01608 proxy MS-SSIM [+0.00074, +0.03142], -0.0443 LPIPS
  [-0.0652, -0.0243], and diagnostic AUC -0.5929 [-1.7871, +0.5845]. This is a heterogeneous
  tradeoff and only one seed; it is not an official-environment/full-resolution or rate-distortion
  result. Artifact: `results/native_image_gs_siggraph25_proxy_seed0/index.html`.
- Provisioned and verified the official Image-GS Python 3.11.10 / Torch 2.4.1 / CUDA 12.4
  environment under `results/native_envs/image_gs_official/`. The setup constrains MKL 2023.1.0
  (avoiding the Torch `iJIT_NotifyEvent` loader failure) and `cuda-version=12.4`, builds pinned
  fused-SSIM and bundled gsplat, and records exact Conda/pip/binary/linkage hashes. The official
  500-step fixed-N rerun (COCO4 x seeds 0/1) supports StructSplat final quality: Image-GS-minus-
  terminal-default gains are -3.6639 dB PSNR, -0.01907 proxy MS-SSIM, -0.1773 LPIPS, and
  -2.7060 diagnostic AUC. Timing remains non-strict because accounting differs. Artifact:
  `results/native_image_gs_fixedn_500_official_two_seed/`.
- Repeated the `siggraph25` 5k lane in that official environment and paired it separately against
  the terminal default and `structsplat_best_checkpoint`. Against the terminal default, Image-GS
  gains +0.2201 dB PSNR, +0.01959 proxy MS-SSIM, and -0.0369 LPIPS. Against the checkpoint row,
  it gains -0.3601 dB PSNR, +0.01038 proxy MS-SSIM, and -0.0566 LPIPS. Both are tradeoffs with
  confidence intervals crossing on PSNR; artifact:
  `results/native_image_gs_siggraph25_official_seed0/`.
- Added `benchmarks/native_gaussianimage_compare.py`,
  `benchmarks/native_runners/gaussianimage.py`, `scripts/setup_native_gaussianimage_env.sh`, and
  focused tests for base GaussianImage. The official isolated checkout pins GaussianImage
  `d53393bee7c9fbb24e3510614e3ff2c85b8fbbc1`, gsplat
  `bcca3ecae966a052e3bf8dd1ff9910cf7b8f851d`, Python 3.10, Torch 2.0.0+cu118, and a retained
  wheel/build hash. The runner preserves native fixed-N Cholesky/RS, L2, Adan, 20k LR steps, and
  terminal selection; the harness validates source/build/environment/target/checkpoint provenance
  and centrally scores float pixels. Its resume key includes the shared comparison implementation;
  cached manifests are revalidated and rescored, and only current requested keys survive journal
  compaction.
- Completed native GaussianImage COCO4 evidence. At 500 steps x seeds 0/1, it is ~0.28 s faster
  than the terminal default but loses 13.7463 dB PSNR, 0.25929 MS-SSIM, 0.5037 LPIPS, and
  14.6578 AUC, demonstrating that the short proxy is far from its native convergence horizon.
  At 5k/seed0 versus the checkpoint candidate, GaussianImage is ~6.44 s faster and +0.01298
  MS-SSIM, while StructSplat is +0.1207 dB PSNR, +0.0253 LPIPS gain, and +1.5337 AUC. The 5k
  result is a tradeoff. Artifacts: `results/native_gaussianimage_matched_500_official_two_seed/`
  and `results/native_gaussianimage_matched_5000_official_seed0/`.
- Audited the released GaussianImage Kodak/RD path. The current adapter is representation-only:
  `release_cholesky`/`release_rs` do not enforce native Kodak orientation (768x512 or 512x768),
  the released count set, ordered seed-1 process semantics, or the second 50k QAT trajectory.
  Released Kodak uses N={800,1000,3000,5000,7000,9000}, 50k representation + 50k QAT steps, and
  best-training-PSNR QAT selection. Its `compress_wo_ec` output is not a self-contained stream;
  the corrected fixed-width analytical rate is `56*N+1728` bits and actual codec bytes/bpp must
  stay null. The
  smallest faithful next slice is Cholesky-only `kodim01`, N=800, seed1, native resolution,
  including cold checkpoint reload and in-memory quantized-decode equality before expanding to
  Kodak24 x six counts.

## Completed 2026-07-13

- Completed the available-repository fixed-storage suite and documented it in
  `ara/evidence/bench001-external-complete-2026-07-13/run.md`. The native official
  GaussianImage lane completed 8/8 COCO4 × seed cells at max-side 160, N=5,376, and 10,000
  requested steps. It averaged 35.6571 dB in 6.392 s and about 4,412 render FPS. Against the
  historical pinned StructSplat row it is a -13.142 dB / +8.288 s quality-speed tradeoff. Its
  78.591 bpp value is float-parameter accounting, not an encoded stream.
- Ran native AIR inference separately on the same four source images at max-side 256 because AIR's
  MS-SSIM path rejects the 160-side lane. It completed 4/4 images, averaging 25.254 dB,
  37.007 ms inference, 3,511.25 Gaussians, and a native-reported quantized 4.328 bpp. Resolution,
  learned-checkpoint, metric, and rate semantics differ, so this is environment evidence rather
  than a paired ranking.
- The common-harness 320-cell report remains a local-mechanism study. It must not be used as the
  native-reference table merely because some rows have paper-derived labels.

## Next actions

1. Treat BENCH-007's completed negative Stage-1 actual-rate result as a claim boundary, not a native
   leaderboard target. Native methods must still be evaluated at their real rate definitions, not
   forced into the 168 KiB proxy; this lane cannot retroactively promote tensor-WSE.
2. Add native Structure-Guided Allocation first if official code is available; it is the direct
   handcrafted structure baseline. Then prioritize SAD and WIPES as representation controls.
3. Expand GaussianImage++ across multiple native-resolution rates and an iteration/time envelope;
   include its real quantized stream path before actual-bpp claims.
4. Expand Image-GS release quick-start and native-authentic/full-resolution multi-rate/time-matched
   tracks. Add a real packed-stream path before codec-bpp claims; never reinterpret
   `analytical_bpp` as actual rate.
5. Add `release_kodak_cholesky_qat_woec`: enforce native resolution/counts/order, run 50k
   representation + 50k QAT, preserve upstream best-QAT selection, validate cold/in-memory decode,
   and report both upstream and corrected `56*N+1728` analytical rates. Keep actual bytes/bpp null;
   no released self-contained bitstream exists.
6. Harden the AIR adapter as a BENCH-005 native row with central original-pixel rate accounting and
   checkpoint/build provenance; retain the max-side mismatch in every report.
7. Provision Instant-GI's `torch_kdtree`, native extensions, and learned checkpoint; report its
   adaptive N rather than truncating it into a fixed-N claim.

## Interfaces

`benchmarks/native_reference_compare.py`, `benchmarks/native_image_gs_compare.py`,
`benchmarks/native_gaussianimage_compare.py`, `benchmarks/native_runners/`,
`scripts/setup_native_image_gs_env.sh`, `scripts/setup_native_gaussianimage_env.sh`,
`tests/test_native_reference_compare.py`, `tests/test_native_image_gs_compare.py`,
`tests/test_native_gaussianimage_compare.py`,
`benchmarks/README.md`.

## Depends on

BENCH-001, BENCH-002, BENCH-003, ABL-004.
