# BENCH-005 official Image-GS and GaussianImage evidence

Date: 2026-07-10

## Provenance

- Image-GS official commit `03088368d42684fb54225c981cfd94b58cc0393a`, Python 3.11.10,
  Torch 2.4.1, CUDA 12.4, fused-SSIM
  `b4fd8324e81c48c9b2b9f62e1b9c6431fece6ab3`, bundled gsplat. Exact environment exports and
  extension hashes are under `results/native_envs/image_gs_official/`.
- GaussianImage official commit `d53393bee7c9fbb24e3510614e3ff2c85b8fbbc1`, gsplat
  `bcca3ecae966a052e3bf8dd1ff9910cf7b8f851d`, Python 3.10, Torch 2.0.0+cu118. Exact wheel,
  dependency, linkage, source, and binary hashes are under
  `results/native_envs/gaussianimage_official/provenance/`.
- Every benchmark exports float pixels and centrally recomputes PSNR/SSIM/proxy-MS-SSIM/LPIPS.
  Paired rows require identical run-recorded decoded-pixel hashes. Analytical payload, actual
  codec bytes, trajectory cadence, and non-comparable timing protocols remain separate.

## Image-GS

- Fixed-N 500 steps, COCO4 x seeds 0/1, max-side 160, N=640: versus terminal StructSplat,
  Image-GS gains -3.6639 dB PSNR, -0.01907 proxy MS-SSIM, -0.1773 LPIPS, and -2.7060 diagnostic
  AUC. Familywise final-quality bounds support StructSplat on this bounded ablation; strict
  implementation dominance is not tested because allocation/timing/trajectory semantics differ.
- SIGGRAPH25 5k profile, COCO4 seed0, cap640/start320: versus terminal StructSplat, Image-GS gains
  +0.2201 dB PSNR, +0.01959 proxy MS-SSIM, and -0.0369 LPIPS. Versus the full-count-checkpoint
  candidate it gains -0.3601 dB PSNR, +0.01038 proxy MS-SSIM, and -0.0566 LPIPS. Both comparisons
  are tradeoffs/inconclusive rather than dominance.

## GaussianImage

- Fixed-N 500 steps, COCO4 x seeds 0/1: GaussianImage is ~0.28 s faster than terminal StructSplat
  but loses 13.7463 dB PSNR, 0.25929 proxy MS-SSIM, 0.5037 LPIPS, and 14.6578 AUC. Its native
  optimizer is far from converged at the review proxy horizon.
- Fixed-N 5k, COCO4 seed0: versus the StructSplat checkpoint candidate, GaussianImage is ~6.44 s
  faster and +0.01298 proxy MS-SSIM, while StructSplat is +0.1207 dB PSNR, +0.0253 LPIPS gain,
  and +1.5337 AUC. This is a speed/MS-vs-PSNR/perceptual/convergence tradeoff.

## Files

Prefixed config, paired-summary, and human-readable summary files for Image-GS 500/5k and
GaussianImage 500/5k are colocated in this evidence directory. Full cell manifests,
reconstructions, checkpoints, and histories remain under their `results/` artifacts.
