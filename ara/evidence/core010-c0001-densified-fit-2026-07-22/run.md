# CORE-010 C0001 mask-contained densified fit — 2026-07-22

## Scope

User-requested single-image, single-seed procedural fit. This is not a benchmark, held-out
comparison, default promotion, compression result, speedup claim, or novelty claim.

## Input and execution

- Original RGB: /home/alex/Documents/datasets/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg — SHA-256
  ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b, 5328x4608.
- Original mask: /home/alex/Documents/datasets/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png — SHA-256
  94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3.
- Fit window: native pixels [258, 1948, 3707, 2883], size
  3449x935; no resampling.
  Full renders paste this crop into the original 5328x4608 canvas.
- Initialization: quadtree_wse, seed 0, 5,000 mask-contained Gaussians.
- Fit: exact CUDA normalized renderer, loss_weighting=mask, mask_contain=true,
  mask_margin=1.5, support_fade=true, L1 + 0.3 SSIM.
- Densification: residual_tensor_add_nms, +3,000 at each of steps
  1,000/2,000/3,000/4,000/5,000, capped at 20,000. The fit target is matted outside the mask so
  sampled-add raw-residual birth scoring is mask-scoped.
- Convergence: max 100,000; eligibility after 15,000; log cadence 50; stop after 200 logged
  evaluations without a >=0.0001 dB new best; select best_psnr_final_count.
- Device: NVIDIA GeForce RTX 3050; Torch 2.9.0+cu128; Torch CUDA
  12.8.
- Executed renderer ELF SHA-256: 6d30d680cfc6526b61b9d02e63ff495714538984e0f952fd41e230afafb33ee4; the exact binary
  is archived at runs/live_viewer_frame_00008_C0001_mask_contained_densified_converged_20260722/artifacts/structsplat_render_ext.so.

## Result

The field reached 20,000 Gaussians at step 5,000 and ran 27,901 iterations.
The early-stop replay finds the last eligible logged best at zero-based iteration 17,900 and exactly
200 stale evaluations through zero-based iteration 27,900, satisfying the 10,000-step window.
The retained checkpoint is step 5,051:

- matted fit-crop PSNR: 25.227867126 dB;
- inside-mask display PSNR: 22.518759136 dB;
- outside-mask display maximum: 0.0;
- terminal trajectory PSNR before checkpoint restoration: 24.917289734 dB;
- fit wall time including snapshot I/O: 3004.149 s.

| Step | Gaussians | Inside PSNR (dB) | Outside max |
|---:|---:|---:|---:|
| 1 | 5,000 | 18.653882 | 0.0 |
| 1,000 | 8,000 | 20.467577 | 0.0 |
| 5,000 | 20,000 | 22.318249 | 0.0 |
| 10,000 | 20,000 | 22.308650 | 0.0 |
| 15,000 | 20,000 | 22.318896 | 0.0 |
| 20,000 | 20,000 | 22.279299 | 0.0 |
| 25,000 | 20,000 | 22.175027 | 0.0 |

## Audit

The audit at runs/live_viewer_frame_00008_C0001_mask_contained_densified_converged_20260722/artifacts/audit.json records pass_with_scope_limits
(SHA-256 fcb400bd164e455209e32ef689260183c0db55f499dee92071b5c3543ff2e6dc):

- 52 bound files checked with zero hash mismatch;
- every expected render through termination is present at 5,328x4,608;
- all rendered float arrays and saved field arrays are finite;
- every full PNG has outside-mask maximum 0;
- cold exact-CUDA replay from the saved selected field, with no mask and scale_max removed, has
  maximum absolute difference 2.980232238769531e-07 from the saved float render and remains
  exactly 0.0 outside;
- the archived binary passes the relevant normalized-oracle forward/backward fixture
  (image max abs 1.7881393432617188e-07);
- float32 log/exp scale-cap roundoff is at most 3.5762786865234375e-07 relative, and the
  convenience full-coordinate file adds at most 0.0001220703125 px translation roundoff.

CUDA atomic accumulation is tolerance-reproducible, not bit-exact. The binary is preserved and
hash-bound, but a clean rebuild from the current CUDA source was not established.

## Artifacts

- Manifest: runs/live_viewer_frame_00008_C0001_mask_contained_densified_converged_20260722/artifacts/manifest.json
- Audit: runs/live_viewer_frame_00008_C0001_mask_contained_densified_converged_20260722/artifacts/audit.json
- Full final render: runs/live_viewer_frame_00008_C0001_mask_contained_densified_converged_20260722/artifacts/final_render_full.png
- Native crop final render: runs/live_viewer_frame_00008_C0001_mask_contained_densified_converged_20260722/artifacts/final_render_crop.png
- Render montage: runs/live_viewer_frame_00008_C0001_mask_contained_densified_converged_20260722/artifacts/rendered_updates_montage.png
- Final fields: runs/live_viewer_frame_00008_C0001_mask_contained_densified_converged_20260722/artifacts/final_field_crop_coords.npz and
  runs/live_viewer_frame_00008_C0001_mask_contained_densified_converged_20260722/artifacts/final_field_full_coords.npz

## Exact rerun command

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python3 runs/live_viewer_frame_00008_C0001_mask_contained_densified_converged_20260722/run_mask_contained_densified.py --extension runs/live_viewer_frame_00008_C0001_mask_contained_densified_converged_20260722/artifacts/structsplat_render_ext.so --outdir runs/live_viewer_frame_00008_C0001_mask_contained_densified_converged_20260722/rerun_artifacts
