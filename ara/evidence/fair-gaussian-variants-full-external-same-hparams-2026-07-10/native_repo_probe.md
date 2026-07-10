# Native External Repo Probe

This benchmark keeps the existing fair-harness contract: GaussianImage, GaussianImage++, and
Image-GS rows are matched-policy analogue rows under StructSplat's fitter/renderer. Native external
pipelines are not mixed into the promotion gate because they use different renderers, losses,
metric code, progressive schedules, and checkpoint assumptions.

## Local Repos Checked

- `/home/alex/Documents/GaussianImage` at `d53393b`
  - Native smoke was not usable in this environment: its `gsplat/` submodule directory is empty,
    so imports fall through to another installed/local `gsplat` with an incompatible rasterizer
    signature. With the project `libstdc++` preload workaround applied, the smoke failed with
    `AttributeError: 'int' object has no attribute 'contiguous'`.
- `/home/alex/Documents/GaussianImage_plus` at `549cfaa`
  - Native smoke imports and starts, but its trainer-side `pytorch_msssim.ms_ssim` path asserts
    the image must be larger than 160 pixels. That is incompatible with the exact linked
    same-hyperparameter proxy fixture (`max_side=160`) without patching external metric behavior.
- `/home/alex/Documents/image-gs` at `0308836`
  - Native import is blocked by missing `fused_ssim`.
- `/home/alex/Documents/Instant-GI`
  - Used through the existing benchmark hook:
    `STRUCTSPLAT_INSTANT_GI=/home/alex/Documents/Instant-GI/quard_image.py`.
  - Completed 8/8 same-hyperparameter Instant-GI quadtree rows successfully.

## Decision

The committed comparison uses the local Instant-GI hook plus matched-policy analogue rows for
GaussianImage, GaussianImage++, and Image-GS. This keeps default-promotion decisions apples-to-apples
with the linked fair-density artifact and avoids silently mixing native pipeline semantics into a
matched-policy benchmark.
