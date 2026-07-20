# Exact normalized CUDA block-reduction experiment

- Work scope: `synthetic_microfit_cuda_event_device_timeline_only`
- Device: NVIDIA GeForce RTX 3050
- Timing: current-stream torch.cuda.Event(enable_timing=True), synchronized per sample, median over repeats after warmup
- Nsight Compute: `unavailable_gpu_performance_counter_permission_denied`
- Source/device-bound microprofile speedup claim authorized: **no**
- Default flip authorized: **no**

## Preregistered decision

**keep_benchmark_only_optimization_gate_failed** — pass=False.
Representative reductions: exact backward 51.28%, full step 24.76%, peak allocator-memory regression 0.00%.

A pass supports only this source/device-bound microprofile comparison. The candidate remains opt-in and no default, end-to-end fit, quality, or cross-GPU claim follows.

## Measured rows

| H | N | Requested overlap | Actual overlap | Mode | Forward ms | Exact backward ms | Fit backward ms | Adam ms | Full step ms | Backward CV | Step CV | Peak +MiB |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 512 | 4.0 | 4.67 | cuda | 0.395 | 0.519 | 0.895 | 0.421 | 2.075 | 2.49% | 1.07% | 3.7 |
| 128 | 512 | 4.0 | 4.67 | cuda_block_reduce | 0.392 | 0.512 | 0.873 | 0.419 | 2.137 | 2.10% | 5.78% | 3.7 |
| 128 | 512 | 16.0 | 16.19 | cuda_block_reduce | 0.404 | 0.507 | 0.868 | 0.423 | 2.077 | 3.33% | 2.15% | 3.7 |
| 128 | 512 | 16.0 | 16.19 | cuda | 0.400 | 0.544 | 0.948 | 0.607 | 2.191 | 25.57% | 1.33% | 3.7 |
| 128 | 2048 | 4.0 | 5.88 | cuda | 0.402 | 0.972 | 1.338 | 0.422 | 2.243 | 1.04% | 4.09% | 3.8 |
| 128 | 2048 | 4.0 | 5.88 | cuda_block_reduce | 0.386 | 0.511 | 0.897 | 0.425 | 2.059 | 1.71% | 1.74% | 3.8 |
| 128 | 2048 | 16.0 | 18.63 | cuda_block_reduce | 0.412 | 0.515 | 0.881 | 0.421 | 2.055 | 2.38% | 1.58% | 3.8 |
| 128 | 2048 | 16.0 | 18.63 | cuda | 0.413 | 0.972 | 1.332 | 0.417 | 2.245 | 0.67% | 3.62% | 3.8 |
| 256 | 512 | 4.0 | 4.20 | cuda | 0.408 | 0.504 | 1.317 | 0.424 | 2.553 | 2.00% | 1.35% | 14.5 |
| 256 | 512 | 4.0 | 4.20 | cuda_block_reduce | 0.420 | 0.516 | 1.098 | 0.419 | 2.368 | 3.23% | 1.03% | 14.5 |
| 256 | 512 | 16.0 | 15.30 | cuda_block_reduce | 0.492 | 0.559 | 1.165 | 0.422 | 2.476 | 9.65% | 5.07% | 14.5 |
| 256 | 512 | 16.0 | 15.30 | cuda | 0.494 | 0.551 | 1.376 | 0.427 | 2.619 | 7.18% | 1.06% | 14.5 |
| 256 | 2048 | 4.0 | 4.75 | cuda | 0.432 | 1.154 | 2.021 | 0.428 | 3.254 | 1.43% | 1.97% | 14.7 |
| 256 | 2048 | 4.0 | 4.75 | cuda_block_reduce | 0.434 | 0.580 | 1.203 | 0.425 | 2.454 | 6.36% | 1.24% | 14.7 |
| 256 | 2048 | 16.0 | 16.91 | cuda_block_reduce | 0.530 | 0.583 | 1.237 | 0.427 | 2.564 | 5.12% | 1.93% | 14.7 |
| 256 | 2048 | 16.0 | 16.91 | cuda | 0.517 | 1.196 | 2.080 | 0.435 | 3.408 | 2.10% | 2.38% | 14.7 |

## Limitations

- This is a deterministic synthetic micro-fit, not an end-to-end fit or image benchmark.
- CUDA events measure device work on the current stream; Python launch overhead, extension compilation, input construction, and synchronization wait are excluded.
- Support AABB visits and source atomic counts are explanatory proxies; only paired CUDA-event deltas enter the microprofile speedup gate.
- Peak memory is the PyTorch allocator peak, not total process or device VRAM.
- Medians describe this device/source grid; no confidence interval or cross-GPU claim is made.
- Baseline/candidate timing ratios are emitted only after same-equation forward/backward parity; the PyTorch reference is correctness calibration and is not timed.
- The candidate remains opt-in regardless of this microprofile decision.
