# Exact normalized CUDA block-reduction experiment

- Work scope: `synthetic_microfit_cuda_event_device_timeline_only`
- Device: NVIDIA GeForce RTX 3050
- Timing: current-stream torch.cuda.Event(enable_timing=True), synchronized per sample, median over repeats after warmup
- Nsight Compute: `unavailable_gpu_performance_counter_permission_denied`
- Source/device-bound microprofile speedup claim authorized: **yes**
- Default flip authorized: **no**

## Preregistered decision

**opt_in_microprofile_gate_pass** — pass=True.
Representative reductions: exact backward 57.48%, full step 25.29%, peak allocator-memory regression 0.00%.

A pass supports only this source/device-bound microprofile comparison. The candidate remains opt-in and no default, end-to-end fit, quality, or cross-GPU claim follows.

## Measured rows

| H | N | Requested overlap | Actual overlap | Mode | Forward ms | Exact backward ms | Fit backward ms | Adam ms | Full step ms | Backward CV | Step CV | Peak +MiB |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 512 | 4.0 | 4.67 | cuda | 0.386 | 0.498 | 0.856 | 0.415 | 2.032 | 4.14% | 1.44% | 3.7 |
| 128 | 512 | 4.0 | 4.67 | cuda_block_reduce | 0.391 | 0.513 | 0.855 | 0.418 | 2.043 | 2.72% | 1.21% | 3.7 |
| 128 | 512 | 16.0 | 16.19 | cuda_block_reduce | 0.400 | 0.500 | 0.854 | 0.415 | 2.049 | 1.62% | 0.91% | 3.7 |
| 128 | 512 | 16.0 | 16.19 | cuda | 0.396 | 0.495 | 0.847 | 0.417 | 2.051 | 2.15% | 1.54% | 3.7 |
| 128 | 2048 | 4.0 | 5.88 | cuda | 0.389 | 0.972 | 1.342 | 0.414 | 2.213 | 0.45% | 3.72% | 3.8 |
| 128 | 2048 | 4.0 | 5.88 | cuda_block_reduce | 0.387 | 0.512 | 0.849 | 0.426 | 2.042 | 3.78% | 3.03% | 3.8 |
| 128 | 2048 | 16.0 | 18.63 | cuda_block_reduce | 0.406 | 0.493 | 0.852 | 0.415 | 2.034 | 1.51% | 1.93% | 3.8 |
| 128 | 2048 | 16.0 | 18.63 | cuda | 0.414 | 0.969 | 1.326 | 0.414 | 2.199 | 0.57% | 3.93% | 3.8 |
| 256 | 512 | 4.0 | 4.20 | cuda | 0.417 | 0.497 | 1.316 | 0.422 | 2.538 | 3.66% | 1.68% | 14.5 |
| 256 | 512 | 4.0 | 4.20 | cuda_block_reduce | 0.407 | 0.494 | 1.097 | 0.418 | 2.347 | 1.92% | 0.60% | 14.5 |
| 256 | 512 | 16.0 | 15.30 | cuda_block_reduce | 0.496 | 0.507 | 1.127 | 0.425 | 2.464 | 3.22% | 1.09% | 14.5 |
| 256 | 512 | 16.0 | 15.30 | cuda | 0.493 | 0.495 | 1.339 | 0.413 | 2.627 | 2.87% | 0.98% | 14.5 |
| 256 | 2048 | 4.0 | 4.75 | cuda | 0.420 | 1.109 | 1.971 | 0.426 | 3.225 | 0.71% | 2.13% | 14.7 |
| 256 | 2048 | 4.0 | 4.75 | cuda_block_reduce | 0.416 | 0.514 | 1.160 | 0.423 | 2.433 | 3.36% | 0.50% | 14.7 |
| 256 | 2048 | 16.0 | 16.91 | cuda_block_reduce | 0.512 | 0.501 | 1.197 | 0.415 | 2.559 | 2.34% | 1.18% | 14.7 |
| 256 | 2048 | 16.0 | 16.91 | cuda | 0.506 | 1.178 | 2.045 | 0.435 | 3.425 | 0.96% | 2.93% | 14.7 |

## Limitations

- This is a deterministic synthetic micro-fit, not an end-to-end fit or image benchmark.
- CUDA events measure device work on the current stream; Python launch overhead, extension compilation, input construction, and synchronization wait are excluded.
- Support AABB visits and source atomic counts are explanatory proxies; only paired CUDA-event deltas enter the microprofile speedup gate.
- Peak memory is the PyTorch allocator peak, not total process or device VRAM.
- Medians describe this device/source grid; no confidence interval or cross-GPU claim is made.
- Baseline/candidate timing ratios are emitted only after same-equation forward/backward parity; the PyTorch reference is correctness calibration and is not timed.
- The candidate remains opt-in regardless of this microprofile decision.
