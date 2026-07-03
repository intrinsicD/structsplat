#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

namespace {

__device__ __forceinline__ void support_bounds(
    const float* means,
    const int64_t* radii,
    int i,
    int height,
    int width,
    int& x0,
    int& y0,
    int& tx,
    int& ty) {
  float mx = means[i * 2 + 0];
  float my = means[i * 2 + 1];
  // A NaN/Inf mean (e.g. a diverged fit) casts to INT_MAX and then ix+rx signed-overflows and
  // wraps negative, defeating both clamps and producing a bogus positive tile size that reads
  // out of bounds. Drop such Gaussians (empty tile), matching the reference's zero contribution.
  if (!isfinite(mx) || !isfinite(my)) {
    x0 = 0;
    y0 = 0;
    tx = 0;
    ty = 0;
    return;
  }
  int rx = static_cast<int>(radii[i * 2 + 0]);
  int ry = static_cast<int>(radii[i * 2 + 1]);
  // Clamp the rounded center into a range where ix +/- r cannot overflow int before the cast.
  float fx = nearbyintf(mx);
  float fy = nearbyintf(my);
  fx = fminf(fmaxf(fx, static_cast<float>(-(rx + 1))), static_cast<float>(width + rx));
  fy = fminf(fmaxf(fy, static_cast<float>(-(ry + 1))), static_cast<float>(height + ry));
  int ix = static_cast<int>(fx);
  int iy = static_cast<int>(fy);
  x0 = max(ix - rx, 0);
  int x1 = min(ix + rx, width - 1);
  y0 = max(iy - ry, 0);
  int y1 = min(iy + ry, height - 1);
  tx = max(x1 - x0 + 1, 0);
  ty = max(y1 - y0 + 1, 0);
}

__global__ void accumulate_kernel(
    const float* __restrict__ means,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const int64_t* __restrict__ radii,
    const float* __restrict__ opacities,
    int n,
    int height,
    int width,
    bool has_opacity,
    bool normalize,
    float* __restrict__ num,
    float* __restrict__ den) {
  int i = blockIdx.x;
  if (i >= n) {
    return;
  }
  int x0, y0, tx, ty;
  support_bounds(means, radii, i, height, width, x0, y0, tx, ty);
  int total = tx * ty;
  if (total <= 0) {
    return;
  }

  float mx = means[i * 2 + 0];
  float my = means[i * 2 + 1];
  float a = conics[i * 3 + 0];
  float b = conics[i * 3 + 1];
  float c = conics[i * 3 + 2];
  float op = has_opacity ? opacities[i] : 1.0f;
  float cr = colors[i * 3 + 0];
  float cg = colors[i * 3 + 1];
  float cb = colors[i * 3 + 2];

  for (int t = threadIdx.x; t < total; t += blockDim.x) {
    int px = x0 + (t % tx);
    int py = y0 + (t / tx);
    float dx = static_cast<float>(px) - mx;
    float dy = static_cast<float>(py) - my;
    float q = a * dx * dx + 2.0f * b * dx * dy + c * dy * dy;
    float w = expf(-0.5f * q) * op;
    int flat = py * width + px;
    atomicAdd(&num[flat * 3 + 0], w * cr);
    atomicAdd(&num[flat * 3 + 1], w * cg);
    atomicAdd(&num[flat * 3 + 2], w * cb);
    if (normalize) {
      atomicAdd(&den[flat], w);
    }
  }
}

__global__ void finalize_kernel(
    const float* __restrict__ num,
    const float* __restrict__ den,
    int pixels,
    bool normalize,
    float eps,
    float* __restrict__ out) {
  int flat = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat >= pixels) {
    return;
  }
  if (normalize) {
    float denom = den[flat] + eps;
    out[flat * 3 + 0] = num[flat * 3 + 0] / denom;
    out[flat * 3 + 1] = num[flat * 3 + 1] / denom;
    out[flat * 3 + 2] = num[flat * 3 + 2] / denom;
  } else {
    out[flat * 3 + 0] = num[flat * 3 + 0];
    out[flat * 3 + 1] = num[flat * 3 + 1];
    out[flat * 3 + 2] = num[flat * 3 + 2];
  }
}

__global__ void backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ means,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const int64_t* __restrict__ radii,
    const float* __restrict__ opacities,
    const float* __restrict__ den,
    const float* __restrict__ out,
    int n,
    int height,
    int width,
    bool has_opacity,
    bool normalize,
    float eps,
    float* __restrict__ grad_means,
    float* __restrict__ grad_conics,
    float* __restrict__ grad_colors,
    float* __restrict__ grad_opacities) {
  int i = blockIdx.x;
  if (i >= n) {
    return;
  }
  int x0, y0, tx, ty;
  support_bounds(means, radii, i, height, width, x0, y0, tx, ty);
  int total = tx * ty;
  if (total <= 0) {
    return;
  }

  float mx = means[i * 2 + 0];
  float my = means[i * 2 + 1];
  float a = conics[i * 3 + 0];
  float b = conics[i * 3 + 1];
  float c = conics[i * 3 + 2];
  float op = has_opacity ? opacities[i] : 1.0f;
  float cr = colors[i * 3 + 0];
  float cg = colors[i * 3 + 1];
  float cb = colors[i * 3 + 2];

  float gm0 = 0.0f;
  float gm1 = 0.0f;
  float ga = 0.0f;
  float gb = 0.0f;
  float gc = 0.0f;
  float gcr = 0.0f;
  float gcg = 0.0f;
  float gcb = 0.0f;
  float gop = 0.0f;

  for (int t = threadIdx.x; t < total; t += blockDim.x) {
    int px = x0 + (t % tx);
    int py = y0 + (t / tx);
    int flat = py * width + px;
    float dx = static_cast<float>(px) - mx;
    float dy = static_cast<float>(py) - my;
    float q = a * dx * dx + 2.0f * b * dx * dy + c * dy * dy;
    float base = expf(-0.5f * q);
    float w = base * op;

    float gr = grad_out[flat * 3 + 0];
    float gg = grad_out[flat * 3 + 1];
    float gbch = grad_out[flat * 3 + 2];
    float dw;
    if (normalize) {
      float denom = den[flat] + eps;
      float dnum_r = gr / denom;
      float dnum_g = gg / denom;
      float dnum_b = gbch / denom;
      float dden = -(
          gr * out[flat * 3 + 0] +
          gg * out[flat * 3 + 1] +
          gbch * out[flat * 3 + 2]) / denom;
      gcr += dnum_r * w;
      gcg += dnum_g * w;
      gcb += dnum_b * w;
      dw = dnum_r * cr + dnum_g * cg + dnum_b * cb + dden;
    } else {
      gcr += gr * w;
      gcg += gg * w;
      gcb += gbch * w;
      dw = gr * cr + gg * cg + gbch * cb;
    }

    float dbase = dw * op;
    if (has_opacity) {
      gop += dw * base;
    }
    float dq = -0.5f * base * dbase;
    gm0 += dq * (-2.0f * a * dx - 2.0f * b * dy);
    gm1 += dq * (-2.0f * b * dx - 2.0f * c * dy);
    ga += dq * dx * dx;
    gb += dq * 2.0f * dx * dy;
    gc += dq * dy * dy;
  }

  atomicAdd(&grad_means[i * 2 + 0], gm0);
  atomicAdd(&grad_means[i * 2 + 1], gm1);
  atomicAdd(&grad_conics[i * 3 + 0], ga);
  atomicAdd(&grad_conics[i * 3 + 1], gb);
  atomicAdd(&grad_conics[i * 3 + 2], gc);
  atomicAdd(&grad_colors[i * 3 + 0], gcr);
  atomicAdd(&grad_colors[i * 3 + 1], gcg);
  atomicAdd(&grad_colors[i * 3 + 2], gcb);
  if (has_opacity) {
    atomicAdd(&grad_opacities[i], gop);
  }
}

}  // namespace

std::vector<torch::Tensor> structsplat_render_forward_cuda(
    torch::Tensor means,
    torch::Tensor conics,
    torch::Tensor colors,
    torch::Tensor radii,
    torch::Tensor opacities,
    int64_t height,
    int64_t width,
    bool normalize,
    double eps) {
  int n = static_cast<int>(means.size(0));
  int h = static_cast<int>(height);
  int w = static_cast<int>(width);
  int pixels = h * w;
  auto out_flat = torch::zeros({pixels, 3}, means.options());
  auto num = torch::zeros({pixels, 3}, means.options());
  auto den = torch::zeros({pixels}, means.options());
  if (pixels > 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 256;
    if (n > 0) {
      accumulate_kernel<<<n, threads, 0, stream>>>(
          means.data_ptr<float>(),
          conics.data_ptr<float>(),
          colors.data_ptr<float>(),
          radii.data_ptr<int64_t>(),
          opacities.numel() ? opacities.data_ptr<float>() : nullptr,
          n,
          h,
          w,
          opacities.numel() > 0,
          normalize,
          num.data_ptr<float>(),
          den.data_ptr<float>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    // Run finalize even when n == 0 so the output is defined zeros, matching the reference
    // renderer (num/den are all-zero, so normalized -> 0 and additive -> 0). Previously the
    // caller received uninitialized allocator memory for an empty field (CORE-004).
    int blocks = (pixels + threads - 1) / threads;
    finalize_kernel<<<blocks, threads, 0, stream>>>(
        num.data_ptr<float>(),
        den.data_ptr<float>(),
        pixels,
        normalize,
        static_cast<float>(eps),
        out_flat.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out_flat.view({height, width, 3}), den.view({height, width})};
}

std::vector<torch::Tensor> structsplat_render_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor means,
    torch::Tensor conics,
    torch::Tensor colors,
    torch::Tensor radii,
    torch::Tensor opacities,
    torch::Tensor den,
    torch::Tensor out,
    bool normalize,
    double eps) {
  int n = static_cast<int>(means.size(0));
  int h = static_cast<int>(out.size(0));
  int w = static_cast<int>(out.size(1));
  auto grad_means = torch::zeros_like(means);
  auto grad_conics = torch::zeros_like(conics);
  auto grad_colors = torch::zeros_like(colors);
  auto grad_opacities = opacities.numel()
      ? torch::zeros_like(opacities)
      : torch::empty({0}, means.options());
  if (n > 0 && h > 0 && w > 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 256;
    backward_kernel<<<n, threads, 0, stream>>>(
        grad_out.data_ptr<float>(),
        means.data_ptr<float>(),
        conics.data_ptr<float>(),
        colors.data_ptr<float>(),
        radii.data_ptr<int64_t>(),
        opacities.numel() ? opacities.data_ptr<float>() : nullptr,
        den.data_ptr<float>(),
        out.data_ptr<float>(),
        n,
        h,
        w,
        opacities.numel() > 0,
        normalize,
        static_cast<float>(eps),
        grad_means.data_ptr<float>(),
        grad_conics.data_ptr<float>(),
        grad_colors.data_ptr<float>(),
        grad_opacities.numel() ? grad_opacities.data_ptr<float>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_means, grad_conics, grad_colors, grad_opacities};
}

