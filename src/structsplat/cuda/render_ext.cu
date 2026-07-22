#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <torch/extension.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
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

__device__ __forceinline__ float visible_weight(float raw, bool support_fade, float tail) {
  if (!support_fade) {
    return raw;
  }
  return fmaxf(raw - tail, 0.0f);
}

__device__ __forceinline__ float dvisible_dq(float raw, bool support_fade, float tail) {
  if (support_fade && raw <= tail) {
    return 0.0f;
  }
  return -0.5f * raw;
}

__global__ void tile_count_kernel(
    const float* __restrict__ means,
    const int64_t* __restrict__ radii,
    int n,
    int height,
    int width,
    int tile_size,
    int64_t* __restrict__ counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  int x0, y0, tx, ty;
  support_bounds(means, radii, i, height, width, x0, y0, tx, ty);
  if (tx <= 0 || ty <= 0) {
    counts[i] = 0;
    return;
  }
  int tile_x0 = x0 / tile_size;
  int tile_y0 = y0 / tile_size;
  int tile_x1 = (x0 + tx - 1) / tile_size;
  int tile_y1 = (y0 + ty - 1) / tile_size;
  counts[i] = static_cast<int64_t>(tile_x1 - tile_x0 + 1) *
      static_cast<int64_t>(tile_y1 - tile_y0 + 1);
}

__device__ __forceinline__ float conic_q(
    float dx, float dy, float a, float b, float c) {
  return a * dx * dx + 2.0f * b * dx * dy + c * dy * dy;
}

// Exact continuous minimum of a positive-definite conic quadratic over an axis-aligned
// rectangle. A continuous lower bound is conservative for the integer pixel centers evaluated
// by the renderer: when this minimum exceeds cutoff^2, every sampled weight is exactly zero
// under support fade.
__device__ __forceinline__ float tile_min_q(
    float mx,
    float my,
    float a,
    float b,
    float c,
    float x_lo,
    float x_hi,
    float y_lo,
    float y_hi) {
  float determinant = a * c - b * b;
  if (!(a > 0.0f && c > 0.0f && determinant > 0.0f)) {
    return -FLT_MAX;  // malformed conic: retain the pair rather than cull unsafely
  }
  if (mx >= x_lo && mx <= x_hi && my >= y_lo && my <= y_hi) {
    return 0.0f;
  }

  float best = FLT_MAX;
  float dx = x_lo - mx;
  float dy = fminf(fmaxf(-(b / c) * dx, y_lo - my), y_hi - my);
  best = fminf(best, conic_q(dx, dy, a, b, c));
  dx = x_hi - mx;
  dy = fminf(fmaxf(-(b / c) * dx, y_lo - my), y_hi - my);
  best = fminf(best, conic_q(dx, dy, a, b, c));

  dy = y_lo - my;
  dx = fminf(fmaxf(-(b / a) * dy, x_lo - mx), x_hi - mx);
  best = fminf(best, conic_q(dx, dy, a, b, c));
  dy = y_hi - my;
  dx = fminf(fmaxf(-(b / a) * dy, x_lo - mx), x_hi - mx);
  return fminf(best, conic_q(dx, dy, a, b, c));
}

__global__ void expand_pairs_kernel(
    const float* __restrict__ means,
    const float* __restrict__ conics,
    const int64_t* __restrict__ radii,
    int n,
    int height,
    int width,
    int tile_size,
    int tiles_x,
    int num_tiles,
    bool ellipse_cull,
    float cutoff2,
    const int64_t* __restrict__ pair_starts,
    int32_t* __restrict__ keys,
    int32_t* __restrict__ values) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  int x0, y0, tx, ty;
  support_bounds(means, radii, i, height, width, x0, y0, tx, ty);
  if (tx <= 0 || ty <= 0) {
    return;
  }
  int tile_x0 = x0 / tile_size;
  int tile_y0 = y0 / tile_size;
  int tile_x1 = (x0 + tx - 1) / tile_size;
  int tile_y1 = (y0 + ty - 1) / tile_size;
  int tile_width = tile_x1 - tile_x0 + 1;
  int pair_count = tile_width * (tile_y1 - tile_y0 + 1);
  int64_t start = pair_starts[i];
  float mx = means[i * 2 + 0];
  float my = means[i * 2 + 1];
  float a = conics[i * 3 + 0];
  float b = conics[i * 3 + 1];
  float c = conics[i * 3 + 2];
  int support_x1 = x0 + tx - 1;
  int support_y1 = y0 + ty - 1;

  for (int local = 0; local < pair_count; ++local) {
    int tile_x = tile_x0 + local % tile_width;
    int tile_y = tile_y0 + local / tile_width;
    int key = tile_y * tiles_x + tile_x;
    if (ellipse_cull) {
      int rect_x0 = max(x0, tile_x * tile_size);
      int rect_y0 = max(y0, tile_y * tile_size);
      int rect_x1 = min(support_x1, (tile_x + 1) * tile_size - 1);
      int rect_y1 = min(support_y1, (tile_y + 1) * tile_size - 1);
      float minimum = tile_min_q(
          mx, my, a, b, c,
          static_cast<float>(rect_x0), static_cast<float>(rect_x1),
          static_cast<float>(rect_y0), static_cast<float>(rect_y1));
      float cull_guard = 1e-5f * fmaxf(cutoff2, 1.0f);
      if (minimum > cutoff2 + cull_guard) {
        key = num_tiles;  // sentinel sorts after all live tile ids
      }
    }
    int64_t pair = start + local;
    keys[pair] = static_cast<int32_t>(key);
    values[pair] = static_cast<int32_t>(i);
  }
}

__global__ void tile_offsets_from_keys_kernel(
    const int32_t* __restrict__ sorted_keys,
    int64_t count,
    int num_tiles,
    int32_t* __restrict__ offsets) {
  int tile = blockIdx.x * blockDim.x + threadIdx.x;
  if (tile > num_tiles) {
    return;
  }
  int64_t lo = 0;
  int64_t hi = count;
  while (lo < hi) {
    int64_t mid = lo + (hi - lo) / 2;
    if (sorted_keys[mid] < tile) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  offsets[tile] = static_cast<int32_t>(lo);
}

constexpr int kTileStage = 256;

struct TileStage {
  float mx[kTileStage];
  float my[kTileStage];
  float a[kTileStage];
  float b[kTileStage];
  float c[kTileStage];
  float op[kTileStage];
  float cr[kTileStage];
  float cg[kTileStage];
  float cb[kTileStage];
  int32_t gid[kTileStage];
  int32_t x0[kTileStage];
  int32_t x1[kTileStage];
  int32_t y0[kTileStage];
  int32_t y1[kTileStage];
};

__device__ __forceinline__ void stage_tile_batch(
    TileStage& stage,
    const float* __restrict__ means,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const int64_t* __restrict__ radii,
    const float* __restrict__ opacities,
    const int32_t* __restrict__ tile_gids,
    bool has_opacity,
    int height,
    int width,
    int batch,
    int count) {
  for (int j = threadIdx.x; j < count; j += blockDim.x) {
    int i = tile_gids[batch + j];
    int x0, y0, tx, ty;
    support_bounds(means, radii, i, height, width, x0, y0, tx, ty);
    stage.gid[j] = i;
    stage.x0[j] = x0;
    stage.x1[j] = x0 + tx;
    stage.y0[j] = y0;
    stage.y1[j] = y0 + ty;
    stage.mx[j] = means[i * 2 + 0];
    stage.my[j] = means[i * 2 + 1];
    stage.a[j] = conics[i * 3 + 0];
    stage.b[j] = conics[i * 3 + 1];
    stage.c[j] = conics[i * 3 + 2];
    stage.op[j] = has_opacity ? opacities[i] : 1.0f;
    stage.cr[j] = colors[i * 3 + 0];
    stage.cg[j] = colors[i * 3 + 1];
    stage.cb[j] = colors[i * 3 + 2];
  }
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
    bool support_fade,
    float tail,
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
    float w = visible_weight(expf(-0.5f * q), support_fade, tail) * op;
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

__global__ void tiled_forward_kernel(
    const float* __restrict__ means,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const int64_t* __restrict__ radii,
    const float* __restrict__ opacities,
    const int32_t* __restrict__ tile_gids,
    const int32_t* __restrict__ tile_offsets,
    int height,
    int width,
    int tiles_x,
    int tile_size,
    bool has_opacity,
    bool normalize,
    float eps,
    bool support_fade,
    float tail,
    float* __restrict__ out,
    float* __restrict__ den) {
  __shared__ TileStage stage;
  int tile = blockIdx.x;
  int local = threadIdx.x;
  int tile_pixels = tile_size * tile_size;
  int px = 0;
  int py = 0;
  bool active = local < tile_pixels;
  if (active) {
    int tile_x = tile % tiles_x;
    int tile_y = tile / tiles_x;
    px = tile_x * tile_size + (local % tile_size);
    py = tile_y * tile_size + (local / tile_size);
    active = px < width && py < height;
  }

  float nr = 0.0f;
  float ng = 0.0f;
  float nb = 0.0f;
  float d = 0.0f;
  int start = tile_offsets[tile];
  int end = tile_offsets[tile + 1];
  for (int batch = start; batch < end; batch += kTileStage) {
    int m = min(kTileStage, end - batch);
    stage_tile_batch(stage, means, conics, colors, radii, opacities, tile_gids, has_opacity,
                     height, width, batch, m);
    __syncthreads();
    if (active) {
      float fx = static_cast<float>(px);
      float fy = static_cast<float>(py);
      for (int j = 0; j < m; ++j) {
        if (px < stage.x0[j] || px >= stage.x1[j] ||
            py < stage.y0[j] || py >= stage.y1[j]) {
          continue;
        }
        float dx = fx - stage.mx[j];
        float dy = fy - stage.my[j];
        float q = conic_q(dx, dy, stage.a[j], stage.b[j], stage.c[j]);
        float weight = visible_weight(expf(-0.5f * q), support_fade, tail) * stage.op[j];
        nr += weight * stage.cr[j];
        ng += weight * stage.cg[j];
        nb += weight * stage.cb[j];
        d += weight;
      }
    }
    __syncthreads();
  }

  if (!active) {
    return;
  }

  int flat = py * width + px;
  den[flat] = d;
  if (normalize) {
    float denom = d + eps;
    out[flat * 3 + 0] = nr / denom;
    out[flat * 3 + 1] = ng / denom;
    out[flat * 3 + 2] = nb / denom;
  } else {
    out[flat * 3 + 0] = nr;
    out[flat * 3 + 1] = ng;
    out[flat * 3 + 2] = nb;
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
    bool support_fade,
    float tail,
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
    float raw = expf(-0.5f * q);
    float base = visible_weight(raw, support_fade, tail);
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
    float dq = dvisible_dq(raw, support_fade, tail) * dbase;
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

__device__ __forceinline__ float warp_reduce_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// Experimental untiled backward: preserve one block per Gaussian and the baseline thread-local
// pixel accumulation, then reduce the 8/9 gradient components within the block. Because no other
// block owns Gaussian i, thread 0 can write each result directly instead of issuing 256 contended
// atomicAdds per component. The baseline kernel remains available and is still the default.
__global__ void backward_block_reduce_kernel(
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
    bool support_fade,
    float tail,
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

  float gradients[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int t = threadIdx.x; t < total; t += blockDim.x) {
    int px = x0 + (t % tx);
    int py = y0 + (t / tx);
    int flat = py * width + px;
    float dx = static_cast<float>(px) - mx;
    float dy = static_cast<float>(py) - my;
    float q = a * dx * dx + 2.0f * b * dx * dy + c * dy * dy;
    float raw = expf(-0.5f * q);
    float base = visible_weight(raw, support_fade, tail);
    float weight = base * op;

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
      gradients[5] += dnum_r * weight;
      gradients[6] += dnum_g * weight;
      gradients[7] += dnum_b * weight;
      dw = dnum_r * cr + dnum_g * cg + dnum_b * cb + dden;
    } else {
      gradients[5] += gr * weight;
      gradients[6] += gg * weight;
      gradients[7] += gbch * weight;
      dw = gr * cr + gg * cg + gbch * cb;
    }

    float dbase = dw * op;
    if (has_opacity) {
      gradients[8] += dw * base;
    }
    float dq = dvisible_dq(raw, support_fade, tail) * dbase;
    gradients[0] += dq * (-2.0f * a * dx - 2.0f * b * dy);
    gradients[1] += dq * (-2.0f * b * dx - 2.0f * c * dy);
    gradients[2] += dq * dx * dx;
    gradients[3] += dq * 2.0f * dx * dy;
    gradients[4] += dq * dy * dy;
  }

  constexpr int kGradientComponents = 9;
  constexpr int kWarpsPerBlock = 8;
  __shared__ float warp_sums[kGradientComponents * kWarpsPerBlock];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  #pragma unroll
  for (int component = 0; component < kGradientComponents; ++component) {
    gradients[component] = warp_reduce_sum(gradients[component]);
    if (lane == 0) {
      warp_sums[component * kWarpsPerBlock + warp] = gradients[component];
    }
  }
  __syncthreads();

  if (warp == 0) {
    #pragma unroll
    for (int component = 0; component < kGradientComponents; ++component) {
      float value = lane < kWarpsPerBlock
          ? warp_sums[component * kWarpsPerBlock + lane]
          : 0.0f;
      value = warp_reduce_sum(value);
      if (lane == 0) {
        warp_sums[component * kWarpsPerBlock] = value;
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    grad_means[i * 2 + 0] = warp_sums[0 * kWarpsPerBlock];
    grad_means[i * 2 + 1] = warp_sums[1 * kWarpsPerBlock];
    grad_conics[i * 3 + 0] = warp_sums[2 * kWarpsPerBlock];
    grad_conics[i * 3 + 1] = warp_sums[3 * kWarpsPerBlock];
    grad_conics[i * 3 + 2] = warp_sums[4 * kWarpsPerBlock];
    grad_colors[i * 3 + 0] = warp_sums[5 * kWarpsPerBlock];
    grad_colors[i * 3 + 1] = warp_sums[6 * kWarpsPerBlock];
    grad_colors[i * 3 + 2] = warp_sums[7 * kWarpsPerBlock];
    if (has_opacity) {
      grad_opacities[i] = warp_sums[8 * kWarpsPerBlock];
    }
  }
}

// Tiled backward with shared-memory staging and warp-level gradient reduction (PORT-003).
// Every warp iterates the same staged Gaussian in lockstep, so the nine per-pixel gradient
// components can be shuffle-reduced across the 32 lanes and issued as one atomicAdd per warp
// per component instead of one per pixel — a 32x cut in global atomic traffic. The launch pads
// blockDim.x to a warp multiple so the full-mask shuffles below are always well defined;
// inactive lanes contribute exact zeros.
__global__ void tiled_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ means,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const int64_t* __restrict__ radii,
    const float* __restrict__ opacities,
    const float* __restrict__ den,
    const float* __restrict__ out,
    const int32_t* __restrict__ tile_gids,
    const int32_t* __restrict__ tile_offsets,
    int height,
    int width,
    int tiles_x,
    int tile_size,
    bool has_opacity,
    bool normalize,
    float eps,
    bool support_fade,
    float tail,
    float* __restrict__ grad_means,
    float* __restrict__ grad_conics,
    float* __restrict__ grad_colors,
    float* __restrict__ grad_opacities) {
  __shared__ TileStage stage;
  int tile = blockIdx.x;
  int local = threadIdx.x;
  int tile_pixels = tile_size * tile_size;
  int px = 0;
  int py = 0;
  bool active = local < tile_pixels;
  if (active) {
    int tile_x = tile % tiles_x;
    int tile_y = tile / tiles_x;
    px = tile_x * tile_size + (local % tile_size);
    py = tile_y * tile_size + (local / tile_size);
    active = px < width && py < height;
  }
  float fx = static_cast<float>(px);
  float fy = static_cast<float>(py);

  float gr = 0.0f;
  float gg = 0.0f;
  float gbch = 0.0f;
  float dnum_r = 0.0f;
  float dnum_g = 0.0f;
  float dnum_b = 0.0f;
  float dden = 0.0f;
  if (active) {
    int flat = py * width + px;
    gr = grad_out[flat * 3 + 0];
    gg = grad_out[flat * 3 + 1];
    gbch = grad_out[flat * 3 + 2];
    if (normalize) {
      float denom = den[flat] + eps;
      dnum_r = gr / denom;
      dnum_g = gg / denom;
      dnum_b = gbch / denom;
      dden = -(
          gr * out[flat * 3 + 0] +
          gg * out[flat * 3 + 1] +
          gbch * out[flat * 3 + 2]) / denom;
    }
  }

  int lane = threadIdx.x & 31;
  int start = tile_offsets[tile];
  int end = tile_offsets[tile + 1];
  for (int batch = start; batch < end; batch += kTileStage) {
    int m = min(kTileStage, end - batch);
    stage_tile_batch(stage, means, conics, colors, radii, opacities, tile_gids, has_opacity,
                     height, width, batch, m);
    __syncthreads();
    for (int j = 0; j < m; ++j) {
      bool hit = active && px >= stage.x0[j] && px < stage.x1[j] && py >= stage.y0[j] &&
          py < stage.y1[j];
      if (!__any_sync(0xffffffffu, hit)) {
        continue;
      }
      float gm0 = 0.0f;
      float gm1 = 0.0f;
      float ga = 0.0f;
      float gb = 0.0f;
      float gc = 0.0f;
      float gcr = 0.0f;
      float gcg = 0.0f;
      float gcb = 0.0f;
      float gop = 0.0f;
      if (hit) {
        float a = stage.a[j];
        float b = stage.b[j];
        float c = stage.c[j];
        float op = stage.op[j];
        float dx = fx - stage.mx[j];
        float dy = fy - stage.my[j];
        float q = a * dx * dx + 2.0f * b * dx * dy + c * dy * dy;
        float raw = expf(-0.5f * q);
        float base = visible_weight(raw, support_fade, tail);
        float weight = base * op;

        float dw;
        if (normalize) {
          gcr = dnum_r * weight;
          gcg = dnum_g * weight;
          gcb = dnum_b * weight;
          dw = dnum_r * stage.cr[j] + dnum_g * stage.cg[j] + dnum_b * stage.cb[j] + dden;
        } else {
          gcr = gr * weight;
          gcg = gg * weight;
          gcb = gbch * weight;
          dw = gr * stage.cr[j] + gg * stage.cg[j] + gbch * stage.cb[j];
        }

        float dbase = dw * op;
        if (has_opacity) {
          gop = dw * base;
        }
        float dq = dvisible_dq(raw, support_fade, tail) * dbase;
        gm0 = dq * (-2.0f * a * dx - 2.0f * b * dy);
        gm1 = dq * (-2.0f * b * dx - 2.0f * c * dy);
        ga = dq * dx * dx;
        gb = dq * 2.0f * dx * dy;
        gc = dq * dy * dy;
      }

      gm0 = warp_reduce_sum(gm0);
      gm1 = warp_reduce_sum(gm1);
      ga = warp_reduce_sum(ga);
      gb = warp_reduce_sum(gb);
      gc = warp_reduce_sum(gc);
      gcr = warp_reduce_sum(gcr);
      gcg = warp_reduce_sum(gcg);
      gcb = warp_reduce_sum(gcb);
      if (has_opacity) {
        gop = warp_reduce_sum(gop);
      }
      if (lane == 0) {
        int i = stage.gid[j];
        if (gm0 != 0.0f) atomicAdd(&grad_means[i * 2 + 0], gm0);
        if (gm1 != 0.0f) atomicAdd(&grad_means[i * 2 + 1], gm1);
        if (ga != 0.0f) atomicAdd(&grad_conics[i * 3 + 0], ga);
        if (gb != 0.0f) atomicAdd(&grad_conics[i * 3 + 1], gb);
        if (gc != 0.0f) atomicAdd(&grad_conics[i * 3 + 2], gc);
        if (gcr != 0.0f) atomicAdd(&grad_colors[i * 3 + 0], gcr);
        if (gcg != 0.0f) atomicAdd(&grad_colors[i * 3 + 1], gcg);
        if (gcb != 0.0f) atomicAdd(&grad_colors[i * 3 + 2], gcb);
        if (has_opacity && gop != 0.0f) atomicAdd(&grad_opacities[i], gop);
      }
    }
    __syncthreads();
  }
}

// Threads per tiled block: tile_size^2 rounded up to a warp multiple so the backward kernel's
// full-mask warp shuffles are always well defined (tile_size^2 <= 1024 keeps this <= 1024).
int tiled_block_threads(int tile_size) {
  int tile_pixels = tile_size * tile_size;
  return ((tile_pixels + 31) / 32) * 32;
}

}  // namespace

std::vector<torch::Tensor> structsplat_build_tile_index_cuda(
    torch::Tensor means,
    torch::Tensor conics,
    torch::Tensor radii,
    int64_t height,
    int64_t width,
    int64_t tile_size,
    bool ellipse_cull,
    double sigma_cutoff) {
  int n = static_cast<int>(means.size(0));
  int h = static_cast<int>(height);
  int w = static_cast<int>(width);
  int ts = static_cast<int>(tile_size);
  int tiles_x = (w + ts - 1) / ts;
  int tiles_y = (h + ts - 1) / ts;
  int64_t num_tiles64 = static_cast<int64_t>(tiles_x) * static_cast<int64_t>(tiles_y);
  TORCH_CHECK(num_tiles64 + 1 <= INT32_MAX, "tile grid exceeds int32 tile ids");
  int num_tiles = static_cast<int>(num_tiles64);

  auto int_opts = means.options().dtype(at::kInt);
  auto long_opts = means.options().dtype(at::kLong);
  auto offsets = torch::zeros({num_tiles + 1}, int_opts);
  auto empty_gids = torch::empty({0}, int_opts);
  if (n == 0 || num_tiles == 0) {
    return {empty_gids, offsets};
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  int gaussian_blocks = (n + threads - 1) / threads;

  auto counts = torch::empty({n}, long_opts);
  tile_count_kernel<<<gaussian_blocks, threads, 0, stream>>>(
      means.data_ptr<float>(), radii.data_ptr<int64_t>(), n, h, w, ts,
      counts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto pair_starts = torch::zeros({n + 1}, long_opts);
  size_t scan_bytes = 0;
  cub::DeviceScan::InclusiveSum(
      nullptr, scan_bytes, counts.data_ptr<int64_t>(), pair_starts.data_ptr<int64_t>() + 1, n,
      stream);
  auto scan_temp = torch::empty(
      {static_cast<int64_t>(std::max<size_t>(scan_bytes, 1))},
      means.options().dtype(at::kByte));
  cub::DeviceScan::InclusiveSum(
      scan_temp.data_ptr(), scan_bytes, counts.data_ptr<int64_t>(),
      pair_starts.data_ptr<int64_t>() + 1, n, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // One intentional device->host sync per call: the pair total sizes the sort buffers. This is
  // the same unavoidable scalar readback tile-sorted splatting rasterizers perform.
  int64_t total = pair_starts[n].item<int64_t>();
  TORCH_CHECK(total <= INT32_MAX, "tile index pair count exceeds int32");
  if (total == 0) {
    return {empty_gids, offsets};
  }

  auto keys_in = torch::empty({total}, int_opts);
  auto vals_in = torch::empty({total}, int_opts);
  auto keys_out = torch::empty({total}, int_opts);
  auto vals_out = torch::empty({total}, int_opts);
  float cutoff2 = static_cast<float>(sigma_cutoff * sigma_cutoff);
  expand_pairs_kernel<<<gaussian_blocks, threads, 0, stream>>>(
      means.data_ptr<float>(), conics.data_ptr<float>(), radii.data_ptr<int64_t>(), n, h, w, ts,
      tiles_x, num_tiles, ellipse_cull, cutoff2, pair_starts.data_ptr<int64_t>(),
      keys_in.data_ptr<int32_t>(), vals_in.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Keys span [0, num_tiles] (sentinel included); restrict the radix passes to the live bits.
  int end_bit = 1;
  while ((1LL << end_bit) < num_tiles64 + 1) {
    ++end_bit;
  }
  size_t sort_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr, sort_bytes, keys_in.data_ptr<int32_t>(), keys_out.data_ptr<int32_t>(),
      vals_in.data_ptr<int32_t>(), vals_out.data_ptr<int32_t>(), static_cast<int>(total), 0,
      end_bit, stream);
  auto sort_temp = torch::empty(
      {static_cast<int64_t>(std::max<size_t>(sort_bytes, 1))},
      means.options().dtype(at::kByte));
  cub::DeviceRadixSort::SortPairs(
      sort_temp.data_ptr(), sort_bytes, keys_in.data_ptr<int32_t>(),
      keys_out.data_ptr<int32_t>(), vals_in.data_ptr<int32_t>(), vals_out.data_ptr<int32_t>(),
      static_cast<int>(total), 0, end_bit, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int offset_blocks = (num_tiles + 1 + threads - 1) / threads;
  tile_offsets_from_keys_kernel<<<offset_blocks, threads, 0, stream>>>(
      keys_out.data_ptr<int32_t>(), total, num_tiles, offsets.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {vals_out, offsets};
}

std::vector<torch::Tensor> structsplat_render_forward_cuda(
    torch::Tensor means,
    torch::Tensor conics,
    torch::Tensor colors,
    torch::Tensor radii,
    torch::Tensor opacities,
    int64_t height,
    int64_t width,
    bool normalize,
    double eps,
    bool support_fade,
    double sigma_cutoff) {
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
    float tail = expf(-0.5f * static_cast<float>(sigma_cutoff * sigma_cutoff));
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
          support_fade,
          tail,
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

std::vector<torch::Tensor> structsplat_render_forward_tiled_cuda(
    torch::Tensor means,
    torch::Tensor conics,
    torch::Tensor colors,
    torch::Tensor radii,
    torch::Tensor opacities,
    torch::Tensor tile_gids,
    torch::Tensor tile_offsets,
    int64_t height,
    int64_t width,
    bool normalize,
    double eps,
    int64_t tile_size,
    bool support_fade,
    double sigma_cutoff) {
  int h = static_cast<int>(height);
  int w = static_cast<int>(width);
  int ts = static_cast<int>(tile_size);
  int pixels = h * w;
  int tiles_x = (w + ts - 1) / ts;
  int tiles_y = (h + ts - 1) / ts;
  int num_tiles = tiles_x * tiles_y;
  TORCH_CHECK(tile_offsets.size(0) == num_tiles + 1,
              "tile_offsets length must equal num_tiles + 1");
  auto out_flat = torch::zeros({pixels, 3}, means.options());
  auto den = torch::zeros({pixels}, means.options());
  if (pixels > 0 && num_tiles > 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int threads = tiled_block_threads(ts);
    float tail = expf(-0.5f * static_cast<float>(sigma_cutoff * sigma_cutoff));
    tiled_forward_kernel<<<num_tiles, threads, 0, stream>>>(
        means.data_ptr<float>(),
        conics.data_ptr<float>(),
        colors.data_ptr<float>(),
        radii.data_ptr<int64_t>(),
        opacities.numel() ? opacities.data_ptr<float>() : nullptr,
        tile_gids.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        h,
        w,
        tiles_x,
        ts,
        opacities.numel() > 0,
        normalize,
        static_cast<float>(eps),
        support_fade,
        tail,
        out_flat.data_ptr<float>(),
        den.data_ptr<float>());
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
    double eps,
    bool support_fade,
    double sigma_cutoff) {
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
    float tail = expf(-0.5f * static_cast<float>(sigma_cutoff * sigma_cutoff));
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
        support_fade,
        tail,
        grad_means.data_ptr<float>(),
        grad_conics.data_ptr<float>(),
        grad_colors.data_ptr<float>(),
        grad_opacities.numel() ? grad_opacities.data_ptr<float>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_means, grad_conics, grad_colors, grad_opacities};
}

std::vector<torch::Tensor> structsplat_render_backward_block_reduce_cuda(
    torch::Tensor grad_out,
    torch::Tensor means,
    torch::Tensor conics,
    torch::Tensor colors,
    torch::Tensor radii,
    torch::Tensor opacities,
    torch::Tensor den,
    torch::Tensor out,
    bool normalize,
    double eps,
    bool support_fade,
    double sigma_cutoff) {
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
    float tail = expf(-0.5f * static_cast<float>(sigma_cutoff * sigma_cutoff));
    backward_block_reduce_kernel<<<n, threads, 0, stream>>>(
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
        support_fade,
        tail,
        grad_means.data_ptr<float>(),
        grad_conics.data_ptr<float>(),
        grad_colors.data_ptr<float>(),
        grad_opacities.numel() ? grad_opacities.data_ptr<float>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_means, grad_conics, grad_colors, grad_opacities};
}

std::vector<torch::Tensor> structsplat_render_backward_tiled_cuda(
    torch::Tensor grad_out,
    torch::Tensor means,
    torch::Tensor conics,
    torch::Tensor colors,
    torch::Tensor radii,
    torch::Tensor opacities,
    torch::Tensor den,
    torch::Tensor out,
    torch::Tensor tile_gids,
    torch::Tensor tile_offsets,
    bool normalize,
    double eps,
    int64_t tile_size,
    bool support_fade,
    double sigma_cutoff) {
  int n = static_cast<int>(means.size(0));
  int h = static_cast<int>(out.size(0));
  int w = static_cast<int>(out.size(1));
  int ts = static_cast<int>(tile_size);
  int tiles_x = (w + ts - 1) / ts;
  int tiles_y = (h + ts - 1) / ts;
  int num_tiles = tiles_x * tiles_y;
  TORCH_CHECK(tile_offsets.size(0) == num_tiles + 1,
              "tile_offsets length must equal num_tiles + 1");
  auto grad_means = torch::zeros_like(means);
  auto grad_conics = torch::zeros_like(conics);
  auto grad_colors = torch::zeros_like(colors);
  auto grad_opacities = opacities.numel()
      ? torch::zeros_like(opacities)
      : torch::empty({0}, means.options());
  if (n > 0 && h > 0 && w > 0 && num_tiles > 0) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int threads = tiled_block_threads(ts);
    float tail = expf(-0.5f * static_cast<float>(sigma_cutoff * sigma_cutoff));
    tiled_backward_kernel<<<num_tiles, threads, 0, stream>>>(
        grad_out.data_ptr<float>(),
        means.data_ptr<float>(),
        conics.data_ptr<float>(),
        colors.data_ptr<float>(),
        radii.data_ptr<int64_t>(),
        opacities.numel() ? opacities.data_ptr<float>() : nullptr,
        den.data_ptr<float>(),
        out.data_ptr<float>(),
        tile_gids.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        h,
        w,
        tiles_x,
        ts,
        opacities.numel() > 0,
        normalize,
        static_cast<float>(eps),
        support_fade,
        tail,
        grad_means.data_ptr<float>(),
        grad_conics.data_ptr<float>(),
        grad_colors.data_ptr<float>(),
        grad_opacities.numel() ? grad_opacities.data_ptr<float>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {grad_means, grad_conics, grad_colors, grad_opacities};
}
