#include <torch/extension.h>

#include <vector>

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
    double sigma_cutoff);

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
    double sigma_cutoff);

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
    double sigma_cutoff);

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
    double sigma_cutoff);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_LONG(x) TORCH_CHECK(x.scalar_type() == at::kLong, #x " must be int64")

static void check_inputs(
    const torch::Tensor& means,
    const torch::Tensor& conics,
    const torch::Tensor& colors,
    const torch::Tensor& radii,
    const torch::Tensor& opacities) {
  CHECK_CUDA(means);
  CHECK_CUDA(conics);
  CHECK_CUDA(colors);
  CHECK_CUDA(radii);
  CHECK_CUDA(opacities);
  CHECK_CONTIGUOUS(means);
  CHECK_CONTIGUOUS(conics);
  CHECK_CONTIGUOUS(colors);
  CHECK_CONTIGUOUS(radii);
  CHECK_CONTIGUOUS(opacities);
  CHECK_FLOAT(means);
  CHECK_FLOAT(conics);
  CHECK_FLOAT(colors);
  CHECK_LONG(radii);
  TORCH_CHECK(opacities.numel() == 0 || opacities.scalar_type() == at::kFloat,
              "opacities must be empty or float32");
  TORCH_CHECK(means.dim() == 2 && means.size(1) == 2, "means must be (N, 2)");
  TORCH_CHECK(conics.dim() == 2 && conics.size(1) == 3, "conics must be (N, 3)");
  TORCH_CHECK(colors.dim() == 2 && colors.size(1) == 3, "colors must be (N, 3)");
  TORCH_CHECK(radii.dim() == 2 && radii.size(1) == 2, "radii must be (N, 2)");
  TORCH_CHECK(conics.size(0) == means.size(0), "conics N must match means N");
  TORCH_CHECK(colors.size(0) == means.size(0), "colors N must match means N");
  TORCH_CHECK(radii.size(0) == means.size(0), "radii N must match means N");
  TORCH_CHECK(opacities.numel() == 0 || opacities.numel() == means.size(0),
              "opacities must be empty or length N");
}

static void check_tile_index(const torch::Tensor& tile_gids,
                             const torch::Tensor& tile_offsets) {
  CHECK_CUDA(tile_gids);
  CHECK_CUDA(tile_offsets);
  CHECK_CONTIGUOUS(tile_gids);
  CHECK_CONTIGUOUS(tile_offsets);
  CHECK_LONG(tile_gids);
  CHECK_LONG(tile_offsets);
  TORCH_CHECK(tile_gids.dim() == 1, "tile_gids must be 1-D");
  TORCH_CHECK(tile_offsets.dim() == 1, "tile_offsets must be 1-D");
}

std::vector<torch::Tensor> forward(
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
  check_inputs(means, conics, colors, radii, opacities);
  TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
  return structsplat_render_forward_cuda(
      means, conics, colors, radii, opacities, height, width, normalize, eps,
      support_fade, sigma_cutoff);
}

std::vector<torch::Tensor> forward_tiled(
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
  check_inputs(means, conics, colors, radii, opacities);
  check_tile_index(tile_gids, tile_offsets);
  TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
  TORCH_CHECK(tile_size > 0 && tile_size * tile_size <= 1024,
              "tile_size must be positive with tile_size^2 <= 1024");
  return structsplat_render_forward_tiled_cuda(
      means, conics, colors, radii, opacities, tile_gids, tile_offsets,
      height, width, normalize, eps, tile_size, support_fade, sigma_cutoff);
}

std::vector<torch::Tensor> backward(
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
  check_inputs(means, conics, colors, radii, opacities);
  CHECK_CUDA(grad_out);
  CHECK_CUDA(den);
  CHECK_CUDA(out);
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(den);
  CHECK_CONTIGUOUS(out);
  CHECK_FLOAT(grad_out);
  CHECK_FLOAT(den);
  CHECK_FLOAT(out);
  return structsplat_render_backward_cuda(
      grad_out, means, conics, colors, radii, opacities, den, out, normalize, eps,
      support_fade, sigma_cutoff);
}

std::vector<torch::Tensor> backward_tiled(
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
  check_inputs(means, conics, colors, radii, opacities);
  check_tile_index(tile_gids, tile_offsets);
  CHECK_CUDA(grad_out);
  CHECK_CUDA(den);
  CHECK_CUDA(out);
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(den);
  CHECK_CONTIGUOUS(out);
  CHECK_FLOAT(grad_out);
  CHECK_FLOAT(den);
  CHECK_FLOAT(out);
  TORCH_CHECK(tile_size > 0 && tile_size * tile_size <= 1024,
              "tile_size must be positive with tile_size^2 <= 1024");
  return structsplat_render_backward_tiled_cuda(
      grad_out, means, conics, colors, radii, opacities, den, out,
      tile_gids, tile_offsets, normalize, eps, tile_size, support_fade, sigma_cutoff);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "StructSplat exact render forward (CUDA)");
  m.def("forward_tiled", &forward_tiled, "StructSplat exact tiled render forward (CUDA)");
  m.def("backward", &backward, "StructSplat exact render backward (CUDA)");
  m.def("backward_tiled", &backward_tiled, "StructSplat exact tiled render backward (CUDA)");
}
