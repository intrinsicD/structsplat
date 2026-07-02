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
    double eps);

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
    double eps);

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

std::vector<torch::Tensor> forward(
    torch::Tensor means,
    torch::Tensor conics,
    torch::Tensor colors,
    torch::Tensor radii,
    torch::Tensor opacities,
    int64_t height,
    int64_t width,
    bool normalize,
    double eps) {
  check_inputs(means, conics, colors, radii, opacities);
  TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
  return structsplat_render_forward_cuda(
      means, conics, colors, radii, opacities, height, width, normalize, eps);
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
    double eps) {
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
      grad_out, means, conics, colors, radii, opacities, den, out, normalize, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "StructSplat exact render forward (CUDA)");
  m.def("backward", &backward, "StructSplat exact render backward (CUDA)");
}

