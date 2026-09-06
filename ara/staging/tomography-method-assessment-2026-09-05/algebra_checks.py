"""CPU algebra checks for an advisory review; not a reconstruction benchmark."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from rtgs.lift.beam_fusion import _ci_fuse
from rtgs.lift.field_loss import AnalyticGaussianField2D, field_l2, mixture_inner_product
from rtgs.lift.field_refit import _variable_field_objective

torch.set_num_threads(1)
dtype = torch.float64
repo = Path('/home/alex/Documents/realtime-gs')
source_names = [
    'src/rtgs/lift/beam_fusion.py', 'src/rtgs/lift/field_loss.py',
    'src/rtgs/lift/field_refit.py', 'src/rtgs/lift/inverse_projection_fiber.py',
    'src/rtgs/lift/field_visibility.py', 'src/rtgs/optim/compact_trainer.py',
]
source_hashes = {name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in source_names}

def make_field(means, covs, density, colors):
    m = torch.tensor(means, dtype=dtype, requires_grad=True)
    c = torch.tensor(covs, dtype=dtype)
    a = torch.tensor(density, dtype=dtype)
    rgb = a[:, None] * torch.tensor(colors, dtype=dtype)
    return AnalyticGaussianField2D(m, c, a, rgb)

pred = make_field([[0.3, 0.1]], [[[1.4, 0.15], [0.15, 0.8]]], [0.65], [[0.3, 0.4, 0.55]])
target = make_field([[0.0, 0.0]], [[[1.0, 0.0], [0.0, 1.0]]], [0.8], [[0.2, 0.4, 0.6]])
split = AnalyticGaussianField2D(
    target.means.detach().repeat(2, 1), target.covariances.repeat(2, 1, 1),
    target.density_amplitudes.repeat(2) / 2,
    target.rgb_amplitudes.repeat(2, 1) / 2,
)
plain = field_l2(pred, target)
split_plain = field_l2(pred, split)
assert torch.allclose(plain.density, split_plain.density, atol=1e-12, rtol=1e-12)
assert torch.allclose(plain.rgb_numerator, split_plain.rgb_numerator, atol=1e-12, rtol=1e-12)
# Isolate the RGB contribution; density_weight=0 is an algebra probe, not a FieldRefitConfig.
kwargs = dict(density_weight=0.0, rgb_weight=0.25, include_rgb=True, chunk_size=256)
value = _variable_field_objective(pred, target, **kwargs)
split_value = _variable_field_objective(pred, split, **kwargs)
grad = torch.autograd.grad(value, pred.means, retain_graph=True)[0]
split_grad = torch.autograd.grad(split_value, pred.means, retain_graph=True)[0]
assert torch.allclose(split_value, 2 * value, atol=1e-12, rtol=1e-12)
assert torch.allclose(split_grad, 2 * grad, atol=1e-12, rtol=1e-12)
assert float(grad.norm()) > 1e-6
# A proposed denominator based on field energy has the required representation invariance.
def energy(field):
    return mixture_inner_product(field.means, field.covariances, field.rgb_amplitudes,
                                 field.means, field.covariances, field.rgb_amplitudes)
energy_original, energy_split = energy(target), energy(split)
assert torch.allclose(energy_original, energy_split, atol=1e-12, rtol=1e-12)

# Algebraic orthographic control: three perpendicular views of covariance I;
# each lifted beam has transverse variance 1 and longitudinal variance L^2.
L = 1000.0
precisions = torch.eye(3, dtype=dtype).repeat(3, 1, 1)
for axis in range(3):
    precisions[axis, axis, axis] = 1.0 / L**2
fused_precision, fused_mean = _ci_fuse(precisions, torch.zeros((3, 3), dtype=dtype))
fused_covariance = torch.linalg.inv(fused_precision)
expected = 3.0 / (2.0 + 1.0 / L**2)
assert torch.allclose(fused_covariance, torch.eye(3, dtype=dtype) * expected, atol=1e-12)
# This probes the CI operator, not the complete perspective Beam Fusion pipeline.

# True line-integral peak for an anisotropic, unit-peak 3D Gaussian.
S = torch.diag(torch.tensor([1.0, 4.0, 9.0], dtype=dtype))
ray_integral_peaks = torch.sqrt(2 * torch.pi / torch.diag(torch.linalg.inv(S)))
assert torch.allclose(ray_integral_peaks / ray_integral_peaks[0], torch.tensor([1., 2., 3.], dtype=dtype))

# Normalized-color gauge: exact algebra includes epsilon, so no claim of exact
# invariance at a fixed nonzero epsilon. Doubling numerator/denominator alters D
# substantially while its color change tends to zero when D greatly exceeds epsilon.
D = torch.tensor([1.0, 2.0, 4.0], dtype=dtype)
N = D[:, None] * torch.tensor([[0.2, 0.4, 0.6]], dtype=dtype)
epsilon = 1e-8
base = N / (D[:, None] + epsilon)
scaled = 2 * N / (2 * D[:, None] + epsilon)
result = {
    'scope': 'CPU deterministic algebra verification; no scene data, optimization run or performance claim',
    'python': sys.version, 'torch': torch.__version__, 'device': 'cpu', 'dtype': str(dtype),
    'repository': str(repo),
    'commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
    'repository_status_before': subprocess.check_output(['git', '-C', str(repo), 'status', '--short'], text=True),
    'source_sha256': source_hashes,
    'rgb_split': {
        'plain_density_l2': float(plain.density.detach()),
        'split_density_l2': float(split_plain.density.detach()),
        'plain_rgb_l2': float(plain.rgb_numerator.detach()),
        'split_rgb_l2': float(split_plain.rgb_numerator.detach()),
        'current_rgb_objective': float(value.detach()),
        'split_rgb_objective': float(split_value.detach()),
        'current_rgb_mean_gradient': grad.tolist(),
        'split_rgb_mean_gradient': split_grad.tolist(),
        'gradient_norm_ratio': float(split_grad.norm() / grad.norm()),
        'original_coefficient_energy': float(target.rgb_amplitudes.square().sum()),
        'split_coefficient_energy': float(split.rgb_amplitudes.square().sum()),
        'original_field_energy': float(energy_original.detach()),
        'split_field_energy': float(energy_split.detach()),
    },
    'ci_orthographic_algebra': {
        'longitudinal_standard_deviation': L,
        'true_covariance': torch.eye(3).tolist(),
        'ci_covariance': fused_covariance.tolist(),
        'analytic_scalar': expected,
        'infinite_longitudinal_variance_limit': 1.5,
    },
    'line_integral_peak': {
        '3d_covariance': S.tolist(), 'central_3d_density': 1.0,
        'x_y_z_view_peaks': ray_integral_peaks.tolist(),
    },
    'normalized_gauge': {
        'epsilon': epsilon, 'density_scale': 2.0,
        'maximum_color_change': float((scaled-base).abs().max()),
        'qualification': 'An illustrative epsilon, not a measured current-artifact regime; exact only as epsilon approaches zero.',
    },
    'assertions_passed': True,
}
print(json.dumps(result, indent=2, allow_nan=False))
