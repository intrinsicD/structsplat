from dataclasses import replace

import numpy as np
import pytest
import torch

from structsplat.additive_basis import CachedAdditiveBasis
from structsplat.contraction_refinement import (
    CoefficientProjectionConfig, project_contracted_coefficients,
)
from structsplat.overlap_elimination import lattice_observation_field
from structsplat.pixel_contraction import render_observation_field


@pytest.mark.parametrize("mode", ["scatter", "csr"])
def test_cached_map_matches_dense_forward_adjoint_diagonal_and_owns_storage(mode):
    # Unique row/pixel incidence includes overlap, a zero weight, and an unused row/pixel.
    row = torch.tensor([0, 1, 0, 2, 1])
    pixel = torch.tensor([0, 0, 1, 2, 3])
    weight = torch.tensor([0.7, 0.2, 0.5, 0.0, 0.9], dtype=torch.float64)
    dense = torch.zeros(5, 4, dtype=torch.float64)
    dense[pixel, row] = weight
    cache = CachedAdditiveBasis([(row, pixel, weight)], rows=4, pixels=5,
                                reference=weight, mode=mode)
    weight.zero_()
    row.fill_(3)
    pixel.fill_(4)
    x = torch.arange(12, dtype=torch.float64).reshape(4, 3) / 13
    y = torch.cos(torch.arange(15, dtype=torch.float64)).reshape(5, 3)
    torch.testing.assert_close(cache.apply(x), dense @ x)
    torch.testing.assert_close(cache.transpose(y), dense.T @ y)
    torch.testing.assert_close(cache.normal_diagonal(), dense.square().sum(0))
    torch.testing.assert_close((cache.apply(x) * y).sum(), (x * cache.transpose(y)).sum())
    assert cache.nnz == 4
    assert cache.resident_bytes > 0
    expected_bytes = 4 * (16 + 8) if mode == "scatter" else 2 * 4 * (8 + 8) + 8 * (4 + 5 + 2)
    assert cache.resident_bytes == expected_bytes
    with pytest.raises(ValueError, match="leading"):
        cache.apply(torch.ones(3, 3, dtype=torch.float64))


@pytest.mark.parametrize("mode", ["scatter", "csr"])
def test_empty_cache_and_memory_limit(mode):
    reference = torch.zeros(1)
    cache = CachedAdditiveBasis([], rows=2, pixels=3, reference=reference, mode=mode)
    assert torch.equal(cache.apply(torch.ones(2, 3)), torch.zeros(3, 3))
    assert torch.equal(cache.normal_diagonal(), torch.zeros(2))
    with pytest.raises(MemoryError):
        CachedAdditiveBasis([(torch.tensor([0]), torch.tensor([0]), torch.ones(1))],
                            rows=1, pixels=1, reference=reference, mode=mode, max_bytes=1)


def test_cache_rejects_invalid_indices_and_values():
    with pytest.raises(ValueError, match="outside"):
        CachedAdditiveBasis([(torch.tensor([1]), torch.tensor([0]), torch.ones(1))],
                            rows=1, pixels=1, reference=torch.ones(1))
    with pytest.raises(ValueError, match="finite"):
        CachedAdditiveBasis([(torch.tensor([0]), torch.tensor([0]), torch.tensor([float('nan')]))],
                            rows=1, pixels=1, reference=torch.ones(1))
    with pytest.raises(ValueError, match="basis_cache"):
        CoefficientProjectionConfig(basis_cache="unknown")


@pytest.mark.parametrize("mode", ["scatter", "csr"])
@pytest.mark.parametrize("partial", [False, True])
def test_cached_projection_matches_streaming_with_mask_and_frozen_rows(mode, partial):
    mask = np.ones((11, 13), dtype=bool)
    mask[:, 0] = False
    basis = np.zeros_like(mask)
    basis[3, 3] = basis[3, 7] = basis[7, 5] = True
    coeff = np.asarray([[0.2, 0.4, 0.1], [0.3, 0.2, 0.5], [0.1, 0.6, 0.3]], np.float32)
    true_field = lattice_observation_field(mask, basis, coeff, scale_px=1.5, sigma_cutoff=3.0)
    target = render_observation_field(true_field)
    field = replace(true_field, rgb_coeff=coeff * 0.6)
    touched = np.asarray([True, not partial, True])
    cfg = CoefficientProjectionConfig(max_iterations=12, pixel_rmse_threshold=2,
                                      patch7_rmse_threshold=2)
    off = project_contracted_coefficients(field, target, mask, touched, config=cfg)
    cached = project_contracted_coefficients(
        field, target, mask, touched, config=replace(cfg, basis_cache=mode),
    )
    np.testing.assert_allclose(cached.reconstruction_raw, off.reconstruction_raw, atol=2e-6)
    np.testing.assert_allclose(cached.field.rgb_coeff, off.field.rgb_coeff, atol=2e-5)
    assert cached.basis_cache == mode
    assert cached.basis_cache_bytes > 0
    assert cached.maintained_render_parity_max_abs < 2e-6
    assert cached.adjoint_relative_error < 2e-6
    if partial:
        np.testing.assert_array_equal(cached.field.rgb_coeff[1], field.rgb_coeff[1])


@pytest.mark.parametrize("mode", ["scatter", "csr"])
@pytest.mark.parametrize("fade,dilation", [(0, 0), (1, 0), (1, 0.3)])
def test_anisotropic_cached_projection_preserves_protected_and_untouched_rows(mode, fade, dilation):
    from structsplat.gaussians import GaussianField
    from structsplat.observation_field import CanvasCropTransform, adapt_factorized_additive_gaussian_field

    means = np.asarray([[3, 3], [6, 4], [8, 7]], np.float32)
    source = GaussianField.from_numpy(means, np.asarray([[2, 0.7], [1.5, 0.8], [1, 2]], np.float32),
                                     np.asarray([0.4, -0.7, 0.2], np.float32),
                                     np.full((3, 3), 0.3, np.float32))
    truth = adapt_factorized_additive_gaussian_field(source,
        canvas_crop=CanvasCropTransform(13, 11, 0, 0, 13, 11), coefficient_domain="signed",
        sigma_cutoff=3, support_fade_alpha=fade, aa_dilation_px2=dilation).require_pixel_exact()
    target = render_observation_field(truth)
    field = replace(truth, rgb_coeff=truth.rgb_coeff * 0.5)
    mask = np.ones((11, 13), bool)
    mask[1:4, 7:10] = False
    cfg = CoefficientProjectionConfig(max_iterations=12)
    touched, protected = np.array([True, True, False]), np.array([True, False, False])
    off = project_contracted_coefficients(field, target, mask, touched, protected, config=cfg, render_chunk=1)
    cached = project_contracted_coefficients(field, target, mask, touched, protected,
                                             config=replace(cfg, basis_cache=mode), render_chunk=1)
    np.testing.assert_allclose(cached.reconstruction_raw, off.reconstruction_raw, atol=2e-6)
    np.testing.assert_array_equal(cached.field.rgb_coeff[[0, 2]], field.rgb_coeff[[0, 2]])
    assert cached.protected_rows == 1 and cached.trainable_rows == 1
