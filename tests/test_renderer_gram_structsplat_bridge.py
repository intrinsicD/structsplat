import numpy as np
import pytest
import torch

from benchmarks import renderer_gram_structsplat_bridge as bridge_module
from structsplat.gaussians import GaussianField


def _synthetic_geometry(seed=31, *, n=11, height=13, width=15):
    rng = np.random.default_rng(seed)
    means = np.stack(
        [
            rng.uniform(-2.5, width + 2.5, n),
            rng.uniform(-2.5, height + 2.5, n),
        ],
        axis=1,
    )
    scales = np.exp(rng.uniform(np.log(0.65), np.log(3.2), (n, 2)))
    angles = rng.uniform(-np.pi, np.pi, n)
    colors = rng.uniform(-0.15, 1.15, (n, 3))
    field = GaussianField.from_numpy(means, scales, angles, colors)
    opacities = torch.as_tensor(rng.uniform(0.12, 1.25, n), dtype=torch.float32)
    cotangent = rng.normal(0.0, 0.7, (height, width, 3)).astype(np.float32)
    return {
        "field": field,
        "colors": field.colors.detach().clone(),
        "opacities": opacities,
        "cotangent": cotangent,
        "height": height,
        "width": width,
    }


def _build(fixture, *, use_opacities=True, support_fade=False, fade_alpha=None, eps=None):
    field = fixture["field"]
    kwargs = {}
    if eps is not None:
        kwargs["eps"] = eps
    return bridge_module.build_normalized_renderer_bridge(
        field.means,
        field.conics(),
        field.radii(3.0),
        fixture["opacities"] if use_opacities else None,
        fixture["height"],
        fixture["width"],
        support_fade=support_fade,
        sigma_cutoff=3.0,
        support_fade_alpha=fade_alpha,
        **kwargs,
    )


def test_bridge_uses_rounded_centers_clipped_rectangles_and_empty_outside_support():
    means = torch.tensor([[-0.6, 2.0], [5.6, -0.6], [10.0, 10.0]], dtype=torch.float32)
    conics = torch.tensor([[1.0, 0.0, 1.0]] * 3, dtype=torch.float32)
    radii = torch.tensor([[2, 1], [1, 2], [1, 1]], dtype=torch.long)
    built = bridge_module.build_normalized_renderer_bridge(
        means,
        conics,
        radii,
        None,
        5,
        6,
    )

    assert built.linear_map.footprint(0).pixel_indices.tolist() == [6, 7, 12, 13, 18, 19]
    assert built.linear_map.footprint(1).pixel_indices.tolist() == [5, 11]
    assert built.linear_map.footprint(2).pixel_indices.tolist() == []
    assert built.raw_support_elements == 8
    assert built.nonzero_elements == 8


@pytest.mark.parametrize(
    "use_opacities,support_fade,fade_alpha",
    [
        (False, False, None),
        (True, False, None),
        (True, True, None),
        (True, False, 0.35),
    ],
)
def test_sparse_map_matches_cpu_normalized_forward_and_row_sum_identity(
    use_opacities,
    support_fade,
    fade_alpha,
):
    fixture = _synthetic_geometry()
    built = _build(
        fixture,
        use_opacities=use_opacities,
        support_fade=support_fade,
        fade_alpha=fade_alpha,
    )
    report = bridge_module.validate_cpu_forward(built, fixture["colors"])

    assert report.passed
    assert report.max_absolute_error <= bridge_module.CPU_FORWARD_ATOL
    row_sum = built.linear_map.apply(np.ones((built.linear_map.column_count, 1), dtype=np.float64))[
        :, 0
    ]
    expected = built.denominator / (built.denominator + built.eps)
    assert np.allclose(row_sum, expected, atol=3e-7, rtol=3e-7)


@pytest.mark.parametrize("support_fade,fade_alpha", [(False, None), (True, None), (False, 0.4)])
def test_sparse_transpose_matches_cpu_color_autograd(support_fade, fade_alpha):
    fixture = _synthetic_geometry(seed=44)
    built = _build(
        fixture,
        support_fade=support_fade,
        fade_alpha=fade_alpha,
    )
    report = bridge_module.validate_cpu_color_transpose(
        built,
        fixture["colors"],
        fixture["cotangent"],
    )

    assert report.passed
    assert report.max_absolute_error <= bridge_module.CPU_TRANSPOSE_ATOL


def test_overlap_graph_and_off_diagonal_energy_find_only_shared_support_pair():
    means = torch.tensor([[1.0, 1.0], [1.4, 1.0], [5.0, 5.0]], dtype=torch.float32)
    conics = torch.tensor([[1.0, 0.0, 1.0]] * 3, dtype=torch.float32)
    radii = torch.tensor([[1, 1], [1, 1], [0, 0]], dtype=torch.long)
    built = bridge_module.build_normalized_renderer_bridge(
        means,
        conics,
        radii,
        torch.tensor([0.8, 0.6, 1.0]),
        6,
        6,
    )
    diagnostics = bridge_module.overlap_diagnostics(built)

    assert diagnostics.overlap_edges == ((0, 1),)
    assert diagnostics.degrees == (1, 1, 0)
    assert diagnostics.max_degree == 1
    assert diagnostics.nonzero_gram_edges == 1
    assert diagnostics.off_diagonal_frobenius_sq > 0.0
    assert 0.0 < diagnostics.off_diagonal_energy_fraction < 1.0
    assert 0.0 < diagnostics.max_absolute_correlation <= 1.0 + 1e-12


def test_custom_eps_is_constructed_but_cannot_be_mislabeled_cpu_renderer_parity():
    fixture = _synthetic_geometry(seed=9, n=5, height=7, width=8)
    built = _build(fixture, eps=1e-4)
    row_sum = built.linear_map.apply(np.ones((built.linear_map.column_count, 1), dtype=np.float64))[
        :, 0
    ]
    expected = built.denominator / (built.denominator + 1e-4)
    assert np.allclose(row_sum, expected, atol=3e-7, rtol=3e-7)
    with pytest.raises(ValueError, match="fixed eps=1e-8"):
        bridge_module.validate_cpu_forward(built, fixture["colors"])


def test_bridge_clamp_accounting_reports_linear_outputs_outside_display_range():
    means = torch.tensor([[1.0, 1.0], [2.0, 1.0]], dtype=torch.float32)
    conics = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=torch.float32)
    radii = torch.tensor([[1, 1], [1, 1]], dtype=torch.long)
    built = bridge_module.build_normalized_renderer_bridge(
        means,
        conics,
        radii,
        None,
        4,
        4,
    )
    accounting = built.clamp_accounting(
        np.asarray([[-0.4, 0.2, 0.3], [1.6, 1.4, 0.5]], dtype=np.float64)
    )
    assert accounting.warning
    assert accounting.lower_count > 0
    assert accounting.upper_count > 0
    assert accounting.active_count == accounting.lower_count + accounting.upper_count
    assert accounting.squared_clip_change > 0.0


def test_current_cuda_comparison_is_always_explicitly_nongating():
    fixture = _synthetic_geometry(seed=71, n=6, height=8, width=9)
    built = _build(fixture, support_fade=True)
    reports = bridge_module.current_cuda_nongating_diagnostics(
        built,
        fixture["colors"],
        modes=("cuda",),
    )

    assert len(reports) == 1
    assert reports[0].mode == "cuda"
    assert reports[0].gating is False
    if reports[0].available:
        assert reports[0].reason is None
        assert reports[0].parity is not None
        assert np.isfinite(reports[0].parity.max_absolute_error)
    else:
        assert reports[0].reason
        assert reports[0].parity is None
