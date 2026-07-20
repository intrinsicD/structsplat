# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from benchmarks.topk_responsibility_oracle import topk_responsibility_oracle
from structsplat.gaussians import GaussianField
from structsplat.render import render_field


def test_topk_oracle_full_matches_reference_and_saturates_at_field_count():
    field = GaussianField.from_numpy(
        means=[[1.2, 1.1], [4.2, 1.7], [2.8, 4.4]],
        scales=[[1.8, 1.1], [1.4, 2.0], [2.2, 1.3]],
        angles=[0.1, -0.4, 0.7],
        colors=[[0.9, 0.1, 0.2], [0.1, 0.8, 0.3], [0.2, 0.3, 0.95]],
        opacities=[1.5, -0.5, 0.25],
    )
    height, width = 7, 8
    result = topk_responsibility_oracle(
        field, height, width, ks=(1, 2, 4), tile_size=4
    )
    reference = render_field(
        field.means,
        field.conics(),
        field.colors,
        field.radii(3.0),
        height,
        width,
        mode="normalized",
        opacities=field.opacity_values(),
    )

    assert torch.allclose(result["full"], reference, atol=2e-7, rtol=2e-6)
    assert torch.allclose(result["topk"][4], result["full"], atol=2e-7, rtol=2e-6)
    assert result["retained_counts"][1] <= result["retained_counts"][2]
    assert result["retained_counts"][2] <= result["retained_counts"][4]
    assert result["retained_counts"][4] == result["positive_weight_contributions"]
    assert result["positive_weight_contributions"] <= result["rectangle_evaluations"]

    covered = result["full_denominator"] > 0
    masses = [
        result["topk_denominators"][k][covered] / result["full_denominator"][covered]
        for k in (1, 2, 4)
    ]
    assert torch.all(masses[0] <= masses[1] + 1e-7)
    assert torch.all(masses[1] <= masses[2] + 1e-7)
    assert torch.allclose(masses[2], torch.ones_like(masses[2]), atol=1e-6)
