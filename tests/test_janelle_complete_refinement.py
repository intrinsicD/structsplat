from pathlib import Path
from types import SimpleNamespace
import sys

import torch

from structsplat.gaussians import GaussianField


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deprecated_scripts"))
import fit_janelle_complete_refinement as runner  # noqa: E402


class _AllInsideConstraint:
    def __init__(self, height: int, width: int):
        self.eroded_flat = torch.ones(height * width, dtype=torch.bool)

    def apply(self, field, cfg, refresh=False):
        return field


class _TightScaleConstraint(_AllInsideConstraint):
    def apply(self, field, cfg, refresh=False):
        field.log_scales.fill_(torch.log(torch.tensor(0.4)).item())
        return field


def _fit_config():
    return SimpleNamespace(
        densify_max_axis_ratio=6.0,
        densify_coherence_power=1.0,
        split_scale=0.35,
        relocate_init_opacity=0.05,
    )


def test_merge_is_count_neutral_midpoint_enlargement_plus_batched_teleport():
    height = width = 12
    field = GaussianField.from_numpy(
        means=[[2.0, 2.0], [3.0, 2.0], [8.0, 1.0], [8.0, 4.0]],
        scales=[[1.0, 1.0]] * 4,
        angles=[0.0] * 4,
        # The second mutual-nearest pair fails the color-similarity gate.
        colors=[[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    )
    target = torch.zeros(height, width, 3)
    target[9, 10] = 1.0
    render = torch.zeros_like(target)
    mask = torch.ones(height, width, dtype=torch.bool)
    rewritten, stats = runner.merge_redundant(
        field,
        1,
        _AllInsideConstraint(height, width),
        _fit_config(),
        target,
        render,
        mask,
    )

    assert rewritten.n == field.n == 4
    assert stats["merged"] == stats["teleported"] == 1
    assert stats["count_conserved"] is True
    # The lower-index row owns the merged representation at the exact pair midpoint.
    torch.testing.assert_close(rewritten.means[0], torch.tensor([2.5, 2.0]))
    assert torch.all(rewritten.scales()[0] > field.scales()[0])
    # The other row is reused at the strongest residual site, rather than removed.
    torch.testing.assert_close(rewritten.means[1], torch.tensor([10.0, 9.0]))
    torch.testing.assert_close(rewritten.colors[1], target[9, 10])


def test_merge_envelopes_unequal_translated_covariances():
    height = width = 12
    field = GaussianField.from_numpy(
        means=[[2.0, 2.0], [3.0, 2.0], [8.0, 1.0], [8.0, 4.0]],
        scales=[[1.5, 0.7], [0.8, 1.2], [1.0, 1.0], [1.0, 1.0]],
        angles=[0.2, -0.3, 0.0, 0.0],
        colors=[[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    )
    target = torch.zeros(height, width, 3)
    target[9, 10] = 1.0
    mask = torch.ones(height, width, dtype=torch.bool)

    rewritten, stats = runner.merge_redundant(
        field,
        1,
        _AllInsideConstraint(height, width),
        _fit_config(),
        target,
        torch.zeros_like(target),
        mask,
    )

    assert stats["merged"] == 1
    midpoint = 0.5 * (field.means[0] + field.means[1])
    input_covariances = runner._covariances(field)
    translated = []
    for index in (0, 1):
        offset = field.means[index] - midpoint
        translated.append(input_covariances[index] + offset[:, None] * offset[None, :])
    merged_covariance = runner._covariances(rewritten)[0]
    values, vectors = torch.linalg.eigh(merged_covariance)
    inverse_sqrt = vectors @ torch.diag(values.rsqrt()) @ vectors.T
    worst_ratio = max(
        float(torch.linalg.eigvalsh(inverse_sqrt @ covariance @ inverse_sqrt)[-1])
        for covariance in translated
    )
    assert worst_ratio <= 1.0001


def test_merge_rejects_pair_when_containment_undoes_envelope():
    height = width = 12
    field = GaussianField.from_numpy(
        means=[[2.0, 2.0], [3.0, 2.0], [8.0, 1.0], [8.0, 4.0]],
        scales=[[1.0, 1.0]] * 4,
        angles=[0.0] * 4,
        colors=[[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    )
    target = torch.zeros(height, width, 3)
    target[9, 10] = 1.0
    mask = torch.ones(height, width, dtype=torch.bool)

    rewritten, stats = runner.merge_redundant(
        field,
        1,
        _TightScaleConstraint(height, width),
        _fit_config(),
        target,
        torch.zeros_like(target),
        mask,
    )

    assert rewritten is field
    assert stats["selected_before_containment"] == 1
    assert stats["containment_rejected"] == 1
    assert stats["merged"] == stats["teleported"] == 0
    assert stats["count_conserved"] is True
