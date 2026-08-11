from __future__ import annotations

import numpy as np
import pytest
import torch

from structsplat.config import FitConfig
from structsplat.fit import _render
from structsplat.gaussians import GaussianField
from structsplat.normalized_refinement import (
    NormalizedTailRefinementConfig,
    refine_normalized_color_tail,
)


def _fixture() -> tuple[GaussianField, np.ndarray, np.ndarray, FitConfig]:
    means = np.asarray([[1.0, 1.0], [5.0, 1.0], [1.0, 5.0], [5.0, 5.0]])
    scales = np.full((4, 2), 2.0)
    rotations = np.zeros(4)
    input_colors = np.asarray(
        [[0.1, 0.2, 0.3], [0.8, 0.1, 0.2], [0.2, 0.8, 0.1], [0.1, 0.2, 0.8]]
    )
    target_colors = np.asarray(
        [[0.4, 0.2, 0.3], [0.8, 0.4, 0.2], [0.2, 0.8, 0.4], [0.4, 0.2, 0.8]]
    )
    field = GaussianField.from_numpy(
        means,
        scales,
        rotations,
        input_colors,
        opacities=np.asarray([2.0, 1.0, 0.0, -1.0]),
        scale_max=np.full((4, 2), 3.0),
        background_mask=np.asarray([True, False, False, False]),
        filter_variance=np.asarray([0.1, 0.2, 0.3, 0.4]),
    )
    target_field = GaussianField.from_numpy(
        means,
        scales,
        rotations,
        target_colors,
        opacities=np.asarray([2.0, 1.0, 0.0, -1.0]),
        scale_max=np.full((4, 2), 3.0),
        background_mask=np.asarray([True, False, False, False]),
        filter_variance=np.asarray([0.1, 0.2, 0.3, 0.4]),
    )
    render_config = FitConfig(iters=1, renderer="normalized", render_chunk=1)
    with torch.no_grad():
        target = _render(target_field, render_config, 7, 7).numpy()
    return field, target, np.ones((7, 7), dtype=bool), render_config


def test_tail_refinement_changes_only_colors_and_is_deterministic() -> None:
    field, target, mask, render_config = _fixture()
    original = field.detached()
    config = NormalizedTailRefinementConfig(
        steps=80,
        checkpoint_every=5,
        learning_rate=0.05,
        tail_fraction=0.2,
        tail_weight=4.0,
        max_color_shift=0.5,
        color_abs_limit=2.0,
    )

    first = refine_normalized_color_tail(
        field,
        target,
        mask,
        render_config,
        config=config,
    )
    second = refine_normalized_color_tail(
        field,
        target,
        mask,
        render_config,
        config=config,
    )

    assert first.selected_step > 0
    assert first.final_sse < first.initial_sse
    assert first.final_display_pixel_rmse_max <= first.initial_display_pixel_rmse_max
    assert first.final_display_patch7_rmse_max <= first.initial_display_patch7_rmse_max
    assert first.field.n == field.n
    assert first.non_color_arrays_bit_exact
    assert first.color_shift_max <= config.max_color_shift + 1e-7
    assert first.color_abs_max <= config.color_abs_limit + 1e-7
    assert first.maintained_render_parity_max_abs < 1e-6
    assert torch.equal(field.colors, original.colors)
    assert torch.equal(first.field.colors, second.field.colors)
    assert np.array_equal(first.reconstruction_raw, second.reconstruction_raw)
    assert all(
        checkpoint.eligible
        for checkpoint in first.checkpoints
        if checkpoint.selected
    )


def test_tail_refinement_returns_exact_step_zero_when_target_already_matches() -> None:
    field, _target, mask, render_config = _fixture()
    with torch.no_grad():
        target = _render(field, render_config, 7, 7).numpy()

    result = refine_normalized_color_tail(
        field,
        target,
        mask,
        render_config,
        config=NormalizedTailRefinementConfig(steps=10, checkpoint_every=2),
    )

    assert result.selected_step == 0
    assert torch.equal(result.field.colors, field.colors)
    assert np.array_equal(result.reconstruction_raw, target)


def test_tail_refinement_rejects_additive_semantics() -> None:
    field, target, mask, _render_config = _fixture()

    with pytest.raises(ValueError, match="normalized"):
        refine_normalized_color_tail(
            field,
            target,
            mask,
            FitConfig(iters=1, renderer="additive", render_chunk=1),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"steps": 0}, "steps"),
        ({"checkpoint_every": 0}, "checkpoint_every"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"tail_fraction": 0.0}, "tail_fraction"),
        ({"tail_fraction": 1.1}, "tail_fraction"),
        ({"tail_weight": 0.0}, "tail_weight"),
        ({"max_color_shift": 0.0}, "max_color_shift"),
        ({"color_abs_limit": 0.0}, "color_abs_limit"),
        ({"sse_relative_tolerance": -1.0}, "sse_relative_tolerance"),
    ],
)
def test_tail_refinement_config_fails_closed(kwargs, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        NormalizedTailRefinementConfig(**kwargs)
