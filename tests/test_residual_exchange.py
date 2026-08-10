from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from structsplat.overlap_elimination import lattice_observation_field
from structsplat.pixel_contraction import render_observation_field
from structsplat.residual_exchange import (
    ResidualExchangeConfig,
    _field_supports,
    _row_prices,
    exchange_residual_columns,
)


def _exchange_fixture():
    mask = np.ones((15, 15), dtype=bool)
    active = np.zeros(mask.shape, dtype=bool)
    active[2, 2] = True
    active[12, 12] = True
    field = lattice_observation_field(
        mask,
        active,
        np.asarray([[0.25, 0.10, 0.40], [0.0, 0.0, 0.0]], dtype=np.float32),
        scale_px=0.45,
        sigma_cutoff=3.0,
    )
    desired = np.zeros(mask.shape, dtype=bool)
    desired[7, 7] = True
    residual_field = lattice_observation_field(
        mask,
        desired,
        np.asarray([[0.30, -0.12, 0.18]], dtype=np.float32),
        scale_px=0.45,
        sigma_cutoff=3.0,
    )
    target = render_observation_field(field) + render_observation_field(residual_field)
    return field, target, mask


def test_row_removal_prices_match_direct_reconstruction():
    field, target, mask = _exchange_fixture()
    reconstruction = render_observation_field(field)
    residual = target.astype(np.float64) - reconstruction.astype(np.float64)
    prices = _row_prices(_field_supports(field, mask), field.rgb_coeff, residual)
    initial_sse = float(np.sum((reconstruction.astype(np.float64) - target) ** 2))

    for row in range(field.n):
        coefficients = np.array(field.rgb_coeff, copy=True)
        coefficients[row] = 0.0
        deleted = render_observation_field(replace(field, rgb_coeff=coefficients))
        direct_delta = float(np.sum((deleted.astype(np.float64) - target) ** 2)) - initial_sse
        assert prices[row] == pytest.approx(direct_delta, abs=1e-7)


def test_exchange_recovers_residual_atom_at_exact_count_and_freezes_other_rows():
    field, target, mask = _exchange_fixture()
    result = exchange_residual_columns(
        field,
        target,
        mask,
        config=ResidualExchangeConfig(
            candidate_shapes=((0.45, 0.45, 0.0),),
            max_exchanges=1,
            site_count=4,
            donor_count=2,
            proposal_frontier=4,
        ),
        render_chunk=1,
    )

    assert result.field.n == field.n == 2
    assert result.accepted_exchanges == 1
    assert result.final_sse < 1e-10
    assert result.final_sse < result.initial_sse
    replaced = int(np.flatnonzero(result.replaced_row_mask)[0])
    frozen = 1 - replaced
    assert np.array_equal(result.field.means_xy[frozen], field.means_xy[frozen])
    assert np.array_equal(result.field.log_scales_xy[frozen], field.log_scales_xy[frozen])
    assert np.array_equal(result.field.rotations_rad[frozen], field.rotations_rad[frozen])
    assert np.array_equal(result.field.rgb_coeff[frozen], field.rgb_coeff[frozen])
    assert np.array_equal(result.field.means_xy[replaced], [7.0, 7.0])
    assert result.maintained_render_parity_max_abs < 2e-6
    assert result.repeated_render_parity_max_abs < 2e-6
    assert not result.reconstruction.flags.writeable
    assert not result.replaced_row_mask.flags.writeable


def test_exchange_rejects_global_gain_that_worsens_displayed_pixel_max():
    mask = np.ones((17, 17), dtype=bool)
    donor_mask = np.zeros(mask.shape, dtype=bool)
    donor_mask[2, 2] = True
    field = lattice_observation_field(
        mask,
        donor_mask,
        np.asarray([[0.40, 0.40, 0.40]], dtype=np.float32),
        scale_px=0.18,
        sigma_cutoff=3.0,
    )
    candidate_mask = np.zeros(mask.shape, dtype=bool)
    candidate_mask[12, 12] = True
    candidate = lattice_observation_field(
        mask,
        candidate_mask,
        np.asarray([[0.30, 0.30, 0.30]], dtype=np.float32),
        scale_px=1.0,
        sigma_cutoff=3.0,
    )
    target = render_observation_field(field) + render_observation_field(candidate)

    result = exchange_residual_columns(
        field,
        target,
        mask,
        config=ResidualExchangeConfig(
            candidate_shapes=((1.0, 1.0, 0.0),),
            max_exchanges=1,
            site_count=1,
            donor_count=1,
            proposal_frontier=1,
        ),
        render_chunk=1,
    )

    assert result.stop_reason == "no_cold_safe_pair"
    assert result.accepted_exchanges == 0
    assert result.field.canonical_hash() == field.canonical_hash()
    assert result.final_sse == pytest.approx(result.initial_sse)
    assert result.cold_rendered_pairs == 1


def test_exchange_cpu_replay_is_deterministic_and_locks_replaced_rows():
    field, target, mask = _exchange_fixture()
    config = ResidualExchangeConfig(
        candidate_shapes=((0.30, 0.30, 0.0), (0.45, 0.45, 0.0)),
        max_exchanges=2,
        site_count=6,
        donor_count=2,
        proposal_frontier=4,
    )
    first = exchange_residual_columns(field, target, mask, config=config, render_chunk=1)
    second = exchange_residual_columns(field, target, mask, config=config, render_chunk=1)

    assert first.field.canonical_hash() == second.field.canonical_hash()
    first_records = first.checkpoint_records()
    second_records = second.checkpoint_records()
    for record in (*first_records, *second_records):
        record.pop("elapsed_seconds")
    assert first_records == second_records
    assert first.accepted_exchanges == int(first.replaced_row_mask.sum())
    row_ids = [checkpoint.row_index for checkpoint in first.checkpoints]
    assert len(row_ids) == len(set(row_ids))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"candidate_shapes": ()}, "candidate_shapes"),
        ({"candidate_shapes": ((0.0, 1.0, 0.0),)}, "candidate_shapes"),
        ({"max_exchanges": 0}, "max_exchanges"),
        ({"proposal_frontier": 0}, "proposal_frontier"),
        ({"coefficient_abs_limit": 0.0}, "coefficient_abs_limit"),
        ({"minimum_sse_gain": -1.0}, "minimum_sse_gain"),
    ],
)
def test_exchange_config_fails_closed(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        ResidualExchangeConfig(**kwargs)


def test_import_does_not_load_torch():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import structsplat.residual_exchange; "
            "raise SystemExit(1 if 'torch' in sys.modules else 0)",
        ],
        check=False,
        cwd=root,
        env=environment,
        timeout=30,
    )
    assert completed.returncode == 0
