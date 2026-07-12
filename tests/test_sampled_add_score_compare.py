# ruff: noqa: E402
import pytest

torch = pytest.importorskip("torch")

from benchmarks.sampled_add_score_compare import _aggregate, _decision


def test_fit017_aggregate_pairs_modes_and_applies_preregistered_guard():
    rows = []
    values = {
        "legacy_abs": (20.0, 21.0, 22.0, 10.0),
        "gaussian_abs": (20.2, 21.1, 22.0, 10.2),
        "signed_gaussian": (20.4, 21.2, 22.0, 11.0),
    }
    for image in ("a", "b"):
        for mode, (immediate, post20, post100, elapsed) in values.items():
            rows.append({
                "image": image,
                "seed": 0,
                "score_mode": mode,
                "immediate_psnr": immediate,
                "post20_psnr": post20,
                "post100_psnr": post100,
                "score_seconds": 0.1,
                "total100_seconds": elapsed,
            })

    aggregate = _aggregate(rows)
    decision = _decision(aggregate)

    assert aggregate["signed_gaussian"]["delta_post20_psnr"] == pytest.approx(0.2)
    assert aggregate["signed_gaussian"]["post20_positive_fraction"] == 1.0
    assert aggregate["signed_gaussian"]["total100_overhead_fraction"] == pytest.approx(0.1)
    assert decision["guard_survives"] is True
