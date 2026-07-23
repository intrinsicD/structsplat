"""Shared pytest configuration: deterministic RNG seeding for every test.

NumPy is seeded unconditionally because the init-time math (`structure_tensor`, `density`,
`sampling`) is NumPy and must stay importable without torch (core invariant 1). torch is seeded
only when it is already importable, so a NumPy-only test run never pulls torch in through this
conftest. The seed mirrors the repository default (`InitConfig.seed == 0`).
"""

from __future__ import annotations

import numpy as np
import pytest

_SEED = 0


@pytest.fixture(autouse=True)
def _seed_everything():
    """Reseed NumPy (always) and torch (when available) before each test."""
    np.random.seed(_SEED)
    try:
        import torch
    except ModuleNotFoundError:
        pass
    else:
        torch.manual_seed(_SEED)
    yield


# Environment/evidence-coupled test modules excluded from the default portable gate. They need
# committed results/ evidence bundles (gitignored, absent from a fresh checkout), a Landlock-capable
# kernel, or a specially configured worker environment (thread-pinning env vars, sandbox
# subprocesses). Run them with `-m integration` in a fully provisioned setup.
_INTEGRATION_MODULES = frozenset(
    {
        "test_ssp2e_replay_repair",
        "test_ssp2f_rdo_operator",
        "test_ssp2v_actual_run",
        "test_ssp2v_actual_run_preflight_policy",
        "test_ssp2v_actual_run_producer",
        "test_ssp2v_actual_run_recovery",
        "test_ssp2v_landlock",
        "test_ssp2v_quality",
        "test_local_linear_reproducing_full_assay",
        "test_native_sad_frontier",
        "test_gauge_free_covariance_assay",
    }
)


def pytest_collection_modifyitems(config, items):
    """Auto-mark the environment/evidence-coupled modules above as ``integration``."""
    for item in items:
        path = getattr(item, "path", None)
        if path is not None and path.stem in _INTEGRATION_MODULES:
            item.add_marker(pytest.mark.integration)
