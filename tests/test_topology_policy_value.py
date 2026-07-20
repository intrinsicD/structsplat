import numpy as np
import torch

from benchmarks import topology_policy_value as B
from benchmarks.marginal_cold_stream_rd import _field_sha256
from structsplat.fit import _make_optimizer
from structsplat.gaussians import GaussianField


def _candidate(x: float, y: float, scale: float = 5.0) -> GaussianField:
    return GaussianField.from_numpy(
        np.asarray([[x, y]], dtype=np.float32),
        np.asarray([[scale, scale]], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32),
    )


def test_initial_field_is_deterministic_target_independent_and_seed_distinct():
    first = B.initial_field(0)
    repeated = B.initial_field(0)
    second_seed = B.initial_field(1)

    assert first.n == 64
    assert first.opacities is None
    assert _field_sha256(first) == _field_sha256(repeated)
    assert _field_sha256(first) != _field_sha256(second_seed)


def test_replacement_carries_survivor_adam_state_and_zeros_new_row():
    torch.set_num_threads(1)
    field = B.initial_field(0).trainable()
    cfg = B._fit_config()
    optimizer = _make_optimizer(field, cfg)
    target = torch.zeros(B.SIZE, B.SIZE, 3)
    B._advance(field, optimizer, target, cfg, start_step=0, end_step=1)
    parent_field_hash = _field_sha256(field)
    parent_optimizer_hash = B._optimizer_sha256(optimizer)

    child, child_optimizer, continuation = B._replacement_with_state(
        field, optimizer, _candidate(32.0, 32.0), 7, cfg
    )

    assert child.n == 64
    assert continuation["survivor_state_exact"] is True
    assert continuation["new_row_zero_moments"] is True
    assert _field_sha256(field) == parent_field_hash
    assert B._optimizer_sha256(optimizer) == parent_optimizer_hash
    for group in child_optimizer.param_groups:
        parameter = group["params"][0]
        state = child_optimizer.state[parameter]
        assert torch.count_nonzero(state["exp_avg"][-1]) == 0
        assert torch.count_nonzero(state["exp_avg_sq"][-1]) == 0


def test_no_action_clone_preserves_field_and_optimizer_state_exactly():
    field = B.initial_field(1).trainable()
    cfg = B._fit_config()
    optimizer = _make_optimizer(field, cfg)
    target = torch.ones(B.SIZE, B.SIZE, 3)
    B._advance(field, optimizer, target, cfg, start_step=0, end_step=1)

    child, child_optimizer = B._clone_no_action(field, optimizer, cfg)

    assert _field_sha256(child) == _field_sha256(field)
    assert B._optimizer_sha256(child_optimizer) == B._optimizer_sha256(optimizer)


def test_feasible_candidates_require_untruncated_modal_work():
    cfg = B._fit_config()
    candidates = [_candidate(20.0 + index % 4, 20.0 + index // 4) for index in range(15)]
    candidates.append(_candidate(1.0, 1.0))

    feasible, work = B._feasible_candidates(candidates, cfg)

    assert feasible == list(range(15))
    assert all(work[index]["feasible"] for index in feasible)
    assert work[15]["untruncated"] is False
    assert work[15]["feasible"] is False
    assert len({work[index]["support_work_count"] for index in feasible}) == 1
