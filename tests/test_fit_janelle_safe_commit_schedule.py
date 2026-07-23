from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fit_janelle_safe_commit_schedule as runner  # noqa: E402


def test_default_schedule_matches_5000_8000_10000_11000_plan():
    args = runner.build_parser().parse_args([])
    schedule = runner.build_schedule(args)
    schedule.validate(5_000)
    assert schedule.capacity == 11_000
    assert schedule.coverage_target_gaussians == 8_000
    assert schedule.detail_target_gaussians == 10_000
    assert schedule.bootstrap.target_gaussians == 5_000
    assert schedule.coverage.target_gaussians == 8_000
    assert schedule.detail.target_gaussians == 10_000
    assert schedule.boundary.target_gaussians == 11_000
    assert schedule.redistribution.target_gaussians == 11_000
    assert schedule.polish.target_gaussians == 11_000
    assert schedule.recovery_steps == 80
    assert schedule.refinement_policy == "global"
    assert schedule.local_start_phase == "coverage"
    assert schedule.boundary_recycle_at_capacity is False
    assert schedule.tolerances.guard_p99 is False


def test_local_boundary_variant_is_explicitly_configurable():
    args = runner.build_parser().parse_args([
        "--refinement-policy", "local_neighborhood",
        "--local-start-phase", "boundary",
        "--local-seed-count", "256",
        "--local-neighbor-count", "6",
        "--topology-neighbor-count", "10",
        "--boundary-recycle-at-capacity",
        "--guard-p99",
        "--boundary-residual-mse-threshold", "0.005",
    ])
    schedule = runner.build_schedule(args)
    schedule.validate(5_000)
    assert schedule.refinement_policy == "local_neighborhood"
    assert schedule.local_start_phase == "boundary"
    assert schedule.local_seed_count == 256
    assert schedule.local_neighbor_count == 6
    assert schedule.topology_neighbor_count == 10
    assert schedule.boundary_recycle_at_capacity is True
    assert schedule.tolerances.guard_p99 is True
    assert schedule.boundary_residual_mse_threshold == 0.005


def test_runner_help_exposes_phase_and_event_controls():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/fit_janelle_safe_commit_schedule.py"), "--help"],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    for flag in (
        "--bootstrap-steps",
        "--coverage-steps",
        "--detail-steps",
        "--boundary-steps",
        "--redistribution-steps",
        "--polish-steps",
        "--recovery-steps",
        "--refinement-policy",
        "--local-start-phase",
        "--local-seed-count",
        "--local-neighbor-count",
        "--topology-neighbor-count",
        "--boundary-recycle-at-capacity",
        "--guard-p99",
        "--boundary-residual-mse-threshold",
        "--coverage-birth-count",
        "--detail-split-count",
        "--redistribution-count",
    ):
        assert flag in result.stdout
