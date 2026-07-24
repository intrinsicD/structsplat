from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deprecated_scripts"))
import fit_janelle_safe_commit_schedule as runner  # noqa: E402
import run_janelle_safe_schedule_factorial as factorial  # noqa: E402
import run_janelle_detail_tail_ablation as detail_tail  # noqa: E402


def test_default_schedule_matches_5000_8000_10000_11000_plan():
    args = runner.build_parser().parse_args([])
    schedule = runner.build_schedule(args)
    schedule.validate(5_000)
    assert schedule.capacity == 11_000
    assert schedule.storage_policy == "dynamic"
    assert schedule.base_active_limit == 11_000
    assert schedule.detail_tail_max_rows == 0
    assert schedule.detail_tail_batch_rows == 128
    assert schedule.detail_tail_min_gain_per_row == 0.0
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
    assert schedule.pareto_safe_checkpoints is False
    assert schedule.pareto_checkpoint_every == 50
    assert schedule.event_color_solve is False
    assert schedule.tolerances.guard_p99 is False


def test_fixed_storage_can_reserve_rows_for_the_late_detail_tail():
    args = runner.build_parser().parse_args([
        "--storage-policy", "fixed_capacity",
        "--capacity", "12024",
        "--base-active-limit", "11000",
        "--detail-tail-rows", "512",
        "--detail-tail-batch", "128",
        "--detail-tail-min-gain-per-row", "1e-7",
    ])
    schedule = runner.build_schedule(args)
    schedule.validate(5_000)

    assert schedule.capacity == 12_024
    assert schedule.resolved_base_active_limit() == 11_000
    assert schedule.detail_tail_max_rows == 512
    assert schedule.detail_tail_batch_rows == 128
    assert schedule.detail_tail_min_gain_per_row == 1e-7
    assert schedule.boundary.target_gaussians == 11_000
    assert schedule.redistribution.target_gaussians == 11_000
    assert schedule.polish.target_gaussians == 11_512


def test_local_boundary_variant_is_explicitly_configurable():
    args = runner.build_parser().parse_args([
        "--storage-policy", "fixed_capacity",
        "--refinement-policy", "local_neighborhood",
        "--local-start-phase", "boundary",
        "--local-seed-count", "256",
        "--local-neighbor-count", "6",
        "--topology-neighbor-count", "10",
        "--boundary-recycle-at-capacity",
        "--pareto-safe-checkpoints",
        "--pareto-checkpoint-every", "25",
        "--event-color-solve",
        "--guard-p99",
        "--boundary-residual-mse-threshold", "0.005",
    ])
    schedule = runner.build_schedule(args)
    schedule.validate(5_000)
    assert schedule.storage_policy == "fixed_capacity"
    assert schedule.refinement_policy == "local_neighborhood"
    assert schedule.local_start_phase == "boundary"
    assert schedule.local_seed_count == 256
    assert schedule.local_neighbor_count == 6
    assert schedule.topology_neighbor_count == 10
    assert schedule.boundary_recycle_at_capacity is True
    assert schedule.pareto_safe_checkpoints is True
    assert schedule.pareto_checkpoint_every == 25
    assert schedule.event_color_solve is True
    assert schedule.tolerances.guard_p99 is True
    assert schedule.boundary_residual_mse_threshold == 0.005


def test_runner_help_exposes_phase_and_event_controls():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "deprecated_scripts/fit_janelle_safe_commit_schedule.py"),
            "--help",
        ],
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
        "--storage-policy",
        "--base-active-limit",
        "--detail-tail-rows",
        "--detail-tail-batch",
        "--detail-tail-min-gain-per-row",
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
        "--pareto-safe-checkpoints",
        "--pareto-checkpoint-every",
        "--event-color-solve",
        "--guard-p99",
        "--boundary-residual-mse-threshold",
        "--coverage-birth-count",
        "--detail-split-count",
        "--redistribution-count",
    ):
        assert flag in result.stdout


def test_factorial_runner_builds_the_four_isolated_arm_commands(tmp_path):
    args = factorial.build_parser().parse_args([
        "--out", str(tmp_path),
        "--pareto-checkpoint-every", "25",
    ])
    commands = {
        name: factorial._arm_command(args, name, tmp_path / name)
        for name in factorial.ARMS
    }
    assert "--pareto-safe-checkpoints" not in commands["control"]
    assert "--event-color-solve" not in commands["control"]
    assert "--pareto-safe-checkpoints" in commands["pareto_checkpoint"]
    assert "--event-color-solve" not in commands["pareto_checkpoint"]
    assert "--pareto-safe-checkpoints" not in commands["event_color_solve"]
    assert "--event-color-solve" in commands["event_color_solve"]
    assert "--pareto-safe-checkpoints" in commands["combined"]
    assert "--event-color-solve" in commands["combined"]
    assert all("--no-archive" in command for command in commands.values())
    assert all(
        command[command.index("--pareto-checkpoint-every") + 1] == "25"
        for command in commands.values()
    )


def test_detail_tail_runner_builds_three_matched_storage_commands(tmp_path):
    args = detail_tail.build_parser().parse_args(["--out", str(tmp_path)])
    commands = {
        name: detail_tail._arm_command(args, name, tmp_path / name)
        for name in detail_tail.ARMS
    }

    def value(command, flag):
        return command[command.index(flag) + 1]

    assert all(value(command, "--capacity") == "12024" for command in commands.values())
    assert all(
        value(command, "--storage-policy") == "fixed_capacity"
        for command in commands.values()
    )
    assert value(commands["baseline_11000"], "--base-active-limit") == "11000"
    assert value(commands["baseline_11000"], "--detail-tail-rows") == "0"
    assert value(commands["generic_plus512"], "--base-active-limit") == "11512"
    assert value(commands["generic_plus512"], "--detail-tail-rows") == "0"
    assert (
        value(commands["adaptive_tail_plus512"], "--base-active-limit")
        == "11000"
    )
    assert (
        value(commands["adaptive_tail_plus512"], "--detail-tail-rows")
        == "512"
    )
    assert all("--pareto-safe-checkpoints" in command for command in commands.values())
    assert all("--no-archive" in command for command in commands.values())
