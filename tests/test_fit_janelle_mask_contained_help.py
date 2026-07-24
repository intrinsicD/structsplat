import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deprecated_scripts" / "fit_janelle_mask_contained.py"


def _help(*arguments: str, realtime_root: Path) -> str:
    environment = os.environ.copy()
    environment["STRUCTSPLAT_REALTIME_ROOT"] = str(realtime_root)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    return result.stdout


def test_help_is_available_without_runtime_dependencies(tmp_path):
    missing_realtime_root = tmp_path / "realtime-gs-is-deliberately-absent"

    top_level = _help("--help", realtime_root=missing_realtime_root)
    convert = _help("convert", "--help", realtime_root=missing_realtime_root)
    worker = _help("_worker", "--help", realtime_root=missing_realtime_root)

    assert "convert --help" in top_level
    assert "capacity and growth lifecycle" in convert
    assert "optimization, logging, and stopping" in convert
    assert "mask containment and boundary coverage" in convert
    assert "does not expose fit-time" in convert
    assert "foreground support activity" in convert
    assert "internal view selection" in worker


def test_every_explicit_cli_parameter_has_explanatory_help():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    missing = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        names = [
            value.value
            for value in node.args
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        ]
        help_values = [keyword.value for keyword in node.keywords if keyword.arg == "help"]
        if not help_values:
            missing.extend(names)
            continue
        value = help_values[0]
        if isinstance(value, ast.Constant) and (
            not isinstance(value.value, str) or not value.value.strip()
        ):
            missing.extend(names)

    assert missing == []
