#!/usr/bin/env python3
"""Keep ``scripts/`` durable and one-off experiment scripts in ``scripts/experiments/``.

Without this gate, task-specific drivers accumulate at the top level of ``scripts/`` until an
agent cannot tell repository tooling from a spent protocol runner. The rule is an allowlist:
every top-level file in ``scripts/`` must be declared durable here, with a reason. Anything else
belongs in ``scripts/experiments/`` (see that directory's README).

Torch-free by design, like ``scripts/docs_sync.py``.

Run: python scripts/check_script_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

# name -> why it is durable repository tooling rather than a one-off experiment driver.
DURABLE_SCRIPTS: dict[str, str] = {
    "verify.sh": "the repository verification gate; CI runs the same sequence",
    "docs_sync.py": "structural docs<->code checker, part of verify.sh",
    "check_ara.py": "ara/ claim-ledger structural checker, part of verify.sh",
    "check_task_policy.py": "tasks/ tree and INDEX.md checker, part of verify.sh",
    "check_script_layout.py": "this checker, part of verify.sh",
    "install_skills.sh": "symlinks project skills for global agent discovery",
    "run_ablation.sh": "the standing ablation entrypoint referenced by the benchmark skill",
    "render_paper_figures.py": "reusable figure renderer for publication artifacts",
    "run_with_resources.py": "reusable resource-capped runner used by several lanes",
    # -- Grandfathered task drivers. All are referenced from README.md, tests/, and
    # ara/evidence/*/run.md command records; the Janelle scripts are additionally copied
    # verbatim into runs/*/provenance/ as the source binding for committed evidence.
    # Moving them would invalidate those records. See scripts/experiments/README.md.
    "fit_janelle_complete_refinement.py": "PINNED: provenance-bound (FIT-021..026 lineage)",
    "fit_janelle_mask_contained.py": "PINNED: provenance-bound (CORE-010/011 lineage)",
    "fit_janelle_safe_commit_schedule.py": "PINNED: provenance-bound (FIT-023 lineage)",
    "compare_janelle_pooled_boundary_variants.py": "PINNED: provenance-bound (FIT-021 lineage)",
    "compare_janelle_safe_schedule_variants.py": "PINNED: provenance-bound (FIT-023 lineage)",
    "run_janelle_detail_tail_ablation.py": "PINNED: provenance-bound (FIT-025 lineage)",
    "run_janelle_safe_schedule_factorial.py": "PINNED: provenance-bound (FIT-023/024 lineage)",
    "prepare_abl004_images.py": "PINNED: ABL-004 image preparation bound by evidence commands",
    "run_abl004_full_ablation.sh": "PINNED: ABL-004 lane bound by evidence commands",
    "run_abl005_affine_quality_influence.sh": "PINNED: ABL-005 lane bound by evidence commands",
    "run_abl005_cuda_native_influence.sh": "PINNED: ABL-005 lane bound by evidence commands",
    "run_stage_influence.sh": "PINNED: ABL-002 stage lane bound by evidence commands",
    "run_stage_search_screening.sh": "PINNED: ABL-002 screening lane bound by evidence commands",
    "run_storage_budget_168k.sh": "PINNED: BENCH-006 168 KiB lane bound by evidence commands",
    "setup_native_gaussianimage_env.sh": "PINNED: BENCH-005 native reference env bootstrap",
    "setup_native_image_gs_env.sh": "PINNED: BENCH-005 native reference env bootstrap",
}


def main() -> int:
    if not SCRIPTS.is_dir():
        print("check_script_layout: missing scripts/ directory", file=sys.stderr)
        return 1

    errors: list[str] = []

    for path in sorted(SCRIPTS.iterdir()):
        if path.is_dir() or path.name.startswith("."):
            continue
        if path.name not in DURABLE_SCRIPTS:
            errors.append(
                f"scripts/{path.name} is not declared durable. Move it to scripts/experiments/ "
                "(see scripts/experiments/README.md), or add it to DURABLE_SCRIPTS in "
                "scripts/check_script_layout.py with a reason."
            )

    for name in sorted(DURABLE_SCRIPTS):
        if not (SCRIPTS / name).is_file():
            errors.append(
                f"DURABLE_SCRIPTS lists scripts/{name} but that file does not exist "
                "(remove the stale allowlist entry)"
            )

    experiments = SCRIPTS / "experiments"
    if not experiments.is_dir():
        errors.append("missing scripts/experiments/ directory")
    elif not (experiments / "README.md").is_file():
        errors.append("missing scripts/experiments/README.md (the layout policy lives there)")

    if errors:
        print(f"check_script_layout: {len(errors)} problem(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"check_script_layout: OK ({len(DURABLE_SCRIPTS)} durable scripts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
