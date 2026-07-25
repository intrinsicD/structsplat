#!/usr/bin/env bash
# Lint + CPU tests + docs<->code sync. Run before every commit; CI mirrors these steps.
#
# `ruff check` enforces a correctness baseline (E9 syntax + F pyflakes; see the pinned
# `select` in pyproject.toml) that the whole tree already passes. Broader style/import rules
# and `ruff format --check` are deliberately NOT in the gate yet: most of the tree predates
# them and adopting them is a separate, repo-wide ratchet (DOCS-003 follow-up), not this spine.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== ruff check =="
ruff check src tests scripts benchmarks

echo "== pytest (portable gate: not slow, not integration) =="
python -m pytest -q -m "not slow and not integration"

echo "== docs_sync =="
python scripts/docs_sync.py

echo "== ara claim ledger =="
python scripts/check_ara.py

echo "== task policy =="
python scripts/check_task_policy.py

echo "== script layout =="
python scripts/check_script_layout.py

echo "verify: OK"
