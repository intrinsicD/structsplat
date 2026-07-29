#!/usr/bin/env bash
# Lint + CPU tests + repository structural gates. Run before every commit; CI mirrors these steps.
#
# `ruff check` enforces a correctness baseline (E9 syntax + F pyflakes; see the pinned
# `select` in pyproject.toml) that the whole tree already passes. Broader style/import rules
# and `ruff format --check` are deliberately NOT in the gate yet: most of the tree predates
# them and adopting them is a separate, repo-wide ratchet owned by DOCS-004.
set -euo pipefail
cd "$(dirname "$0")/.."

STRUCTSPLAT_PYTHON_BIN="${STRUCTSPLAT_PYTHON:-}"
if [[ -z "$STRUCTSPLAT_PYTHON_BIN" ]]; then
  if [[ -x .venv/bin/python ]] \
    && .venv/bin/python -c "import pytest, structsplat" >/dev/null 2>&1 \
    && .venv/bin/python -m ruff --version >/dev/null 2>&1; then
    STRUCTSPLAT_PYTHON_BIN=".venv/bin/python"
  else
    STRUCTSPLAT_PYTHON_BIN="python"
  fi
fi

echo "== ruff check =="
"$STRUCTSPLAT_PYTHON_BIN" -m ruff check src tests scripts benchmarks

echo "== pytest (portable gate: not slow, not integration) =="
"$STRUCTSPLAT_PYTHON_BIN" -m pytest -q -m "not slow and not integration"

echo "== docs_sync =="
"$STRUCTSPLAT_PYTHON_BIN" scripts/docs_sync.py

echo "== ara claim ledger =="
"$STRUCTSPLAT_PYTHON_BIN" scripts/check_ara.py

echo "== task policy =="
"$STRUCTSPLAT_PYTHON_BIN" scripts/check_task_policy.py

echo "== script layout =="
"$STRUCTSPLAT_PYTHON_BIN" scripts/check_script_layout.py

echo "== agent workflow =="
"$STRUCTSPLAT_PYTHON_BIN" scripts/check_agent_workflow.py

echo "verify: OK"
