#!/bin/bash
# SessionStart hook: expose the generated work view in every environment. On Claude Code web,
# also create the venv and install the package (CPU torch) so tests/checkers work immediately.
# Idempotent — skips installation when the environment is already functional.
set -euo pipefail

cd "$CLAUDE_PROJECT_DIR"

echo "structsplat session work view:"
sed -n '1,80p' tasks/SESSION-BRIEF.md

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

if ! { .venv/bin/python -c "import structsplat, pytest" && .venv/bin/python -m ruff --version; } >/dev/null 2>&1; then
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install --quiet -e '.[dev,benchmark]'
fi

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi

.venv/bin/python -c "import torch, structsplat; print(f'structsplat env ready: torch {torch.__version__}')"
