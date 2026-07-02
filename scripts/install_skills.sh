#!/usr/bin/env bash
# Symlink StructSplat's project skills into ~/.claude/skills so they're available in every project
# (matching the IntrinsicEngine symlink-based layout). Project-scoped copies in ./.claude/skills are
# already picked up automatically when Claude Code runs inside this repo; this is only for global use.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${HOME}/.claude/skills"
mkdir -p "$DEST"
for d in "$REPO"/.claude/skills/*/; do
  name="$(basename "$d")"
  ln -sfn "$d" "$DEST/$name"
  echo "linked $name -> $DEST/$name"
done
echo "done. 'ls ~/.claude/skills' to verify; restart Claude Code if a brand-new skill dir was added."
