#!/usr/bin/env bash
# Symlink StructSplat's project skills into ~/.claude/skills so they're available in every project
# (matching the IntrinsicEngine symlink-based layout). Project-scoped copies in ./.claude/skills are
# already picked up automatically when Claude Code runs inside this repo; this is only for global use.
#
# Every skill here is `structsplat-`prefixed, which is what makes a global install safe: an
# unprefixed name (`core`, `review`, `docs-sync`) installed globally would shadow, or be shadowed
# by, a sibling repository's skill of the same name. The check below fails closed on an unprefixed
# skill rather than polluting the global namespace.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${HOME}/.claude/skills"
mkdir -p "$DEST"
for d in "$REPO"/.claude/skills/*/; do
  name="$(basename "$d")"
  if [[ "$name" != structsplat-* ]]; then
    echo "refusing to install unprefixed skill '$name' globally: rename it to structsplat-$name" >&2
    exit 1
  fi
  ln -sfn "$d" "$DEST/$name"
  echo "linked $name -> $DEST/$name"
done
echo "done. 'ls ~/.claude/skills' to verify; restart Claude Code if a brand-new skill dir was added."
