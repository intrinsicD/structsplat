#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export STRUCTSPLAT_INSTANT_GI="${STRUCTSPLAT_INSTANT_GI:-/home/alex/Documents/Instant-GI/quard_image.py}"

python -m benchmarks.storage_budget_compare "$@"
