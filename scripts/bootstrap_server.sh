#!/usr/bin/env bash
set -euo pipefail

NEO_ROOT="/mnt/opet-data/neo"

mkdir -p "$NEO_ROOT"/{repo,data/{raw,generated,processed},runs,checkpoints,logs,cache,venv}

python3.12 -m venv "$NEO_ROOT/venv" 2>/dev/null || python3 -m venv "$NEO_ROOT/venv"

echo "workspace=$NEO_ROOT"
df -h "$NEO_ROOT"
du -sh "$NEO_ROOT"
