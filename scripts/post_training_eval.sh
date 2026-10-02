#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:?data path required}"
CHECKPOINT_PATH="${2:?checkpoint path required}"
RESULTS_DIR="${3:?results directory required}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

mkdir -p "$RESULTS_DIR"
cd "$REPO_ROOT"

"$PYTHON" -m py_compile scripts/train_neo.py scripts/benchmark_neo.py scripts/benchmark_latency.py

"$PYTHON" scripts/benchmark_neo.py \
  --data "$DATA_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --output "$RESULTS_DIR/benchmark.json" \
  > "$RESULTS_DIR/benchmark.log" 2>&1

"$PYTHON" scripts/benchmark_latency.py \
  --data "$DATA_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --output "$RESULTS_DIR/latency.json" \
  > "$RESULTS_DIR/latency.log" 2>&1

"$PYTHON" examples/tool_router/router.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --message "Calculate the current exchange rate and explain which tool should handle it." \
  > "$RESULTS_DIR/router_smoke.json"

date -u +%Y-%m-%dT%H:%M:%SZ > "$RESULTS_DIR/completed_at.txt"
echo "Post-training evaluation completed: $RESULTS_DIR"
