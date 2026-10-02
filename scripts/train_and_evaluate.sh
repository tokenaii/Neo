#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:?data path required}"
OUTPUT_PATH="${2:?checkpoint output path required}"
RESULTS_DIR="${3:?results directory required}"
PYTHON="${PYTHON:-python3}"

"$PYTHON" scripts/train_neo.py \
  --data "$DATA_PATH" \
  --output "$OUTPUT_PATH" \
  --batch-size "${NEO_BATCH_SIZE:-1024}" \
  --max-length "${NEO_MAX_LENGTH:-256}" \
  --epochs "${NEO_EPOCHS:-10}" \
  --lr "${NEO_LR:-0.0002}" \
  --choice-weight "${NEO_CHOICE_WEIGHT:-8.0}" \
  --score-weight "${NEO_SCORE_WEIGHT:-1.0}" \
  --noul-weight "${NEO_NOUL_WEIGHT:-1.0}" \
  --tokenizer "${NEO_TOKENIZER:-bert-base-uncased}"

scripts/post_training_eval.sh "$DATA_PATH" "$OUTPUT_PATH" "$RESULTS_DIR"
