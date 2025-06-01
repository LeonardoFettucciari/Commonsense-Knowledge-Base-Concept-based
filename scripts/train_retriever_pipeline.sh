#!/usr/bin/env bash
# run_retriever_pipeline.sh
# Usage: bash run_retriever_pipeline.sh <iteration_number>
# Example: bash run_retriever_pipeline.sh 2

set -euo pipefail

########################################
# 1. Parse & validate arguments
########################################
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <iteration_number>"
  exit 1
fi

ITER="$1"

# Ensure iteration is a positive integer ≥ 2 (we need a previous model)
if ! [[ "$ITER" =~ ^[0-9]+$ ]] || [[ "$ITER" -lt 2 ]]; then
  echo "Error: <iteration_number> must be an integer ≥ 2"
  exit 1
fi

########################################
# 2. Derived paths & names
########################################
PREV_ITER=$((ITER - 1))
RUN_NAME="iteration_${ITER}"
PREV_MODEL="models/retriever_trained_iteration_${PREV_ITER}/final"
OUTPUT_DIR="models/retriever_trained_iteration_${ITER}"

########################################
# 3. Pipeline
########################################
echo "=== [1/4] Augmenting incorrect predictions with knowledge (iteration $ITER) ==="
bash scripts/augment_incorrect_with_knowledge.sh \
  --retriever-model "$PREV_MODEL" \
  --run-name "$RUN_NAME"

echo "=== [2/4] Computing accuracy (iteration $ITER) ==="
bash scripts/compute_accuracy.sh \
  --run-name "$RUN_NAME" \
  --input-dir-root outputs/retriever_trainset

echo "=== [3/4] Extracting positives & negatives (iteration $ITER) ==="
bash scripts/extract_positives_negatives.sh \
  --run-name "$RUN_NAME"

echo "=== [4/4] Training retriever (saving to $OUTPUT_DIR) ==="
bash scripts/retriever_trainer.sh \
  --retriever-model intfloat/e5-base-v2 \
  --run-name "$RUN_NAME" \
  --trainset-base-dir outputs/retriever_trainset \
  --output-dir "$OUTPUT_DIR"

echo "✅  Pipeline finished for $RUN_NAME"
