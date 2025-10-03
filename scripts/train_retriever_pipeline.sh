#!/usr/bin/env bash
# Usage: bash run_retriever_pipeline.sh --iteration <iteration_number> --rerank-type <type>

set -euo pipefail

# 1. Parse & validate arguments
ITER=""
RERANK_TYPE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iteration)
      ITER="$2"
      shift 2
      ;;
    --rerank-type)
      RERANK_TYPE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --iteration <number> --rerank-type <type>"
      exit 1
      ;;
  esac
done

# Check required arguments
if [[ -z "$ITER" || -z "$RERANK_TYPE" ]]; then
  echo "Missing required arguments."
  echo "Usage: $0 --iteration <number> --rerank-type <type>"
  exit 1
fi

# Ensure iteration is a positive integer >= 1
if ! [[ "$ITER" =~ ^[0-9]+$ ]] || [[ "$ITER" -lt 1 ]]; then
  echo "Error: --iteration must be an integer >= 1"
  exit 1
fi

# 2. Derived paths & names
PREV_ITER=$((ITER - 1))
if [[ -n "$RERANK_TYPE" ]]; then
  RUN_NAME="${RERANK_TYPE}/iteration_${ITER}"
else
  RUN_NAME="iteration_${ITER}"
fi

OUTPUT_DIR="models/retriever_trained_iteration_${RERANK_TYPE}${ITER}"

if [[ "$PREV_ITER" -eq 0 ]]; then
  PREV_MODEL="intfloat/e5-base-v2"
else
  PREV_MODEL="models/retriever_trained_iteration_${RERANK_TYPE}${PREV_ITER}/final"
fi

# 3. Pipeline
echo "=== [1/4] Augmenting incorrect predictions with knowledge (iteration $ITER) ==="
bash scripts/augment_incorrect_with_knowledge.sh \
  --retriever-model "$PREV_MODEL" \
  --run-name "$RUN_NAME" \
  --rerank-type "$RERANK_TYPE"

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

echo "Pipeline finished for $RUN_NAME"
