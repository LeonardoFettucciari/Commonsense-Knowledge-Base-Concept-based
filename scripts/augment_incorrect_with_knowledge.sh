#!/bin/sh

# Default values
DEFAULT_DATASETS="obqa,qasc,csqa"
DEFAULT_MODELS="llama8B,qwen7B"
DEFAULT_RETRIEVAL_STRATEGIES="retriever"
DEFAULT_INPUT_RUN_NAME="my_experiment"
DEFAULT_OUTPUT_RUN_NAME="my_experiment_augmented"
DEFAULT_PROMPT_NAME="zscot"
DEFAULT_BATCH_SIZE=1

# Parse input arguments or use defaults
RETRIEVAL_STRATEGY_LIST=${1:-$DEFAULT_RETRIEVAL_STRATEGIES}
DATASET_LIST=${2:-$DEFAULT_DATASETS}
MODEL_LIST=${3:-$DEFAULT_MODELS}
INPUT_RUN_NAME=${4:-$DEFAULT_INPUT_RUN_NAME}
OUTPUT_RUN_NAME=${5:-$DEFAULT_OUTPUT_RUN_NAME}
PROMPT_NAME=${6:-$DEFAULT_PROMPT_NAME}
BATCH_SIZE=${7:-$DEFAULT_BATCH_SIZE}

# Convert comma-separated to space-separated lists
IFS=',' read -r -a RETRIEVAL_STRATEGIES <<< "$RETRIEVAL_STRATEGY_LIST"
IFS=',' read -r -a DATASETS <<< "$DATASET_LIST"
IFS=',' read -r -a MODELS <<< "$MODEL_LIST"

INPUT_DIR_ROOT="outputs/inference"
OUTPUT_DIR="outputs/inference"
CKB_PATH="data/ckb/cleaned/merged_filtered.jsonl"
TOP_K="20"
CONFIG_PATH="settings/config1.yaml"

# Print what will be run
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"
echo "Retrieval strategies: ${RETRIEVAL_STRATEGIES[*]}"
echo "Batch size: $BATCH_SIZE"

# Loop over datasets, models, and strategies
for DATASET_NAME in "${DATASETS[@]}"; do
  for MODEL_NAME in "${MODELS[@]}"; do
    for RETRIEVAL_STRATEGY in "${RETRIEVAL_STRATEGIES[@]}"; do
      echo "Running inference for dataset '$DATASET_NAME' with model '$MODEL_NAME', retrieval strategy '$RETRIEVAL_STRATEGY', input run '$INPUT_RUN_NAME', output run '$OUTPUT_RUN_NAME', prompt '$PROMPT_NAME', batch size $BATCH_SIZE"

      python src/retriever/augment_incorrect_with_knowledge.py \
        --input_dir_root "$INPUT_DIR_ROOT" \
        --input_run_name "$INPUT_RUN_NAME" \
        --prompt_name "$PROMPT_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --output_run_name "$OUTPUT_RUN_NAME" \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --ckb_path "$CKB_PATH" \
        --retrieval_strategy "$RETRIEVAL_STRATEGY" \
        --top_k "$TOP_K" \
        --config_path "$CONFIG_PATH" \
        --batch_size "$BATCH_SIZE"
    done
  done
done
