#!/bin/sh

# Default values
DEFAULT_DATASETS="obqa,qasc,csqa"
DEFAULT_MODELS="llama3B,llama8B,qwen1.5B,qwen7B"
DEFAULT_RETRIEVAL_STRATEGIES="retriever"

# Parse input arguments or use defaults
RETRIEVAL_STRATEGY_LIST=${1:-$DEFAULT_RETRIEVAL_STRATEGIES}
DATASET_LIST=${2:-$DEFAULT_DATASETS}
MODEL_LIST=${3:-$DEFAULT_MODELS}

# Convert comma-separated to space-separated lists
IFS=',' read -r -a RETRIEVAL_STRATEGIES <<< "$RETRIEVAL_STRATEGY_LIST"
IFS=',' read -r -a DATASETS <<< "$DATASET_LIST"
IFS=',' read -r -a MODELS <<< "$MODEL_LIST"

INPUT_DIR_ROOT="outputs/inference/train/"
OUTPUT_DIR="outputs/retriever/training_data/"
CKB_PATH="data/ckb/cleaned/full_ckb.jsonl"
TOP_K="20"

# Print what will be run
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"

# Loop over datasets and models
for DATASET_NAME in "${DATASETS[@]}"; do
    for MODEL_NAME in "${MODELS[@]}"; do
        for RETRIEVAL_STRATEGY in "${RETRIEVAL_STRATEGIES[@]}"; do
            echo "Running inference for dataset '$DATASET_NAME' with model '$MODEL_NAME' and retrieval strategy '$RETRIEVAL_STRATEGY"

            python src/retriever/augment_incorrect_with_knowledge.py \
                --inpur_dir_root "$INPUT_DIR_ROOT" \
                --output_dir "$OUTPUT_DIR" \
                --model_name "$MODEL_NAME" \
                --dataset_name "$DATASET_NAME" \
                --ckb_path "$CKB_PATH" \
                --retrieval_strategy "$RETRIEVAL_STRATEGY" \
                --top_k "$TOP_K"

        done
    done
done
