#!/bin/sh

# List of available datasets
AVAILABLE_DATASETS="obqa qasc csqa"

# Ensure dataset name is provided
if [ -z "$1" ]; then
    echo "Error: No dataset name provided."
    echo "Usage: sh script.sh <dataset_name> [model_names] [prompt_types]"
    echo "Available datasets:"
    for dataset in $AVAILABLE_DATASETS; do
        echo "  - $dataset"
    done
    exit 1
fi

DATASET_NAME="$1"

# Optional model names (comma-separated string)
DEFAULT_MODEL_NAMES="meta-llama/Llama-3.2-3B-Instruct,meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-7B-Instruct"
MODEL_NAMES=${2:-$DEFAULT_MODEL_NAMES}

# Optional prompt types (comma-separated string)
DEFAULT_PROMPT_TYPES="zeroshot_with_knowledge,fewshot_with_knowledge"
PROMPT_TYPES=${3:-$DEFAULT_PROMPT_TYPES}

# Validate dataset name
VALID_DATASET="false"
for dataset in $AVAILABLE_DATASETS; do
    if [ "$DATASET_NAME" = "$dataset" ]; then
        VALID_DATASET="true"
        break
    fi
done

if [ "$VALID_DATASET" = "false" ]; then
    echo "Error: Invalid dataset name '$DATASET_NAME'."
    echo "Available datasets:"
    for dataset in $AVAILABLE_DATASETS; do
        echo "  - $dataset"
    done
    exit 1
fi

OUTPUT_DIR="outputs/retriever/training_data/zeroshot"
CKB_PATH="data/ckb/cleaned/full_ckb.jsonl"
TOP_K_VALUES="1,3,5,10,20"

# Convert comma-separated model names to array
IFS=',' read -r -a MODEL_NAME_ARRAY <<< "$MODEL_NAMES"

# Run inference for specified or default models
for MODEL_NAME in "${MODEL_NAME_ARRAY[@]}"; do
    python src/inference/inference.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --ckb_path "$CKB_PATH" \
        --retrieval_strategy "full_ckb" \
        --prompt_types "$PROMPT_TYPES" \
        --top_k_values "$TOP_K_VALUES"

    python src/inference/inference.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --ckb_path "$CKB_PATH" \
        --retrieval_strategy "cner_filter" \
        --prompt_types "$PROMPT_TYPES" \
        --top_k_values "$TOP_K_VALUES"
done
