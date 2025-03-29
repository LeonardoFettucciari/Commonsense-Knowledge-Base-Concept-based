#!/bin/sh

# List of available datasets (space-separated string)
AVAILABLE_DATASETS="obqa qasc csqa"

# Ensure dataset name is provided
if [ -z "$1" ]; then
    echo "Error: No dataset name provided."
    echo "Usage: sh script.sh <dataset_name>"
    echo "Available datasets:"
    for dataset in $AVAILABLE_DATASETS; do
        echo "  - $dataset"
    done
    exit 1
fi

DATASET_NAME="$1"

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

OUTPUT_DIR="outputs/vera/inference"
CKB_PATH="data/ckb/cleaned/vera_full_ckb.jsonl"
TOP_K_VALUES="1,3,5,10,20"

# Run inference for multiple models
for MODEL_NAME in "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" \
                  "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-7B-Instruct"; do

    python src/inference/inference.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --ckb_path "$CKB_PATH" \
        --retrieval_strategy "full_ckb" \
        --prompt_types "zeroshot_with_knowledge,fewshot_with_knowledge" \
        --top_k_values "$TOP_K_VALUES"

    python src/inference/inference.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --ckb_path "$CKB_PATH" \
        --retrieval_strategy "cner_filter" \
        --prompt_types "zeroshot_with_knowledge,fewshot_with_knowledge" \
        --top_k_values "$TOP_K_VALUES"
done
