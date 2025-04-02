#!/bin/sh

# List of available datasets
AVAILABLE_DATASETS="obqa qasc csqa"

# Ensure dataset name is provided
if [ -z "$1" ]; then
    echo "Error: No dataset name provided."
    echo "Usage: sh script.sh <dataset_name> <model_name>"
    echo "Available datasets:"
    for dataset in $AVAILABLE_DATASETS; do
        echo "  - $dataset"
    done
    exit 1
fi

# Ensure model name is provided
if [ -z "$2" ]; then
    echo "Error: No model name provided."
    echo "Usage: sh script.sh <dataset_name> <model_name>"
    echo "Example model names:"
    echo "  - meta-llama/Llama-3.2-3B-Instruct"
    echo "  - meta-llama/Llama-3.1-8B-Instruct"
    echo "  - Qwen/Qwen2.5-1.5B-Instruct"
    echo "  - Qwen/Qwen2.5-7B-Instruct"
    exit 1
fi

DATASET_NAME="$1"
MODEL_NAME="$2"

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

OUTPUT_DIR="outputs/retriever/training_data/"
CKB_PATH="data/ckb/cleaned/full_ckb.jsonl"
TOP_K="20"

# Run inference
python src/retriever/create_training_data.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --ckb_path "$CKB_PATH" \
    --retrieval_strategy "full_ckb" \
    --top_k "$TOP_K"

python src/retriever/create_training_data.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --ckb_path "$CKB_PATH" \
    --retrieval_strategy "cner_filter" \
    --top_k "$TOP_K"
