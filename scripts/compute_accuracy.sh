#!/bin/sh

BASE_DIR="outputs/inference"

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

# List of models (space-separated string)
MODELS="Llama-3.2-3B-Instruct Llama-3.1-8B-Instruct Qwen2.5-1.5B-Instruct Qwen2.5-7B-Instruct"

# Loop over models
for MODEL in $MODELS; do
    INPUT_DIR="$BASE_DIR/$DATASET_NAME/$MODEL"

    # Check if input directory exists
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Warning: Input directory '$INPUT_DIR' does not exist. Skipping."
        continue
    fi

    python src/evaluation/eval.py --input_dir "$INPUT_DIR"
done
