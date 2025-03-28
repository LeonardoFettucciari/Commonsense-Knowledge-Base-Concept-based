#!/usr/bin/env bash

# Usage example:
#   bash script.sh <dataset_name> <base_directory> <output_column_name>
#
# Example command:
#   bash script.sh obqa /my/base/dir model_output_revised

# List of available datasets (space-separated string)
AVAILABLE_DATASETS="obqa qasc csqa"

# Ensure dataset name, base directory, and output column name are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: You must provide <dataset_name>, <base_directory>."
    echo "Usage: bash script.sh <dataset_name> <base_directory>"
    echo "Available datasets:"
    for dataset in $AVAILABLE_DATASETS; do
        echo "  - $dataset"
    done
    exit 1
fi

DATASET_NAME="$1"
BASE_DIR="$2"

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

    # Run evaluation
    python src/evaluation/eval_refine.py \
        --input_dir "$INPUT_DIR"
done
