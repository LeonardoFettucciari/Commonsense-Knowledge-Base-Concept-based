#!/usr/bin/env bash

# Usage:
#   bash script.sh <base_directory> <dataset_name> [model1 model2 ...]
#
# Example:
#   bash script.sh /my/base/dir obqa llama8B qwen1.5B


# Model aliases mapping
declare -A MODEL_ALIASES=(
    ["llama3B"]="Llama-3.2-3B-Instruct"
    ["llama8B"]="Llama-3.1-8B-Instruct"
    ["qwen1.5B"]="Qwen2.5-1.5B-Instruct"
    ["qwen7B"]="Qwen2.5-7B-Instruct"
)

# Default models if none are provided
DEFAULT_MODELS=("Llama-3.2-3B-Instruct" "Llama-3.1-8B-Instruct" "Qwen2.5-1.5B-Instruct" "Qwen2.5-7B-Instruct")

# Check minimum required arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: You must provide <base_directory> and <dataset_name>."
    echo "Usage: bash script.sh <base_directory> <dataset_name> [model1 model2 ...]"

    exit 1
fi

BASE_DIR="$1"
DATASET_NAME="$2"
shift 2




# Determine list of models to use
if [ "$#" -eq 0 ]; then
    MODELS=("${DEFAULT_MODELS[@]}")
else
    MODELS=()
    for alias in "$@"; do
        if [[ ${MODEL_ALIASES[$alias]+_} ]]; then
            MODELS+=("${MODEL_ALIASES[$alias]}")
        else
            echo "Warning: Unknown model alias '$alias'. Using it as-is."
            MODELS+=("$alias")  # Accept raw model name
        fi
    done
fi

# Loop over each model
for MODEL in "${MODELS[@]}"; do
    INPUT_DIR="$BASE_DIR/$DATASET_NAME/$MODEL"
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Warning: Input directory '$INPUT_DIR' does not exist. Skipping."
        continue
    fi

    echo "Evaluating $MODEL..."
    python src/evaluation/eval.py \
        --input_dir "$INPUT_DIR"
done
