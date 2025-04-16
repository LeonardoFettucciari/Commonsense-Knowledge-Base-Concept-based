#!/usr/bin/env bash

# Default values
DEFAULT_DATASETS="obqa,qasc,csqa"
DEFAULT_MODELS="Llama-3.1-8B-Instruct,Llama-3.2-3B-Instruct,Qwen2.5-1.5B-Instruct,Qwen2.5-7B-Instruct"

# Parse input arguments or use defaults
INPUT_DIR_ROOT="$1"
DATASET_LIST=${2:-$DEFAULT_DATASETS}
MODEL_LIST=${3:-$DEFAULT_MODELS}

# Convert comma-separated to space-separated lists
IFS=',' read -r -a DATASETS <<< "$DATASET_LIST"
IFS=',' read -r -a MODELS <<< "$MODEL_LIST"

# Loop over datasets and models
for DATASET_NAME in "${DATASETS[@]}"; do
    for MODEL_NAME in "${MODELS[@]}"; do
        INPUT_DIR="$INPUT_DIR_ROOT/$DATASET_NAME/$MODEL_NAME"
        
        echo "Evaluating '$INPUT_DIR'."
        python src/evaluation/eval.py \
            --input_dir "$INPUT_DIR"

    done
done
