#!/bin/bash

# List of available datasets
AVAILABLE_DATASETS="obqa qasc csqa"

# Ensure dataset name is provided
if [ -z "$1" ]; then
    echo "Error: No dataset name provided."
    echo "Usage: bash script.sh <dataset_name> [model_names] [top_k_values]"
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
    if [ "$DATASET_NAME" == "$dataset" ]; then
        VALID_DATASET="true"
        break
    fi
done

if [ "$VALID_DATASET" == "false" ]; then
    echo "Error: Invalid dataset name '$DATASET_NAME'."
    echo "Available datasets:"
    for dataset in $AVAILABLE_DATASETS; do
        echo "  - $dataset"
    done
    exit 1
fi

# Default model list
DEFAULT_MODELS=(
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

# Parse models from argument 2 if given
if [ -n "$2" ]; then
    IFS=',' read -r -a MODELS <<< "$2"
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# Parse top_k values from argument 3 if given
if [ -n "$3" ]; then
    TOP_K_VALUES="$3"
else
    TOP_K_VALUES="1,3,5,10,20"
fi

OUTPUT_DIR="outputs/inference_revised"
CKB_PATH="data/ckb/cleaned/ckb_data=wordnet|model=gemini-1.5-flash.jsonl"

# Run inference
for MODEL_NAME in "${MODELS[@]}"; do
    echo "Running inference with model: $MODEL_NAME and top_k: $TOP_K_VALUES"

    python src/inference/inference_revised_with_knowledge.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --ckb_path "$CKB_PATH" \
        --retrieval_scope "full_ckb" \
        --prompt_types "zeroshot,zeroshot_cot,fewshot_cot" \
        --top_k_values "$TOP_K_VALUES"

    python src/inference/inference_revised_with_knowledge.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --ckb_path "$CKB_PATH" \
        --retrieval_scope "cner_synset_filtered_kb" \
        --prompt_types "zeroshot,zeroshot_cot,fewshot_cot" \
        --top_k_values "$TOP_K_VALUES"
done
