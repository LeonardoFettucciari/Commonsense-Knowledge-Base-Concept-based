#!/bin/sh

# Default values
DEFAULT_CKB_TYPE="regular"
DEFAULT_DATASETS="qasc,csqa,obqa"
DEFAULT_PROMPT_TYPES="zeroshot"
DEFAULT_MODEL_NAMES="meta-llama/Llama-3.2-3B-Instruct,meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-7B-Instruct"
DEFAULT_RETRIEVAL_STRATEGIES="retriever"

# Parse input arguments or use defaults
CKB_TYPE=${1:-$DEFAULT_CKB_TYPE}
DATASET_LIST=${2:-$DEFAULT_DATASETS}
PROMPT_TYPES=${3:-$DEFAULT_PROMPT_TYPES}
MODEL_NAMES=${4:-$DEFAULT_MODEL_NAMES}
RETRIEVAL_STRATEGY_LIST=${5:-$DEFAULT_RETRIEVAL_STRATEGIES}

# Set CKB path based on the CKB type
if [ "$CKB_TYPE" = "vera" ]; then
    CKB_PATH="data/ckb/cleaned/full_ckb_vera.jsonl"
else
    CKB_PATH="data/ckb/cleaned/full_ckb.jsonl"
fi

# Convert comma-separated model names to array
IFS=',' read -r -a MODEL_NAME_ARRAY <<< "$MODEL_NAMES"
IFS=',' read -r -a DATASETS <<< "$DATASET_LIST"
IFS=',' read -r -a RETRIEVAL_STRATEGIES <<< "$RETRIEVAL_STRATEGY_LIST"

OUTPUT_DIR="outputs/inference/train"
TOP_K_VALUES="1,3,5,10,20"

# Loop over datasets and models
for DATASET_NAME in "${DATASETS[@]}"; do
    for MODEL_NAME in "${MODEL_NAME_ARRAY[@]}"; do
        for RETRIEVAL_STRATEGY in "${RETRIEVAL_STRATEGIES[@]}"; do
            echo "Running inference for dataset '$DATASET_NAME' with model '$MODEL_NAME' and CKB type '$CKB_TYPE'"

            python src/inference/new_inference.py \
                --output_dir "$OUTPUT_DIR" \
                --model_name "$MODEL_NAME" \
                --dataset_name "$DATASET_NAME" \
                --ckb_path "$CKB_PATH" \
                --retrieval_strategy "$RETRIEVAL_STRATEGY" \
                --prompt_types "$PROMPT_TYPES" \
                --top_k_values "$TOP_K_VALUES"
        done
    done
done
