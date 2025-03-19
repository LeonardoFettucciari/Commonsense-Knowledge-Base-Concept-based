#!/bin/bash

OUTPUT_DIR="outputs/inference"
CKB_PATH="data/ckb/cleaned/ckb_data=wordnet|model=gemini-1.5-flash.jsonl"
PROMPT_TYPES="all"
TOP_K_VALUES="1,3,5,10,20"
DATASET_NAME="allenai/openbookqa"

# Run inference for multiple models
#for MODEL_NAME in "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" \
#                  "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-7B-Instruct"; do

for MODEL_NAME in "Qwen/Qwen2.5-7B-Instruct"; do

    python src/inference/inference.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --ckb_path "$CKB_PATH" \
        --retrieval_scope "full_ckb" \
        --prompt_types "$PROMPT_TYPES" \
        --top_k_values "$TOP_K_VALUES"

    python src/inference/inference.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        --ckb_path "$CKB_PATH" \
        --retrieval_scope "cner_synset_filtered_kb" \
        --prompt_types "zeroshot_with_knowledge,fewshot_with_knowledge" \
        --top_k_values "$TOP_K_VALUES"
done
