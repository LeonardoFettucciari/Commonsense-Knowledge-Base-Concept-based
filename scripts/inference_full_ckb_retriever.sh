#!/bin/bash
OUTPUT_DIR="outputs/inference/"
CKB_PATH="data/ckb/cleaned/cleaned_1_steps_ckb_data=wordnet_model=gemini-1.5-flash.jsonl"
PROMPT_TYPES="zeroshot_with_knowledge,fewshot_with_knowledge"
TOP_K_VALUES="1,3,5,10,20"

python src/inference/inference_full_ckb_retriever.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset_name allenai/openbookqa \
    --ckb_path "$CKB_PATH" \
    --prompt_types "$PROMPT_TYPES" \
   --top_k_values "$TOP_K_VALUES"


python src/inference/inference_full_ckb_retriever.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name allenai/openbookqa \
    --ckb_path "$CKB_PATH" \
    --prompt_types "$PROMPT_TYPES" \
    --top_k_values "$TOP_K_VALUES"


python src/inference/inference_full_ckb_retriever.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name allenai/openbookqa \
    --ckb_path "$CKB_PATH" \
    --prompt_types "$PROMPT_TYPES" \
    --top_k_values "$TOP_K_VALUES"


python src/inference/inference_full_ckb_retriever.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset_name allenai/openbookqa \
    --ckb_path "$CKB_PATH" \
    --prompt_types "$PROMPT_TYPES" \
    --top_k_values "$TOP_K_VALUES"