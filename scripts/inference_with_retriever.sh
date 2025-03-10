#!/bin/bash

CKB_PATH="data/ckb/cleaned/cleaned_1_steps_ckb_data=wordnet_model=gemini-1.5-flash.jsonl"
PROMPT_TYPES="zeroshot_with_knowledge,fewshot_with_knowledge"

python src/inference/inference.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset_name allenai/openbookqa \
    --ckb_path "$CKB_PATH" \
    --prompt_types "$PROMPT_TYPES" \


python src/inference/inference.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name allenai/openbookqa \
    --ckb_path "$CKB_PATH" \
    --prompt_types "$PROMPT_TYPES" \


python src/inference/inference.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name allenai/openbookqa \
    --ckb_path "$CKB_PATH" \
    --prompt_types "$PROMPT_TYPES" \


python src/inference/inference.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset_name allenai/openbookqa \
    --ckb_path "$CKB_PATH" \
    --prompt_types "$PROMPT_TYPES" \