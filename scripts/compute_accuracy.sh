#!/bin/bash

BASE_DIR="outputs/inference"
CKB_NAME="cleaned_1_steps_ckb_data=wordnet_model=gemini-1.5-flash"
DATASET_NAME="obqa"

python src/evaluation/eval.py \
    --input_dir "$BASE_DIR/$CKB_NAME/$DATASET_NAME/Llama-3.2-3B-Instruct"

python src/evaluation/eval.py \
    --input_dir "$BASE_DIR/$CKB_NAME/$DATASET_NAME/Llama-3.1-8B-Instruct"

python src/evaluation/eval.py \
    --input_dir "$BASE_DIR/$CKB_NAME/$DATASET_NAME/Qwen2.5-1.5B-Instruct"

python src/evaluation/eval.py \
    --input_dir "$BASE_DIR/$CKB_NAME/$DATASET_NAME/Qwen2.5-7B-Instruct"