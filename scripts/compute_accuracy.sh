#!/bin/bash

BASE_DIR="outputs/inference"
DATASET_NAME="obqa"
OVERWRITE="--overwrite"


python src/evaluation/eval.py \
    --input_dir "$BASE_DIR/$DATASET_NAME/Llama-3.2-3B-Instruct"


python src/evaluation/eval.py \
    --input_dir "$BASE_DIR/$DATASET_NAME/Llama-3.1-8B-Instruct"


python src/evaluation/eval.py \
    --input_dir "$BASE_DIR/$DATASET_NAME/Qwen2.5-1.5B-Instruct"


python src/evaluation/eval.py \
    --input_dir "$BASE_DIR/$DATASET_NAME/Qwen2.5-7B-Instruct"