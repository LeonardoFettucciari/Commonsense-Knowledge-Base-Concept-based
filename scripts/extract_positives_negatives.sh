#!/bin/sh

MODELS_LIST="Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct"
DATASETS_LIST="obqa qasc csqa"
BASE_INPUT_DIR="outputs/retriever_trainset"
RUN_NAME="iteration_2"

for model in $MODELS_LIST; do
  for dataset in $DATASETS_LIST; do
    INPUT_DIR="${BASE_INPUT_DIR}/${dataset}/${model}/${RUN_NAME}"
    OUTPUT_RUN_NAME="${RUN_NAME}_positives_negatives"

    echo "Running on MODEL=$model, DATASET=$dataset"
    
    python src/retriever/extract_positives_negatives.py \
      --input_dir "$INPUT_DIR" \
      --output_dir "$BASE_INPUT_DIR" \
      --output_run_name "$OUTPUT_RUN_NAME"
  done
done
