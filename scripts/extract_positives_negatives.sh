#!/bin/sh

INPUT_DIR="outputs/retriever/training_data/"

# Run inference
python src/retriever/extract_positives_negatives.py \
    --input_dir "$INPUT_DIR" \
