#!/usr/bin/env bash

# Usage:
#   bash script.sh <input_dir>
#
# Example:
#   bash script.sh /path/to/eval/data

# Check if input_dir is provided
if [ -z "$1" ]; then
    echo "Error: You must provide <input_dir>."
    echo "Usage: bash script.sh <input_dir>"
    exit 1
fi

INPUT_DIR="$1"

# Check if input_dir exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Run evaluation
echo "Evaluating directory: $INPUT_DIR"
python src/evaluation/eval.py \
    --input_dir "$INPUT_DIR"
