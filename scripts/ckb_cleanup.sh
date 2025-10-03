#!/usr/bin/env bash

# Default values
INPUT_PATH="data/ckb/raw/kb.jsonl"

# Parse long flags
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --input_path) INPUT_PATH="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

echo "Using:"
echo "Input path: $INPUT_PATH"

# Run Python script
python src/ckb_management/ckb_cleanup.py \
  --input_path "$INPUT_PATH"
