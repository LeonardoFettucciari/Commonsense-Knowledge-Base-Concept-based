#!/usr/bin/env bash

# Default values
SOURCE1_PATH="data/source1.jsonl"
SOURCE2_PATH="data/source2.jsonl"
OUTPUT_DIR="data/merged"

# Parse long flags
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --source1) SOURCE1_PATH="$2"; shift ;;
    --source2) SOURCE2_PATH="$2"; shift ;;
    --output_dir) OUTPUT_DIR="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# Print resolved paths
echo "Using:"
echo "  Source 1:     $SOURCE1_PATH"
echo "  Source 2:     $SOURCE2_PATH"
echo "  Output Dir:   $OUTPUT_DIR"

# Run Python script
python src/ckb_management/ckb_merge.py \
  --source1_path "$SOURCE1_PATH" \
  --source2_path "$SOURCE2_PATH" \
  --output_dir "$OUTPUT_DIR"
