#!/usr/bin/env bash

# Default values
SOURCE_DIR="outputs/batches/contextual_ckb/results/"
OUTPUT_DIR="data/ckb/raw/"

# Parse long flags
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --source_dir) SOURCE_DIR="$2"; shift ;;
    --output_dir) OUTPUT_DIR="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

# Print resolved paths
echo "Using:"
echo "  Source Dir:   $SOURCE_DIR"
echo "  Output Dir:   $OUTPUT_DIR"

# Run Python script
python src/ckb_management/ckb_merge_all.py \
  --source_dir "$SOURCE_DIR" \
  --output_dir "$OUTPUT_DIR"
