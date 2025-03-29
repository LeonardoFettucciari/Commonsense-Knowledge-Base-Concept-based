#!/usr/bin/env bash

# Ensure at least the kb_jsonl_path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <kb_jsonl_path> [threshold]"
    exit 1
fi

KB_JSONL_PATH="$1"

# If threshold is provided, store it in a variable to be appended
if [ -n "$2" ]; then
    THRESHOLD_ARG="--threshold $2"
fi

python src/ckb_management/ckb_vera.py \
    --kb_jsonl_path "$KB_JSONL_PATH" \
    $THRESHOLD_ARG
