#!/bin/bash

python src/inference/inference.py \
    --output_dir outputs/inference_1 \
    --kb_path data/ckb_data=obqa_split=test_model=gemini-1.5-flash.jsonl \
    #--limit_samples 500 \
    #--top_k_list 10 \