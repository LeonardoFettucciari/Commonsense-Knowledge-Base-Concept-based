#!/bin/bash

python src/inference/inference.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset_name allenai/openbookqa \
    --output_dir outputs/inference_1 \
    --kb_path data/ckb_data=obqa_split=test_model=gemini-1.5-flash.jsonl \


python src/inference/inference.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name allenai/openbookqa \
    --output_dir outputs/inference_1 \
    --kb_path data/ckb_data=obqa_split=test_model=gemini-1.5-flash.jsonl \


python src/inference/inference.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name allenai/openbookqa \
    --output_dir outputs/inference_1 \
    --kb_path data/ckb_data=obqa_split=test_model=gemini-1.5-flash.jsonl \


python src/inference/inference.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset_name allenai/openbookqa \
    --output_dir outputs/inference_1 \
    --kb_path data/ckb_data=obqa_split=test_model=gemini-1.5-flash.jsonl \
