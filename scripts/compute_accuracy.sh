#!/bin/bash


python src/evaluation/eval.py \
    --input_dir outputs/inference/ckb_data=obqa_split=test_model=gemini-1.5-flash/obqa/Llama-3.2-3B-Instruct 

python src/evaluation/eval.py \
    --input_dir outputs/inference/ckb_data=obqa_split=test_model=gemini-1.5-flash/obqa/Llama-3.1-8B-Instruct 

python src/evaluation/eval.py \
    --input_dir outputs/inference/ckb_data=obqa_split=test_model=gemini-1.5-flash/obqa/Qwen2.5-1.5B-Instruct

python src/evaluation/eval.py \
    --input_dir outputs/inference/ckb_data=obqa_split=test_model=gemini-1.5-flash/obqa/Qwen2.5-7B-Instruct