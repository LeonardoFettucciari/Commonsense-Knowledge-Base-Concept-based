#!/bin/bash


python src/inference/inference.py \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --dataset_name allenai/openbookqa \


python src/inference/inference.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name allenai/openbookqa \


python src/inference/inference.py \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name allenai/openbookqa \


python src/inference/inference.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset_name allenai/openbookqa \
