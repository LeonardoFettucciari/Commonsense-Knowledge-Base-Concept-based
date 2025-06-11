#!/usr/bin/env bash
set -euo pipefail

PYTHON_SCRIPT="src/utils/extra/compare.py"

# Manually specify configurations below
datasets=("obqa" "csqa" "qasc")
models=("llama8B" "llama3B" "qwen1.5B" "qwen7B")
experiments1=("oracle_fscotk10")
prompts1=("fscotk10")

experiments2=("trained_retriever_mmr_fewshot")
prompts2=("fscotk5")

# Loop through combinations
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for i in "${!experiments1[@]}"; do
      exp1="${experiments1[i]}"
      prompt1="${prompts1[i]}"
      exp2="${experiments2[i]}"
      prompt2="${prompts2[i]}"

      echo "🔍 Running comparison for:"
      echo "   Dataset:     $dataset"
      echo "   Model:       $model"
      echo "   Baseline:    $exp1 ($prompt1)"
      echo "   With KB:     $exp2 ($prompt2)"
      echo "--------------------------------"

      python "$PYTHON_SCRIPT" \
        --dataset "$dataset" \
        --model "$model" \
        --exp1 "$exp1" \
        --prompt1 "$prompt1" \
        --exp2 "$exp2" \
        --prompt2 "$prompt2"
    done
  done
done
