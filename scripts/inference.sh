#!/usr/bin/env bash
# -----------------------------------------
# run_inference.sh – batch inference driver
# -----------------------------------------
set -Eeuo pipefail
IFS=$'\n\t'

# ── defaults ────────────────────────────────────────────────────────────────
OUTPUT_DIR="outputs/inference"
PROMPT_TYPES="zscotk"
CKB_PATH="data/ckb/cleaned/merged_filtered.jsonl"
DATASET_LIST="obqa,csqa,qasc"
MODEL_LIST="llama8B,llama3B,qwen1.5B,qwen7B"
RETRIEVAL_STRATEGY_LIST="retriever"
TOP_K_VALUES="1,3,5,10,20"
RETRIEVER_MODEL="models/retriever_zebra/final"
LAMBDA=0.8
DIVERSITY_THRESHOLD=0.9
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
# ────────────────────────────────────────────────────────────────────────────

die() { printf "❌  %s\n" "$*" >&2; exit 1; }

# GNU getopt for long‑flag parsing
PARSED=$(getopt -o h \
  --long help,rerank-type:,lambda:,prompt-types:,ckb-path:,dataset-list:,model-names:,\
retrieval-strategy-list:,top-k-values:,retriever-model:,output-dir:,diversity-threshold:,run-name: \
  -- "$@") || exit 1
eval set -- "$PARSED"

while true; do
  case "$1" in
    --rerank-type)            RERANK_TYPE=$2; shift 2 ;;
    --lambda)                 LAMBDA=$2; shift 2 ;;
    --prompt-types)           PROMPT_TYPES=$2; shift 2 ;;
    --ckb-path)               CKB_PATH=$2; shift 2 ;;
    --dataset-list)           DATASET_LIST=$2; shift 2 ;;
    --model-names)            MODEL_NAMES=$2; shift 2 ;;
    --retrieval-strategy-list) RETRIEVAL_STRATEGY_LIST=$2; shift 2 ;;
    --top-k-values)           TOP_K_VALUES=$2; shift 2 ;;
    --retriever-model)        RETRIEVER_MODEL=$2; shift 2 ;;
    --output-dir)             OUTPUT_DIR=$2; shift 2 ;;
    --diversity-threshold)    DIVERSITY_THRESHOLD=$2; shift 2 ;;
    --run-name)               RUN_NAME=$2; shift 2 ;;
    -h|--help)
        cat <<EOF
Batch inference runner
----------------------
Required:
  --rerank-type <TYPE>          rerank strategy (e.g. mmr, none…)

Optional (defaults in brackets):
  --lambda <FLOAT>              balance factor [$LAMBDA]
  --prompt-types <LIST>         prompt templates [$PROMPT_TYPES]
  --ckb-path <PATH>             cleaned CKB jsonl [$CKB_PATH]
  --dataset-list <LIST>         datasets comma‑sep [$DATASET_LIST]
  --model-names <LIST>          models comma‑sep [$MODEL_NAMES]
  --retrieval-strategy-list     retrieval strategies [$RETRIEVAL_STRATEGY_LIST]
  --top-k-values <LIST>         K values comma‑sep [$TOP_K_VALUES]
  --retriever-model <PATH>      retriever checkpoint [$RETRIEVER_MODEL]
  --output-dir <DIR>            write outputs here [$OUTPUT_DIR]
  --diversity-threshold <FLOAT> MMR diversity [$DIVERSITY_THRESHOLD]
  --run-name <STRING>           experiment tag [$RUN_NAME]
  -h, --help                    show this help
EOF
        exit 0 ;;
    --) shift; break ;;
    *) die "Unknown flag $1" ;;
  esac
done


# ── helpers ────────────────────────────────────────────────────────────────
split_csv() { IFS=',' read -r -a "$2" <<< "$1"; }

split_csv "$DATASET_LIST"            DATASETS
split_csv "$MODEL_LIST"              MODELS
split_csv "$RETRIEVAL_STRATEGY_LIST" RETRIEVAL_STRATEGIES

mkdir -p "$OUTPUT_DIR"

# ── main loop ──────────────────────────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for RETRIEVAL in "${RETRIEVAL_STRATEGIES[@]}"; do
      echo "▶ dataset=$DATASET  model=$MODEL  ${RERANK_TYPE:+rerank=$RERANK_TYPE}  λ=$LAMBDA  top‑k=$TOP_K_VALUES"

      python src/inference/inference.py \
        --output_dir          "$OUTPUT_DIR" \
        --model_name          "$MODEL" \
        --dataset_name        "$DATASET" \
        --ckb_path            "$CKB_PATH" \
        --retrieval_strategy  "$RETRIEVAL" \
        --prompt_types        "$PROMPT_TYPES" \
        --top_k_values        "$TOP_K_VALUES" \
        ${RERANK_TYPE:+--rerank_type "$RERANK_TYPE"} \
        --lambda              "$LAMBDA" \
        --retriever_model     "$RETRIEVER_MODEL" \
        --diversity_threshold "$DIVERSITY_THRESHOLD" \
        --run_name            "$RUN_NAME"
    done
  done
done
