#!/usr/bin/env bash
# -----------------------------------------
# run_retriever_tester.sh – interactive retriever test loop
# -----------------------------------------
set -Eeuo pipefail
IFS=$'\n\t'

# ── defaults ────────────────────────────────────────────────────────────────
CKB_PATH="data/ckb/cleaned/merged_filtered.jsonl"
DATASET_LIST="obqa,csqa,qasc"
RETRIEVAL_STRATEGY_LIST="retriever"
RETRIEVER_MODEL="models/retriever_trained_all_datasets/final"
RERANK_TYPE="mmr"
LAMBDA=0.8
DIVERSITY_THRESHOLD=0.9
CONFIG_PATH="settings/config.yaml"
# ────────────────────────────────────────────────────────────────────────────

die() { printf "❌  %s\n" "$*" >&2; exit 1; }

# GNU getopt for long‑flag parsing
PARSED=$(getopt -o h \
  --long help,rerank-type:,lambda:,ckb-path:,dataset-list:,\
retrieval-strategy-list:,retriever-model:,config-path:,diversity-threshold: \
  -- "$@") || exit 1
eval set -- "$PARSED"

while true; do
  case "$1" in
    --rerank-type)            RERANK_TYPE=$2; shift 2 ;;
    --lambda)                 LAMBDA=$2; shift 2 ;;
    --ckb-path)               CKB_PATH=$2; shift 2 ;;
    --dataset-list)           DATASET_LIST=$2; shift 2 ;;
    --retrieval-strategy-list) RETRIEVAL_STRATEGY_LIST=$2; shift 2 ;;
    --retriever-model)        RETRIEVER_MODEL=$2; shift 2 ;;
    --config-path)            CONFIG_PATH=$2; shift 2 ;;
    --diversity-threshold)    DIVERSITY_THRESHOLD=$2; shift 2 ;;
    -h|--help)
        cat <<EOF
Retriever-only tester
---------------------
Required:
  --rerank-type <TYPE>          rerank strategy (e.g. mmr, none…)

Optional (defaults in brackets):
  --lambda <FLOAT>              balance factor [$LAMBDA]
  --ckb-path <PATH>             cleaned CKB jsonl [$CKB_PATH]
  --dataset-list <LIST>         datasets comma‑sep [$DATASET_LIST]
  --retrieval-strategy-list     retrieval strategies [$RETRIEVAL_STRATEGY_LIST]
  --retriever-model <PATH>      retriever checkpoint [$RETRIEVER_MODEL]
  --config-path <PATH>          config YAML [$CONFIG_PATH]
  --diversity-threshold <FLOAT> MMR diversity [$DIVERSITY_THRESHOLD]
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
split_csv "$RETRIEVAL_STRATEGY_LIST" RETRIEVAL_STRATEGIES

# ── main loop ──────────────────────────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
  for RETRIEVAL in "${RETRIEVAL_STRATEGIES[@]}"; do
    echo "▶ dataset=$DATASET  rerank=$RERANK_TYPE  λ=$LAMBDA"

    python src/retriever/retriever_tester.py \
      --dataset_name        "$DATASET" \
      --ckb_path            "$CKB_PATH" \
      --retrieval_strategy  "$RETRIEVAL" \
      --retriever_model     "$RETRIEVER_MODEL" \
      --rerank_type         "$RERANK_TYPE" \
      --lambda_             "$LAMBDA" \
      --diversity_threshold "$DIVERSITY_THRESHOLD" \
      --config_path         "$CONFIG_PATH"
  done
done
