#!/usr/bin/env bash
# -----------------------------------------
# run_retriever_tester.sh – interactive retriever test loop (multi-model)
# -----------------------------------------
set -Eeuo pipefail
IFS=$'\n\t'

# ── defaults ────────────────────────────────────────────────────────────────
CKB_PATH="data/ckb/cleaned/merged_filtered.jsonl"
DATASET_LIST="obqa,csqa,qasc"
RETRIEVAL_STRATEGY_LIST="retriever"
RERANK_TYPE=""
LAMBDA=0.8
DIVERSITY_THRESHOLD=0.9
CONFIG_PATH="settings/config.yaml"
# ────────────────────────────────────────────────────────────────────────────

die() { printf "❌  %s\n" "$*" >&2; exit 1; }

PARSED=$(getopt -o h \
  --long help,rerank-type:,lambda:,ckb-path:,dataset-list:,\
retrieval-strategy-list:,config-path:,diversity-threshold: \
  -- "$@") || exit 1
eval set -- "$PARSED"

while true; do
  case "$1" in
    --rerank-type)            RERANK_TYPE=$2; shift 2 ;;
    --lambda)                 LAMBDA=$2; shift 2 ;;
    --ckb-path)               CKB_PATH=$2; shift 2 ;;
    --dataset-list)           DATASET_LIST=$2; shift 2 ;;
    --retrieval-strategy-list) RETRIEVAL_STRATEGY_LIST=$2; shift 2 ;;
    --config-path)            CONFIG_PATH=$2; shift 2 ;;
    --diversity-threshold)    DIVERSITY_THRESHOLD=$2; shift 2 ;;
    -h|--help)
        cat <<EOF
Retriever-only tester (multi-model)
-----------------------------------
Required:
  --rerank-type <TYPE>          rerank strategy (e.g. mmr, none…)

Optional (defaults in brackets):
  --lambda <FLOAT>              balance factor [$LAMBDA]
  --ckb-path <PATH>             cleaned CKB jsonl [$CKB_PATH]
  --dataset-list <LIST>         datasets comma-sep [$DATASET_LIST]
  --retrieval-strategy-list     retrieval strategies [$RETRIEVAL_STRATEGY_LIST]
  --config-path <PATH>          config YAML [$CONFIG_PATH]
  --diversity-threshold <FLOAT> MMR diversity [$DIVERSITY_THRESHOLD]
  -h, --help                    show this help

Notes:
  • Do not pass --retriever-models here. The Python script will auto-discover
    “intfloat/e5-base-v2” plus every folder matching models/*/final.
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
      --rerank_type         "$RERANK_TYPE" \
      --lambda_             "$LAMBDA" \
      --diversity_threshold "$DIVERSITY_THRESHOLD" \
      --config_path         "$CONFIG_PATH"
  done
done
