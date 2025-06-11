#!/usr/bin/env bash
# -----------------------------------------
# run_single_retriever_tester.sh – interactive retriever test loop (single retriever with dynamic settings)
# -----------------------------------------
set -Eeuo pipefail
IFS=$'\n\t'

# ── defaults ────────────────────────────────────────────────────────────────
CKB_PATH="data/ckb/cleaned/merged_filtered.jsonl"
DATASET_LIST="obqa,csqa,qasc"
CONFIG_PATH="settings/config.yaml"
TOP_K=5
# ────────────────────────────────────────────────────────────────────────────

die() { printf "❌  %s\n" "$*" >&2; exit 1; }

PARSED=$(getopt -o h \
  --long help,ckb-path:,dataset-list:,config-path:,top-k: \
  -- "$@") || exit 1
eval set -- "$PARSED"

while true; do
  case "$1" in
    --ckb-path)      CKB_PATH=$2; shift 2 ;;
    --dataset-list)  DATASET_LIST=$2; shift 2 ;;
    --config-path)   CONFIG_PATH=$2; shift 2 ;;
    --top-k)         TOP_K=$2; shift 2 ;;
    -h|--help)
        cat <<EOF
Single retriever tester (fully interactive version)
--------------------------------------------------
Required:
  (no required flags — model, retrieval strategy, rerank type, threshold, lambda are selected interactively)

Optional (defaults in brackets):
  --ckb-path <PATH>             cleaned CKB jsonl [$CKB_PATH]
  --dataset-list <LIST>         datasets comma-sep [$DATASET_LIST]
  --config-path <PATH>          config YAML [$CONFIG_PATH]
  --top-k <INT>                 number of retrieved statements [$TOP_K]

Notes:
  • The script will prompt you for model, retrieval strategy, rerank type, diversity threshold, lambda.
  • You can change these settings anytime by typing 'settings' during the interactive session.
EOF
        exit 0 ;;
    --) shift; break ;;
    *) die "Unknown flag $1" ;;
  esac
done

# ── helpers ────────────────────────────────────────────────────────────────
split_csv() { IFS=',' read -r -a "$2" <<< "$1"; }

split_csv "$DATASET_LIST" DATASETS

# ── main loop ──────────────────────────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
    echo "▶ dataset=$DATASET"

    python src/retriever/test_retriever.py \
      --dataset_name        "$DATASET" \
      --ckb_path            "$CKB_PATH" \
      --top_k               "$TOP_K" \
      --config_path         "$CONFIG_PATH"
done
