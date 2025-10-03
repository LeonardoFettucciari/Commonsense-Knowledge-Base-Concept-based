#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# ── defaults ────────────────────────────────────────────────────────────────
INPUT_DIR_ROOT="outputs/inference"
DATASET_LIST="obqa,qasc,csqa"
MODELS="Llama-3.1-8B-Instruct,Llama-3.2-3B-Instruct,Qwen2.5-1.5B-Instruct,Qwen2.5-7B-Instruct"
RUN_NAME="default_run"
# ────────────────────────────────────────────────────────────────────────────────

die() { printf "%s\n" "$*" >&2; exit 1; }

# GNU getopt for long‑flag parsing
PARSED=$(getopt -o h \
  --long help,input-dir-root:,dataset-list:,models:,run-name: \
  -- "$@") || exit 1
eval set -- "$PARSED"

while true; do
  case "$1" in
    --input-dir-root)  INPUT_DIR_ROOT=$2; shift 2 ;;
    --dataset-list)    DATASET_LIST=$2; shift 2 ;;
    --models)      MODELS=$2; shift 2 ;;
    --run-name)        RUN_NAME=$2; shift 2 ;;
    -h|--help)
cat <<EOF
Batch evaluation runner
-----------------------
Optional (defaults in brackets):
  --input-dir-root <DIR>      root input directory
  --dataset-list <LIST>       datasets comma‑sep [${DATASET_LIST}]
  --models <LIST>         models comma‑sep [${MODELS}]
  --run-name <STRING>         experiment run tag [${RUN_NAME}]
  -h, --help                  show this help
EOF
      exit 0 ;;
    --) shift; break ;;
    *) die "Unknown flag: $1" ;;
  esac
done

split_csv() { IFS=',' read -r -a "$2" <<< "$1"; }
split_csv "$DATASET_LIST" DATASETS
split_csv "$MODELS" MODELS

for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    INPUT_DIR="$INPUT_DIR_ROOT/$DATASET/$MODEL/$RUN_NAME"

    python src/evaluation/eval.py \
      --input_dir "$INPUT_DIR"
  done
done

