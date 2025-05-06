#!/usr/bin/env bash
# -----------------------------------------
# eval.sh – batch evaluation driver
# -----------------------------------------
set -Eeuo pipefail
IFS=$'\n\t'

# ── defaults ────────────────────────────────────────────────────────────────
INPUT_DIR_ROOT="outputs/inference"
DATASET_LIST="obqa,qasc,csqa"
MODEL_LIST="Llama-3.1-8B-Instruct,Llama-3.2-3B-Instruct,Qwen2.5-1.5B-Instruct,Qwen2.5-7B-Instruct"
RUN_NAME="default_run"
# ────────────────────────────────────────────────────────────────────────────────

die() { printf "❌  %s\n" "$*" >&2; exit 1; }

# GNU getopt for long‑flag parsing
PARSED=$(getopt -o h \
  --long help,input-dir-root:,dataset-list:,model-list:,run-name: \
  -- "$@") || exit 1
eval set -- "$PARSED"

while true; do
  case "$1" in
    --input-dir-root)  INPUT_DIR_ROOT=$2; shift 2 ;;
    --dataset-list)    DATASET_LIST=$2; shift 2 ;;
    --model-list)      MODEL_LIST=$2; shift 2 ;;
    --run-name)        RUN_NAME=$2; shift 2 ;;
    -h|--help)
cat <<EOF
Batch evaluation runner
-----------------------
Optional (defaults in brackets):
  --input-dir-root <DIR>      root input directory
  --dataset-list <LIST>       datasets comma‑sep [${DATASET_LIST}]
  --model-list <LIST>         models comma‑sep [${MODEL_LIST}]
  --run-name <STRING>         experiment run tag [${RUN_NAME}]
  -h, --help                  show this help
EOF
      exit 0 ;;
    --) shift; break ;;
    *) die "Unknown flag: $1" ;;
  esac
done

# ── helpers ────────────────────────────────────────────────────────────────
split_csv() { IFS=',' read -r -a "$2" <<< "$1"; }

split_csv "$DATASET_LIST" DATASETS
split_csv "$MODEL_LIST" MODELS

# ── main loop ──────────────────────────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    RUN_DIR="$INPUT_DIR_ROOT/$DATASET/$MODEL/$RUN_NAME"
    INPUT_DIR=$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)

    if [ -z "$INPUT_DIR" ]; then
      die "No subdirectory found in $RUN_DIR"
    fi

    echo "Evaluating '$INPUT_DIR'."
    python src/evaluation/eval.py \
      --input_dir "$INPUT_DIR"
  done
done

