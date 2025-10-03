#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

MODELS_LIST="Llama-3.1-8B-Instruct,Qwen2.5-7B-Instruct"
DATASETS_LIST="obqa,qasc,csqa"
BASE_INPUT_DIR="outputs/retriever_trainset"
RUN_NAME="iteration_1"

die() { printf "❌  %s\n" "$*" >&2; exit 1; }

PARSED=$(getopt -o h --long help,models:,datasets:,run-name:,base-input-dir: -- "$@") || die
eval set -- "$PARSED"

while true; do
  case "$1" in
    --models)         MODELS_LIST=$2; shift 2 ;;
    --datasets)       DATASETS_LIST=$2; shift 2 ;;
    --run-name)       RUN_NAME=$2; shift 2 ;;
    --base-input-dir) BASE_INPUT_DIR=$2; shift 2 ;;
    -h|--help)
      cat <<EOF
Stage-3: build positives/negatives + triplets.

  --models <LIST>          comma-sep list [$MODELS_LIST]
  --datasets <LIST>        comma-sep list [$DATASETS_LIST]
  --run-name <NAME>        iteration tag  [$RUN_NAME]
  --base-input-dir <DIR>   root dir       [$BASE_INPUT_DIR]
EOF
      exit 0 ;;
    --) shift; break ;;
    *) die "Unknown flag $1" ;;
  esac
done

split_csv() { IFS=',' read -r -a "$2" <<< "$1"; }
split_csv "$MODELS_LIST"   MODELS
split_csv "$DATASETS_LIST" DATASETS

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    INPUT_DIR="$BASE_INPUT_DIR/$DATASET/$MODEL/$RUN_NAME"
    OUTPUT_RUN_NAME="${RUN_NAME}_positives_negatives"

    echo "dataset=$DATASET | model=$MODEL"

    python src/retriever/extract_positives_negatives.py \
      --input_dir      "$INPUT_DIR" \
      --output_dir     "$BASE_INPUT_DIR" \
      --output_run_name "$OUTPUT_RUN_NAME"
  done
done
