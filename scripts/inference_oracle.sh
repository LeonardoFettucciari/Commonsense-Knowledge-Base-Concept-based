#!/usr/bin/env bash
# -----------------------------------------
# run_inference.sh – batch‐aware inference driver
# -----------------------------------------
set -Eeuo pipefail
IFS=$'\n\t'


# ── defaults ────────────────────────────────────────────────────────────────
OUTPUT_DIR="outputs/inference"
PROMPT_TYPES="zscotk"
DATASET_LIST="obqa,csqa,qasc"
MODEL_NAMES="llama8B,llama3B,qwen1.5B,qwen7B"
TOP_K_VALUES="10"
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=4
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# ────────────────────────────────────────────────────────────────────────────

die() { printf "❌  %s\n" "$*" >&2; exit 1; }

# GNU getopt for long-flag parsing
PARSED=$(getopt -o h \
  --long help,prompt-types:,dataset-list:,model-names:,\
top-k-values:,output-dir:,\
run-name:,batch-size: \
  -- "$@") || exit 1
eval set -- "$PARSED"

while true; do
  case "$1" in
    --prompt-types)           PROMPT_TYPES=$2; shift 2 ;;
    --dataset-list)           DATASET_LIST=$2; shift 2 ;;
    --model-names)            MODEL_NAMES=$2; shift 2 ;;
    --top-k-values)           TOP_K_VALUES=$2; shift 2 ;;
    --output-dir)             OUTPUT_DIR=$2; shift 2 ;;
    --run-name)               RUN_NAME=$2; shift 2 ;;
    --batch-size)             BATCH_SIZE=$2; shift 2 ;;
    -h|--help)
        cat <<EOF
Batch inference runner
----------------------
Optional (defaults in brackets):
  --prompt-types <LIST>           prompt templates           [$PROMPT_TYPES]
  --dataset-list <LIST>           datasets comma-sep         [$DATASET_LIST]
  --model-names <LIST>            models comma-sep           [$MODEL_NAMES]
  --top-k-values <LIST>           K values comma-sep         [$TOP_K_VALUES]
  --output-dir <DIR>              write outputs here         [$OUTPUT_DIR]
  --run-name <STRING>             experiment tag             [$RUN_NAME]
  --batch-size <INT>              inference batch size       [$BATCH_SIZE]
  -h, --help                      show this help
EOF
        exit 0 ;;
    --) shift; break ;;
    *) die "Unknown flag $1" ;;
  esac
done

# ── helpers ────────────────────────────────────────────────────────────────
split_csv() { IFS=',' read -r -a "$2" <<< "$1"; }

split_csv "$DATASET_LIST"            DATASETS
split_csv "$MODEL_NAMES"             MODELS

mkdir -p "$OUTPUT_DIR"

# ── main loop ──────────────────────────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "▶ run-name=$RUN_NAME dataset=$DATASET  model=$MODEL prompts=$PROMPT_TYPES top-k=$TOP_K_VALUES  batch=$BATCH_SIZE"

    python src/inference/inference_oracle.py \
      --output_dir          "$OUTPUT_DIR" \
      --model_name          "$MODEL" \
      --dataset_name        "$DATASET" \
      --prompt_types        "$PROMPT_TYPES" \
      --top_k_values        "$TOP_K_VALUES" \
      --run_name            "$RUN_NAME" \
      --batch_size          "$BATCH_SIZE" \
      --timestamp           "$TIMESTAMP"
  done
done
