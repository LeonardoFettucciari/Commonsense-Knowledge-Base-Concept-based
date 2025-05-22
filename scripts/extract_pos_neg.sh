#!/usr/bin/env bash
# -----------------------------------------
# extract.sh – batch extraction driver
# -----------------------------------------
set -Eeuo pipefail
IFS=$'\n\t'

# ── defaults ────────────────────────────────────────────────────────────────
SCRIPT_PATH="src/retriever/extract_positives_negatives.py"   # <-- point this to your file1
INPUT_DIR_ROOT="outputs/inference"
OUTPUT_DIR_ROOT="data/"
DATASET_LIST="obqa,qasc,csqa"
MODELS="Llama-3.1-8B-Instruct,Qwen2.5-7B-Instruct"
RUN_NAME="default_run"
OUTPUT_RUN_NAME="$RUN_NAME"
# ────────────────────────────────────────────────────────────────────────────────

die() { printf "❌  %s\n" "$*" >&2; exit 1; }

# GNU getopt for long-flag parsing
PARSED=$(getopt -o h \
  --long help,script-path:,input-dir-root:,output-dir-root:,dataset-list:,models:,run-name:,output-run-name: \
  -- "$@") || exit 1
eval set -- "$PARSED"

while true; do
  case "$1" in
    --script-path)     SCRIPT_PATH=$2; shift 2 ;;
    --input-dir-root)  INPUT_DIR_ROOT=$2; shift 2 ;;
    --output-dir-root) OUTPUT_DIR_ROOT=$2; shift 2 ;;
    --dataset-list)    DATASET_LIST=$2; shift 2 ;;
    --models)          MODELS=$2; shift 2 ;;
    --run-name)        RUN_NAME=$2; shift 2 ;;
    --output-run-name) OUTPUT_RUN_NAME=$2; shift 2 ;;
    -h|--help)
cat <<EOF
Batch extraction runner
-----------------------
Optional (defaults in brackets):
  --script-path <FILE>       path to your extraction script [${SCRIPT_PATH}]
  --input-dir-root <DIR>     root input directory     [${INPUT_DIR_ROOT}]
  --output-dir-root <DIR>    root output directory    [${OUTPUT_DIR_ROOT}]
  --dataset-list <LIST>      comma-sep datasets       [${DATASET_LIST}]
  --models <LIST>            comma-sep models         [${MODELS}]
  --run-name <STRING>        original run tag         [${RUN_NAME}]
  --output-run-name <STRING> label for extraction run [${OUTPUT_RUN_NAME}]
  -h, --help                 show this help
EOF
      exit 0 ;;
    --) shift; break ;;
    *) die "Unknown flag: $1" ;;
  esac
done

# ── helpers ────────────────────────────────────────────────────────────────
split_csv() { IFS=',' read -r -a "$2" <<< "$1"; }

split_csv "$DATASET_LIST" DATASETS
split_csv "$MODELS" MODELS

# ── main loop ──────────────────────────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    INPUT_DIR="$INPUT_DIR_ROOT/$DATASET/$MODEL/$RUN_NAME"
    echo "➡️  Extracting: $DATASET / $MODEL / $RUN_NAME"

    python "$SCRIPT_PATH" \
      --input_dir       "$INPUT_DIR" \
      --output_dir      "$OUTPUT_DIR_ROOT" \
      --output_run_name "$OUTPUT_RUN_NAME"
  done
done
