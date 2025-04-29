#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# defaults for optional flags
OUTPUT_DIR="outputs/inference/trained_retriever_mmr8"
PROMPT_TYPES="zsk,zscotk"
CKB_TYPE="regular"
DATASET_LIST="obqa"
MODEL_NAMES="llama8B"
RETRIEVAL_STRATEGY_LIST="retriever"
TOP_K_VALUES="1,3,5,10,20"
RETRIEVER_MODEL="models/retriever_mnr/final"
LAMBDA=0.7
RERANK_TYPE=
DIVERSITY_THRESHOLD=0.9

usage() {
  cat <<EOF
Usage: $0 -y RERANK_TYPE -l LAMBDA [-p PROMPT_TYPES] [-c CKB_TYPE]
          [-d DATASET_LIST] [-m MODEL_NAMES] [-s RETRIEVAL_STRATEGY_LIST]
          [-k TOP_K_VALUES]

  -y  RERANK_TYPE               (required)
  -l   LAMBDA                    (default: $LAMBDA)
  -p   PROMPT_TYPES              (default: $PROMPT_TYPES)
  -c   CKB_TYPE                  (default: $CKB_TYPE)
  -d   DATASET_LIST              (default: $DATASET_LIST)
  -m   MODEL_NAMES               (default: $MODEL_NAMES)
  -s   RETRIEVAL_STRATEGY_LIST   (default: $RETRIEVAL_STRATEGY_LIST)
  -k   TOP_K_VALUES              (default: $TOP_K_VALUES)
  -r  RETRIEVER_MODEL           (default: $RETRIEVER_MODEL)
  -o   OUTPUT_DIR                (default: $OUTPUT_DIR)
  -t  DIVERSITY_THRESHOLD       (default: $DIVERSITY_THRESHOLD)
  -h   show this help and exit
EOF
}

# parse flags
while getopts ":y:l:p:c:d:m:s:k:r:o:t:h" opt; do
  case $opt in
    y) RERANK_TYPE=$OPTARG ;;
    l) LAMBDA=$OPTARG      ;;
    p) PROMPT_TYPES=$OPTARG ;;
    c) CKB_TYPE=$OPTARG    ;;
    d) DATASET_LIST=$OPTARG ;;
    m) MODEL_NAMES=$OPTARG  ;;
    s) RETRIEVAL_STRATEGY_LIST=$OPTARG ;;
    k) TOP_K_VALUES=$OPTARG ;;
    r) RETRIEVER_MODEL=$OPTARG ;;
    o) OUTPUT_DIR=$OPTARG ;;
    t) DIVERSITY_THRESHOLD=$OPTARG ;;
    h|\?) usage; exit 0 ;;
  esac
done
shift $((OPTIND-1))

# set CKB path (same as before)
if [ "$CKB_TYPE" = "vera" ]; then
  CKB_PATH="data/ckb/cleaned/merged_filtered.jsonl"
else
  CKB_PATH="data/ckb/cleaned/merged_filtered.jsonl"
fi

# split comma‐lists into arrays
IFS=',' read -r -a DATASETS <<< "$DATASET_LIST"
IFS=',' read -r -a MODEL_NAME_ARRAY <<< "$MODEL_NAMES"
IFS=',' read -r -a RETRIEVAL_STRATEGIES <<< "$RETRIEVAL_STRATEGY_LIST"

# main loop
for DATASET_NAME in "${DATASETS[@]}"; do
  for MODEL_NAME in "${MODEL_NAME_ARRAY[@]}"; do
    for RETRIEVAL_STRATEGY in "${RETRIEVAL_STRATEGIES[@]}"; do
      echo "Running inference: dataset='$DATASET_NAME', model='$MODEL_NAME', rerank_type='$RERANK_TYPE', lambda='$LAMBDA', top_k='$TOP_K_VALUES'"

      python src/inference/inference.py \
        --output_dir         "$OUTPUT_DIR" \
        --model_name         "$MODEL_NAME" \
        --dataset_name       "$DATASET_NAME" \
        --ckb_path           "$CKB_PATH" \
        --retrieval_strategy "$RETRIEVAL_STRATEGY" \
        --prompt_types       "$PROMPT_TYPES" \
        --top_k_values       "$TOP_K_VALUES" \
        --rerank_type        "$RERANK_TYPE" \
        --lambda             "$LAMBDA" \
        --retriever_model    "$RETRIEVER_MODEL" \
        --diversity_threshold "$DIVERSITY_THRESHOLD"
    done
  done
done
