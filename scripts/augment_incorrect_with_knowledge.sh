#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# defaults
DATASET_LIST="obqa,qasc,csqa"
MODEL_LIST="llama8B,qwen7B"
RETRIEVAL_STRATEGIES="retriever"
CKB_PATH="data/ckb/cleaned/merged_filtered.jsonl"
TOP_K=20
BATCH_SIZE=2
RERANK_TYPE=""
FILTER_THRESHOLD=0.85

INPUT_DIR_ROOT="outputs/inference"
INPUT_RUN_NAME="zscot_augment_incorrect"
PROMPT_NAME="zscot"

RETRIEVER_MODEL="models/retriever_trained_iteration_0/final"
RUN_NAME="iteration_1"
OUTPUT_DIR="outputs/retriever_trainset"

die() { printf "❌  %s\n" "$*" >&2; exit 1; }

PARSED=$(getopt -o h --long help,rerank-type:,filter-threshold:,datasets:,models:,retrieval-strategies:,run-name:,input-run-name:,prompt-name:,input-dir-root:,ckb-path:,top-k:,batch-size:,retriever-model:,output-dir: -- "$@") || die
eval set -- "$PARSED"

while true; do
  case "$1" in
    --datasets)              DATASET_LIST=$2; shift 2 ;;
    --models)                MODEL_LIST=$2; shift 2 ;;
    --retrieval-strategies)  RETRIEVAL_STRATEGIES=$2; shift 2 ;;
    --run-name)              RUN_NAME=$2; shift 2 ;;
    --input-run-name)        INPUT_RUN_NAME=$2; shift 2 ;;
    --prompt-name)           PROMPT_NAME=$2; shift 2 ;;
    --input-dir-root)        INPUT_DIR_ROOT=$2; shift 2 ;;
    --ckb-path)              CKB_PATH=$2; shift 2 ;;
    --top-k)                 TOP_K=$2; shift 2 ;;
    --batch-size)            BATCH_SIZE=$2; shift 2 ;;
    --retriever-model)       RETRIEVER_MODEL=$2; shift 2 ;;
    --output-dir)            OUTPUT_DIR=$2; shift 2 ;;
    --rerank-type)           RERANK_TYPE=$2; shift 2 ;;
    --filter-threshold)      FILTER_THRESHOLD=$2; shift 2 ;;
    -h|--help)
      cat <<EOF
Generate positives/negatives for retriever training.

Important flags (comma-separated lists):
  --datasets <LIST>              [$DATASET_LIST]
  --models <LIST>                [$MODEL_LIST]
  --retrieval-strategies <LIST>  [$RETRIEVAL_STRATEGIES]

Other flags:
  --run-name <NAME>              iteration tag              [$RUN_NAME]
  --input-run-name <NAME>        inference run folder       [$INPUT_RUN_NAME]
  --prompt-name <NAME>           prompt substring           [$PROMPT_NAME]
  --input-dir-root <DIR>         inference outputs root     [$INPUT_DIR_ROOT]
  --ckb-path <FILE>              knowledge base jsonl       [$CKB_PATH]
  --retriever-model <PATH>       base retriever checkpoint  [$RETRIEVER_MODEL]
  --top-k <N>                    passages per question      [$TOP_K]
  --batch-size <N>               per-device batch size      [$BATCH_SIZE]
  --output-dir <DIR>             output root                [$OUTPUT_DIR]
  -h, --help                     show this help
EOF
      exit 0 ;;
    --) shift; break ;;
    *) die "Unknown flag $1" ;;
  esac
done

split_csv() { IFS=',' read -r -a "$2" <<< "$1"; }
split_csv "$DATASET_LIST" DATASETS
split_csv "$MODEL_LIST"   MODELS
split_csv "$RETRIEVAL_STRATEGIES" RETRIEVERS


for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for RETRIEVAL_STRATEGY in "${RETRIEVERS[@]}"; do
      echo "→ dataset=$DATASET | model=$MODEL | strategy=$RETRIEVAL_STRATEGY"

      python src/retriever/augment_incorrect_with_knowledge.py \
        --input_dir_root   "$INPUT_DIR_ROOT" \
        --input_run_name   "$INPUT_RUN_NAME" \
        --output_run_name  "$RUN_NAME" \
        --prompt_name      "$PROMPT_NAME" \
        --model_name       "$MODEL" \
        --dataset_name     "$DATASET" \
        --ckb_path         "$CKB_PATH" \
        --retrieval_strategy "$RETRIEVAL_STRATEGY" \
        --top_k            "$TOP_K" \
        --batch_size       "$BATCH_SIZE" \
        --retriever_model  "$RETRIEVER_MODEL" \
        --output_dir       "$OUTPUT_DIR" \
        --rerank_type      "$RERANK_TYPE" \
        --filter_threshold "$FILTER_THRESHOLD" \
    done
  done
done
