#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# defaults
RETRIEVER_MODEL_TO_FINETUNE="intfloat/e5-base-v2"
RUN_NAME="iteration_3"
TRAINSET_BASE_DIR="outputs/retriever_trainset"
OUTPUT_DIR="models/retriever_trained_${RUN_NAME}"

die() { printf "%s\n" "$*" >&2; exit 1; }

PARSED=$(getopt -o h --long help,retriever-model-to-finetune:,run-name:,trainset-base-dir:,output-dir: -- "$@") || die
eval set -- "$PARSED"

while true; do
  case "$1" in
    --retriever-model-to-finetune)    RETRIEVER_MODEL=$2; shift 2 ;;
    --run-name)           RUN_NAME=$2; shift 2 ;;
    --trainset-base-dir)  TRAINSET_BASE_DIR=$2; shift 2 ;;
    --output-dir)         OUTPUT_DIR=$2; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: retriever_trainer.sh --retriever-model <PATH> --run-name <NAME> [options]

Options:
  --retriever-model-to-finetune <PATH>      base model to fine-tune      (required)
  --run-name <NAME>             iteration tag                [${RUN_NAME}]
  --trainset-base-dir <DIR>     positives/negatives root     [${TRAINSET_BASE_DIR}]
  --output-dir <DIR>            explicit save dir            [models/retriever_trained_<run-name>]
  -h, --help                    show this help
EOF
      exit 0 ;;
    --) shift; break ;;
    *) die "Unknown flag $1" ;;
  esac
done

[[ -z $RETRIEVER_MODEL_TO_FINETUNE ]] && die "--retriever-model is required"

python src/retriever/retriever_trainer.py \
  --retriever_model_to_finetune   "$RETRIEVER_MODEL_TO_FINETUNE" \
  --run_name          "$RUN_NAME" \
  --trainset_base_dir "$TRAINSET_BASE_DIR" \
  --output_dir        "$OUTPUT_DIR"
