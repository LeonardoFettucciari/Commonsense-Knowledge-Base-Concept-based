#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# ── defaults ────────────────────────────────────────────────────────────────
SOURCE_ROOT="outputs/inference"
OUTPUT_DIR="outputs/averages"
DATASET_LIST="obqa,qasc,csqa"
MODELS="Llama-3.1-8B-Instruct,Llama-3.2-3B-Instruct,Qwen2.5-1.5B-Instruct,Qwen2.5-7B-Instruct"
RUN_NAMES="\
zs,
zscot"
# ────────────────────────────────────────────────────────────────────────────

die() { printf "%s\n" "$*" >&2; exit 1; }

# GNU getopt for long‑flag parsing
PARSED=$(getopt -o h \
  --long help,source-root:,output-dir:,dataset-list:,models:,run-names: \
  -- "$@") || exit 1
eval set -- "$PARSED"

while true; do
  case "$1" in
    --source-root)  SOURCE_ROOT=$2; shift 2 ;;
    --output-dir)   OUTPUT_DIR=$2; shift 2 ;;
    --dataset-list) DATASET_LIST=$2; shift 2 ;;
    --models)       MODELS=$2; shift 2 ;;
    --run-names)    RUN_NAMES=$2; shift 2 ;;
    -h|--help)
cat <<EOF
Batch Accuracy Merger
-----------------------
Optional (defaults in brackets):
  --source-root <DIR>        root input directory [${SOURCE_ROOT}]
  --output-dir <DIR>         output directory [${OUTPUT_DIR}]
  --dataset-list <LIST>      datasets comma‑sep [${DATASET_LIST}]
  --models <LIST>            models comma‑sep [${MODELS}]
  --run-names <LIST>         run names comma‑sep [${RUN_NAMES}]
  -h, --help                 show this help
EOF
      exit 0 ;;
    --) shift; break ;;
    *) die "Unknown flag: $1" ;;
  esac
done

split_csv() { IFS=',' read -r -a "$2" <<< "$1"; }

split_csv "$DATASET_LIST" DATASETS
split_csv "$MODELS" MODELS
split_csv "$RUN_NAMES" RUNS

echo "Merging accuracy files with config:"
echo "  Source root:   $SOURCE_ROOT"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Datasets:      ${DATASETS[*]}"
echo "  Models:        ${MODELS[*]}"
echo "  Run names:     ${RUNS[*]}"

python src/utils/extra/avg_accuracy.py \
  --source_root "$SOURCE_ROOT" \
  --experiments "${RUN_NAMES}" \
  --models "${MODELS}" \
  --output_dir "$OUTPUT_DIR"
