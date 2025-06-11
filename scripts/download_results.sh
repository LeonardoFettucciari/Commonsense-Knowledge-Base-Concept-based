#!/usr/bin/env bash
# -----------------------------------------
# download_accuracy.sh – batch TSV downloader
# -----------------------------------------
set -Eeuo pipefail
IFS=$'\n\t'

# ── defaults ────────────────────────────────────────────────────────────────
SOURCE_ROOT="outputs/inference"
OUTPUT_ROOT="outputs/download"
DATASET_LIST="obqa,qasc,csqa"
MODELS="Llama-3.1-8B-Instruct,Llama-3.2-3B-Instruct,Qwen2.5-1.5B-Instruct,Qwen2.5-7B-Instruct"
RUN_NAMES="fs,fscot,baselines"
NO_ACCURACY=false
# ────────────────────────────────────────────────────────────────────────────

die() { printf "❌  %s\n" "$*" >&2; exit 1; }

# GNU getopt for long‑flag parsing
PARSED=$(getopt -o h \
  --long help,source-root:,output-root:,dataset-list:,models:,run-names:,no-accuracy \
  -- "$@") || exit 1
eval set -- "$PARSED"

while true; do
  case "$1" in
    --source-root)  SOURCE_ROOT=$2; shift 2 ;;
    --output-root)  OUTPUT_ROOT=$2; shift 2 ;;
    --dataset-list) DATASET_LIST=$2; shift 2 ;;
    --models)       MODELS=$2; shift 2 ;;
    --run-names)    RUN_NAMES=$2; shift 2 ;;
    --no-accuracy)  NO_ACCURACY=true; shift ;;
    -h|--help)
cat <<EOF
Batch TSV downloader
-----------------------
Optional (defaults in brackets):
  --source-root <DIR>        root input directory [${SOURCE_ROOT}]
  --output-root <DIR>        output directory [${OUTPUT_ROOT}]
  --dataset-list <LIST>      datasets comma‑sep [${DATASET_LIST}]
  --models <LIST>            models comma‑sep [${MODELS}]
  --run-names <LIST>         run names comma‑sep [${RUN_NAMES}]
  --no-accuracy              if set, DO NOT use accuracy folder (default: false)
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
split_csv "$RUN_NAMES" RUNS

# ── main call ──────────────────────────────────────────────────────────────
echo "📋 Downloading TSV files with config:"
echo "  Source root:   $SOURCE_ROOT"
echo "  Output root:   $OUTPUT_ROOT"
echo "  Datasets:      ${DATASETS[*]}"
echo "  Models:        ${MODELS[*]}"
echo "  Run names:     ${RUNS[*]}"
echo "  Use accuracy/: $( [[ "$NO_ACCURACY" == true ]] && echo "NO" || echo "YES" )"

python src/utils/extra/download_results.py \
  --run_names "${RUNS[@]}" \
  $( [[ "$NO_ACCURACY" == true ]] && echo "--no_accuracy" || echo "" )
