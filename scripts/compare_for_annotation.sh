#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'


DATASETS="obqa,csqa,qasc"
MODELS="llama8B,llama3B,qwen1.5B,qwen7B"

EXPERIMENT1="fscot"
PROMPT1="fscot"

EXPERIMENT2="tr_fscotk5_it2"
PROMPT2="fscotk5"

OUTPUT_DIR="outputs/annotations"
N_SAMPLES=10

die() { printf "%s\n" "$*" >&2; exit 1; }

# GNU getopt for long-flag parsing
PARSED=$(getopt -o h \
  --long help,python-script:,datasets:,models:,experiment1:,prompt1:,experiment2:,prompt2:,output-dir:,n: \
  -- "$@") || exit 1
eval set -- "$PARSED"

while true; do
  case "$1" in
    --python-script)  PYTHON_SCRIPT=$2; shift 2 ;;
    --datasets)       DATASETS=$2; shift 2 ;;
    --models)         MODELS=$2; shift 2 ;;
    --experiment1)   EXPERIMENT1=$2; shift 2 ;;
    --prompt1)       PROMPT1=$2; shift 2 ;;
    --experiment2)   EXPERIMENT2=$2; shift 2 ;;
    --prompt2)       PROMPT2=$2; shift 2 ;;
    --output-dir)     OUTPUT_DIR=$2; shift 2 ;;
    --n)              N_SAMPLES=$2; shift 2 ;;
    -h|--help)
      cat <<EOF
Compare experiment pairs for annotation
--------------------------------------
Optional (defaults in brackets):
  --python-script <PATH>   Python entrypoint               [$PYTHON_SCRIPT]
  --datasets <LIST>        datasets (comma-separated)      [$DATASETS]
  --models <LIST>          model names (comma-separated)   [$MODELS]
  --experiment1 <LIST>    baseline experiment            [$EXPERIMENT1]
  --prompt1 <LIST>        baseline prompt                [$PROMPT1]
  --experiment2 <LIST>    KB experiment                  [$EXPERIMENT2]
  --prompt2 <LIST>        KB prompt                      [$PROMPT2]
  --output-dir <DIR>       where to write outputs          [$OUTPUT_DIR]
  --n <INT>                number of samples per run       [$N_SAMPLES]
  -h, --help               show this help

EOF
      exit 0 ;;
    --) shift; break ;;
    *) die "Unknown flag $1" ;;
  esac
done

# Helpers
split_csv() { IFS=',' read -r -a "$2" <<< "$1"; }

# Expand CSVs into arrays
split_csv "$DATASETS"     DATASET_ARR
split_csv "$MODELS"       MODEL_ARR
split_csv "$EXPERIMENT1" EXP1_ARR
split_csv "$PROMPT1"     PR1_ARR
split_csv "$EXPERIMENT2" EXP2_ARR
split_csv "$PROMPT2"     PR2_ARR

mkdir -p "$OUTPUT_DIR"

for dataset in "${DATASET_ARR[@]}"; do
  for model in "${MODEL_ARR[@]}"; do
    for i in "${!EXP1_ARR[@]}"; do
      exp1="${EXP1_ARR[i]}"; prompt1="${PR1_ARR[i]}"
      exp2="${EXP2_ARR[i]}"; prompt2="${PR2_ARR[i]}"

      echo "   Running comparison for:"
      echo "   Dataset:     $dataset"
      echo "   Model:       $model"
      echo "   Baseline:    $exp1 ($prompt1)"
      echo "   With KB:     $exp2 ($prompt2)"
      echo "--------------------------------"

      python "src/utils/extra/compare_for_annotation.py" \
        --dataset   "$dataset" \
        --model     "$model" \
        --exp1      "$exp1" \
        --prompt1   "$prompt1" \
        --exp2      "$exp2" \
        --prompt2   "$prompt2" \
        --output_dir "$OUTPUT_DIR" \
        --n         "$N_SAMPLES"
    done
  done
done
