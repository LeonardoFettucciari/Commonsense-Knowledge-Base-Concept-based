#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

OUTPUT_NAME="RACo"
DATASET_LIST="csqa,obqa,qasc"
RUN_NAMES="\
zs,\
zscot,\
"
GROUPING="1,1"
COLORS="pink:1,pink:2"
COLUMN_NAMES="zs,zscot"

PYTHON_SCRIPT="src/utils/extra/avg_accuracy_plot.py"

die() { printf "%s\n" "$*\n" >&2; exit 1; }

# ── long-flag parsing ───────────────────────────────────────────────────────
PARSED=$(getopt -o h --long help,dataset-list:,run-names:,grouping:,colors:,column-names:,output-name: -- "$@") || exit 1
eval set -- "$PARSED"

while true; do
  case "$1" in
    --dataset-list)  DATASET_LIST=$2; shift 2 ;;
    --run-names)     RUN_NAMES=$2;    shift 2 ;;
    --grouping)      GROUPING=$2;     shift 2 ;;
    --colors)        COLORS=$2;       shift 2 ;;
    --column-names)  COLUMN_NAMES=$2; shift 2 ;;
    --output-name)   OUTPUT_NAME=$2;  shift 2 ;;
    -h|--help)
cat <<EOF
Per-Dataset Accuracy Plot Generator
-----------------------------------
Optional flags (defaults shown):
  --dataset-list  <LIST>   datasets           [$DATASET_LIST]
  --run-names     <LIST>   run names          [$RUN_NAMES]
  --grouping      <LIST>   grouping sizes     [$GROUPING]
  --colors        <LIST>   color:shade list   [$COLORS]
  --column-names  <LIST>   x-axis labels      [$COLUMN_NAMES]
  --output-name   <STR>    output file name   [$OUTPUT_NAME]
  -h, --help               show this help
EOF
      exit 0 ;;
    --) shift; break ;;
    *) die "Unknown flag: $1" ;;
  esac
done

echo "Datasets : $DATASET_LIST"
echo "Run names: $RUN_NAMES"
python3 "$PYTHON_SCRIPT" \
  --datasets   "$DATASET_LIST" \
  --run_names  "$RUN_NAMES" \
  --grouping   "$GROUPING" \
  --colors     "$COLORS" \
  --column_names "$COLUMN_NAMES" \
  --output_name "$OUTPUT_NAME"
echo "Plotting complete."
