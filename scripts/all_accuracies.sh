#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_merge_accuracy.sh – wrapper for merge_accuracy_with_individuals.py
# -----------------------------------------------------------------------------
set -Eeuo pipefail
IFS=$'\n\t'

# ── defaults ────────────────────────────────────────────────────────────────
PY_SCRIPT="src/utils/extra/all_accuracies.py"    # path to your Python file
SOURCE_ROOT="outputs/inference"
OUTPUT_DIR="outputs/accuracies"
RUN_NAMES="zs,zscot,untrained_retriever_zscotk5_retriever_filter,untrained_retriever_zscotk5_RACo_filter"
MODELS="Llama-3.1-8B-Instruct,Llama-3.2-3B-Instruct,Qwen2.5-1.5B-Instruct,Qwen2.5-7B-Instruct"
OUTPUT_FILE=""            # leave empty to let the Python script timestamp it
PER_DATASET="false"       # pass --per_dataset when true
PYTHON_BIN="python3"      # override with a venv if you like
# ────────────────────────────────────────────────────────────────────────────

die() { printf "❌  %s\n" "$*" >&2; exit 1; }

# GNU getopt for long-flag parsing
PARSED=$(getopt -o h \
  --long help,python:,source-root:,output-dir:,run-names:,models:,output-file:,per-dataset,python-bin: \
  -- "$@") || exit 1
eval set -- "$PARSED"

while true; do
  case "$1" in
    --python)       PY_SCRIPT=$2; shift 2 ;;
    --source-root)  SOURCE_ROOT=$2; shift 2 ;;
    --output-dir)   OUTPUT_DIR=$2; shift 2 ;;
    --run-names)    RUN_NAMES=$2; shift 2 ;;
    --models)       MODELS=$2; shift 2 ;;
    --output-file)  OUTPUT_FILE=$2; shift 2 ;;
    --per-dataset)  PER_DATASET="true"; shift 1 ;;
    --python-bin)   PYTHON_BIN=$2; shift 2 ;;
    -h|--help)
cat <<EOF
merge-accuracy wrapper
----------------------
Runs merge_accuracy_with_individuals.py with convenient defaults.

Options (defaults in brackets):
  --python <PATH>        path to the Python script [${PY_SCRIPT}]
  --python-bin <BIN>     python executable to use [${PYTHON_BIN}]
  --source-root <DIR>    root folder of inference outputs [${SOURCE_ROOT}]
  --output-dir <DIR>     directory for consolidated outputs [${OUTPUT_DIR}]
  --run-names <LIST>     experiments (comma-sep) [${RUN_NAMES}]
  --models <LIST>        models to include (comma-sep) [${MODELS}]
  --output-file <NAME>   explicit consolidated JSONL filename [auto timestamp]
  --per-dataset          also write one JSONL per dataset [${PER_DATASET}]
  -h, --help             show this help

Notes:
- The Python script discovers datasets under --source-root automatically.
- If --output-file is omitted or empty, the script uses summary_<timestamp>.jsonl.
EOF
      exit 0 ;;
    --) shift; break ;;
    *) die "Unknown flag: $1" ;;
  esac
done

# ── helpers (pretty print CSVs) ─────────────────────────────────────────────
split_csv() { IFS=',' read -r -a "$2" <<< "$1"; }
split_csv "$RUN_NAMES" RUNS_ARR
split_csv "$MODELS" MODELS_ARR

# ── sanity checks ───────────────────────────────────────────────────────────
[[ -f "$PY_SCRIPT" ]] || die "Python script not found: $PY_SCRIPT"
[[ -n "$RUN_NAMES" ]] || die "You must provide at least one experiment via --run-names"
mkdir -p "$OUTPUT_DIR"

# ── display config ─────────────────────────────────────────────────────────
echo "📋 Running merge with config:"
echo "  Python script:  $PY_SCRIPT"
echo "  Python binary:  $PYTHON_BIN"
echo "  Source root:    $SOURCE_ROOT"
echo "  Output dir:     $OUTPUT_DIR"
echo "  Experiments:    ${RUNS_ARR[*]}"
echo "  Models:         ${MODELS_ARR[*]}"
echo "  Output file:    ${OUTPUT_FILE:-<auto>}"
echo "  Per-dataset:    $PER_DATASET"
echo

# ── build args & execute ───────────────────────────────────────────────────
ARGS=(
  --source_root "$SOURCE_ROOT"
  --experiments "$RUN_NAMES"
  --models "$MODELS"
  --output_dir "$OUTPUT_DIR"
)

# optional flags
if [[ -n "${OUTPUT_FILE}" ]]; then
  ARGS+=( --output_file "$OUTPUT_FILE" )
fi
if [[ "${PER_DATASET}" == "true" ]]; then
  ARGS+=( --per_dataset )
fi

# Run
set -x
"$PYTHON_BIN" "$PY_SCRIPT" "${ARGS[@]}"
set +x
