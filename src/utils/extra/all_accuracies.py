#!/usr/bin/env python3
import os
import json
import glob
import argparse
import datetime
import sys

# ----------------------------
# Logging setup (stdout + file)
# ----------------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(log_dir, f"merge_accuracy_with_individuals_{timestamp}.log")

class Tee:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to Tee
sys.stdout = Tee(log_file_path)
print(f"[LOGGING] All output will be logged to: {log_file_path}\n")


# ----------------------------
# Arg parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate accuracies across models for multiple experiments, "
                    "saving individual per-model scores and their average."
    )
    parser.add_argument("--source_root", type=str, default="outputs/inference",
                        help="Root folder containing inference outputs.")
    parser.add_argument("--experiments", type=str, required=True,
                        help="Comma-separated list of experiment names.")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated list of models to consider. If empty, use all models.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where output JSONL(s) will be written.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Name of the consolidated JSONL file (default: summary_<timestamp>.jsonl).")
    parser.add_argument("--per_dataset", action="store_true",
                        help="Also write one JSONL per dataset (under <output_dir>/<dataset>/data/).")
    return parser.parse_args()


# ----------------------------
# Helpers
# ----------------------------
def find_latest_accuracy_file(model_path, experiment_name):
    """
    Returns the first xf_acc*.jsonl file found in the most recent date folder:
    <model_path>/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>/accuracy/xf_acc*.jsonl
    """
    exp_path = os.path.join(model_path, experiment_name)
    if not os.path.isdir(exp_path):
        print(f"      [LOOKUP] Experiment path does not exist: {exp_path}")
        return None

    date_folders = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]
    if not date_folders:
        print(f"      [LOOKUP] No date folders found in: {exp_path}")
        return None

    latest_date = sorted(date_folders)[-1]
    accuracy_path = os.path.join(exp_path, latest_date, "accuracy")
    print(f"      [LOOKUP] Checking accuracy path: {accuracy_path}")

    acc_files = glob.glob(os.path.join(accuracy_path, "xf_acc*.jsonl"))
    if not acc_files:
        print(f"      [LOOKUP] No xf_acc*.jsonl files found in: {accuracy_path}")
        return None

    print(f"      [LOOKUP] Using accuracy file: {acc_files[0]}")
    return acc_files[0]


def load_first_jsonl_line(file_path):
    """Load the first JSON object from a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            return json.loads(line)
    return None


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    experiment_names = [x.strip() for x in args.experiments.split(",") if x.strip()]
    model_filter = [x.strip() for x in args.models.split(",")] if args.models else []

    os.makedirs(args.output_dir, exist_ok=True)
    consolidated_name = args.output_file or f"summary_{timestamp}.jsonl"
    consolidated_path = os.path.join(args.output_dir, consolidated_name)

    datasets = sorted([d for d in os.listdir(args.source_root) if os.path.isdir(os.path.join(args.source_root, d))])

    print(f"Datasets found: {datasets}")
    print(f"Experiments: {experiment_names}")
    print(f"Models filter: {model_filter if model_filter else '(ALL)'}")
    print(f"Output base dir: {args.output_dir}")
    print(f"Consolidated JSONL: {consolidated_path}")

    total_written = 0
    with open(consolidated_path, "w", encoding="utf-8") as consolidated_out:
        for dataset in datasets:
            print(f"\n🔹 Processing dataset: {dataset}")

            # Optional per-dataset file
            if args.per_dataset:
                dataset_out_dir = os.path.join(args.output_dir, dataset, "data")
                os.makedirs(dataset_out_dir, exist_ok=True)
                ds_file_name = f"experiments_summary_{timestamp}.jsonl"
                ds_file_path = os.path.join(dataset_out_dir, ds_file_name)
                ds_handle = open(ds_file_path, "w", encoding="utf-8")
                print(f"    [WRITE] Per-dataset JSONL: {ds_file_path}")
            else:
                ds_handle = None

            dataset_path = os.path.join(args.source_root, dataset)
            models = sorted([m for m in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, m))])
            if model_filter:
                models = [m for m in models if m in model_filter]

            print(f"    Models considered: {models if models else '(none)'}")

            for experiment_name in experiment_names:
                print(f"  ▶ Experiment: {experiment_name}")
                individual_scores = {}  # model -> accuracy
                first_file_data = None

                for model in models:
                    model_path = os.path.join(dataset_path, model)
                    acc_file = find_latest_accuracy_file(model_path, experiment_name)
                    if acc_file is None:
                        print(f"      ⚠️  No accuracy file found for model '{model}'")
                        continue

                    data = load_first_jsonl_line(acc_file)
                    if data is None:
                        print(f"      ⚠️  Empty accuracy file for model '{model}'")
                        continue

                    # Use the field name from your original script
                    acc = data.get("xfinder_average_accuracy")
                    if acc is None:
                        print(f"      ⚠️  Missing 'xfinder_average_accuracy' in {acc_file} for model '{model}'")
                        continue

                    individual_scores[model] = acc
                    if first_file_data is None:
                        first_file_data = data

                    print(f"      ✅  {model} - {acc_file} - accuracy={acc:.4f}")

                if not individual_scores:
                    print(f"      ⚠️  No valid accuracies found for experiment '{experiment_name}' in dataset '{dataset}'")
                    continue

                # Average across available models for this experiment
                avg_accuracy = sum(individual_scores.values()) / len(individual_scores)

                # Base payload, excluding per-model averages that might be in source
                base_payload = {}
                if first_file_data:
                    base_payload = {
                        k: v for k, v in first_file_data.items()
                        if k not in [
                            "xfinder_avg_accuracy_llama",
                            "xfinder_avg_accuracy_qwen",
                            "xfinder_average_accuracy"
                        ]
                    }

                # Final record for dataset × experiment
                record = {
                    **base_payload,
                    "dataset": dataset,
                    "experiment_name": experiment_name,
                    "models": sorted(individual_scores.keys()),
                    "individual_scores": {m: individual_scores[m] for m in sorted(individual_scores.keys())},
                    "xfinder_average_accuracy": avg_accuracy,
                    "num_models": len(individual_scores),
                }

                # Write one line to consolidated JSONL
                consolidated_out.write(json.dumps(record) + "\n")
                total_written += 1

                # Optional: also write to per-dataset file
                if ds_handle is not None:
                    ds_handle.write(json.dumps(record) + "\n")

                print(f"      ✅  Written: avg={avg_accuracy:.4f} (models={len(individual_scores)})")

            if ds_handle is not None:
                ds_handle.close()

    print(f"\n✅ Done. Wrote {total_written} lines to consolidated file: {consolidated_path}")
    if total_written == 0:
        print("⚠️  No lines written. Check your --source_root, --experiments, and --models filters.")
    print("Tip: Use --per_dataset to also generate per-dataset summaries.")
    

if __name__ == "__main__":
    main()
