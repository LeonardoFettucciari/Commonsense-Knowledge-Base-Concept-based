import os
import json
import glob
import argparse

import datetime
import sys


# Setup logging to file + stdout
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(log_dir, f"merge_accuracy_{timestamp}.log")

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


def parse_args():
    parser = argparse.ArgumentParser(description="Merge xf_acc|.jsonl accuracies across models per dataset.")
    parser.add_argument("--source_root", type=str, default="outputs/inference", help="Root folder containing inference outputs.")
    parser.add_argument("--experiments", type=str, required=True, help="Comma-separated list of experiment names.")
    parser.add_argument("--models", type=str, default="", help="Comma-separated list of models to consider. If empty, use all models.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to write merged jsonl files.")
    return parser.parse_args()

def find_latest_accuracy_file(model_path, experiment_name):
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

    # Show what we are using:
    print(f"      [LOOKUP] Using accuracy file: {acc_files[0]}")
    return acc_files[0]


def load_accuracy(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            return data
    return None

def main():
    args = parse_args()
    experiment_names = [x.strip() for x in args.experiments.split(",")]
    model_filter = [x.strip() for x in args.models.split(",")] if args.models else []

    datasets = sorted([d for d in os.listdir(args.source_root) if os.path.isdir(os.path.join(args.source_root, d))])

    print(f"Datasets found: {datasets}")
    print(f"Experiments: {experiment_names}")
    print(f"Models filter: {model_filter if model_filter else '(ALL)'}")
    print(f"Output base dir: {args.output_dir}")

    for dataset in datasets:
        print(f"\n🔹 Processing dataset: {dataset}")
        dataset_out_dir = os.path.join(args.output_dir, dataset, "data")
        os.makedirs(dataset_out_dir, exist_ok=True)

        for experiment_name in experiment_names:
            print(f"  Processing experiment: {experiment_name}")

            out_file_name = f"{experiment_name}.jsonl"
            out_file_path = os.path.join(dataset_out_dir, out_file_name)

            dataset_path = os.path.join(args.source_root, dataset)
            models_path = dataset_path
            models = sorted([m for m in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, m))])

            if model_filter:
                models = [m for m in models if m in model_filter]

            acc_values = []
            first_file_data = None

            print(f"    Models: {models}")

            for model in models:
                model_path = os.path.join(models_path, model)
                acc_file = find_latest_accuracy_file(model_path, experiment_name)
                if acc_file is None:
                    print(f"      ⚠️  No accuracy file found for model '{model}'")
                    continue

                data = load_accuracy(acc_file)
                if data is None:
                    print(f"      ⚠️  Empty accuracy file for model '{model}'")
                    continue

                acc = data.get("xfinder_average_accuracy")
                if acc is not None:
                    acc_values.append(acc)
                    if first_file_data is None:
                        first_file_data = data

                print(f"      ✅  {model} - {acc_file} - accuracy={acc:.4f}")

            if not acc_values:
                print(f"      ⚠️  No valid accuracies found for experiment '{experiment_name}'")
                continue

            avg_accuracy = sum(acc_values) / len(acc_values)

            output_data = {k: v for k, v in first_file_data.items()
                           if k not in ["xfinder_avg_accuracy_llama", "xfinder_avg_accuracy_qwen", "xfinder_average_accuracy"]}

            output_data["experiment_name"] = experiment_name
            output_data["dataset"] = dataset
            output_data["xfinder_average_accuracy"] = avg_accuracy

            with open(out_file_path, "w", encoding="utf-8") as fout:
                fout.write(json.dumps(output_data) + "\n")

            print(f"      ✅  Avg accuracy = {avg_accuracy:.4f} written to {out_file_name}")

    print(f"\n✅ All done. One file per experiment per dataset written under: {args.output_dir}")


if __name__ == "__main__":
    main()
