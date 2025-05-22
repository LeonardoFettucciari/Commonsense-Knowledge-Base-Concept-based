import logging
import os
import datetime
from argparse import ArgumentParser
from collections import defaultdict
from datasets import Dataset
from src.datasets.dataset_loader import load_local_dataset, preprocess_dataset
from src.utils.io_utils import file_already_processed, mark_file_as_processed


# ———— UTILS —————————————————————————————————————————————————————————————

def get_latest_datetime_dir(base_dir: str) -> str:
    """
    Scan direct subdirectories of base_dir, parse those named YYYY-MM-DD_HH-MM-SS,
    and return the path to the most recent one. If none found, return base_dir.
    """
    dt_dirs = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            dt = datetime.datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            continue
        dt_dirs.append((dt, path))
    if not dt_dirs:
        return base_dir
    return max(dt_dirs, key=lambda x: x[0])[1]


# ———— CORE TRANSFORMS —————————————————————————————————————————————————

def extract_positives_negatives(input_path: str, output_path: str) -> None:
    logging.info(f"Loading dataset from: {input_path}")
    dataset = load_local_dataset(input_path)
    transformed = []
    grouped = defaultdict(list)
    for row in dataset:
        grouped[row["id"]].append(row)

    for gid, group in grouped.items():
        if len(group) != 20:
            continue
        sample = {
            "id": gid,
            "question": group[0]["question"],
            "choices": group[0]["choices"],
            "ground_truth": group[0]["ground_truth"],
            "positives": [],
            "negatives": [],
        }
        for row in group:
            if row["xfinder_extracted_answers_mismatch"] != 0:
                continue
            if row["xfinder_acc_llama"] == 1:
                sample["positives"].append(row["ckb_statements"])
            else:
                sample["negatives"].append(row["ckb_statements"])
        transformed.append(sample)

    out_ds = Dataset.from_list(transformed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_ds.to_json(output_path, lines=True)
    logging.info(f"→ Saved positives/negatives to {output_path}")


def anchor_format(input_path: str, output_path: str) -> None:
    logging.info(f"Formatting anchor data from: {input_path}")
    dataset = load_local_dataset(input_path)
    triplets = []
    for row in dataset:
        q = row["question"]
        choices = " ".join(row["choices"].split("\n"))
        for pos in row["positives"]:
            for neg in row["negatives"]:
                triplets.append({
                    "anchor": f"{q} {choices}",
                    "positive": pos,
                    "negative": neg,
                })
    out_ds = Dataset.from_list(triplets)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_ds.to_json(output_path, lines=True)
    logging.info(f"→ Saved anchor triplets to {output_path}")


# ———— DATA WALKER —————————————————————————————————————————————————

def extract_training_data(
    accuracy_dir: str,
    output_base: str,
    output_run_name: str
) -> None:
    """
    Walks through a single 'accuracy' folder, processes each file once,
    and writes two JSONL outputs under:
      output_base / dataset / model / output_run_name / <timestamp> /
    """
    # infer dataset/model/timestamp from the accuracy_dir path
    # expected: .../<dataset>/<model>/<run-name>/<timestamp>/accuracy
    parts = accuracy_dir.rstrip(os.sep).split(os.sep)
    timestamp = parts[-2]
    model = parts[-4]
    dataset = parts[-5]

    # build your common output folder
    common_out = os.path.join(
        output_base, dataset, model, output_run_name, timestamp
    )
    os.makedirs(common_out, exist_ok=True)
    logging.info(f"→ Writing outputs to: {common_out}")

    for fname in os.listdir(accuracy_dir):
        if fname.startswith(("xf", ".")):
            continue
        in_path = os.path.join(accuracy_dir, fname)
        if not os.path.isfile(in_path) or file_already_processed(in_path):
            continue

        stem = os.path.splitext(fname)[0]
        posneg_path = os.path.join(common_out, f"{stem}.jsonl")
        triplet_path = os.path.join(common_out, f"triplets_{stem}.jsonl")

        logging.info(f"Processing {fname} → {stem}.jsonl")
        try:
            extract_positives_negatives(in_path, posneg_path)
            anchor_format(posneg_path, triplet_path)
            mark_file_as_processed(in_path)
        except Exception as e:
            logging.error(f"Failed on {fname}: {e}")


# ———— ENTRYPOINT —————————————————————————————————————————————————

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Extract positives/negatives + triplets from an xFinder accuracy folder."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Root path to your model runs, e.g. "
            "`dataset/model/run-name/` — the script will automatically "
            "pick the latest timestamp and look for its `accuracy/` subfolder."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base folder under which to save all outputs.",
    )
    parser.add_argument(
        "--output_run_name",
        type=str,
        required=True,
        help="Label to use instead of the original run-name for your outputs.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Finding latest run…")
    latest_run = get_latest_datetime_dir(args.input_dir)
    accuracy_folder = os.path.join(latest_run, "accuracy")
    logging.info(f"→ Using accuracy folder: {accuracy_folder}")

    extract_training_data(
        accuracy_dir=accuracy_folder,
        output_base=args.output_dir,
        output_run_name=args.output_run_name,
    )
    logging.info("All done.")
