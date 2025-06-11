#!/usr/bin/env python3

import json
import csv
import argparse
from tqdm import tqdm

def load_incorrect_ids_from_tsv(tsv_path):
    incorrect_ids = set()
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                mismatch = int(row.get("xfinder_extracted_answers_mismatch", "1"))
                acc_llama = int(row.get("xfinder_acc_llama", "1"))
            except ValueError:
                continue

            if mismatch == 0 and acc_llama == 0:
                incorrect_ids.add(row["id"])
    return incorrect_ids

def load_saved_ids(output_jsonl_path):
    saved_ids = set()
    try:
        with open(output_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                saved_ids.add(example["id"])
    except FileNotFoundError:
        # If output file doesn't exist yet, no saved ids
        pass
    return saved_ids

def main(input_jsonl_path, tsv_path, output_jsonl_path):
    # Step 1: Load incorrect ids from tsv
    print("Loading incorrect ids from TSV...")
    incorrect_ids = load_incorrect_ids_from_tsv(tsv_path)
    print(f"Found {len(incorrect_ids)} incorrect ids.")

    # Step 2: Load already saved ids
    print("Loading already saved ids from output jsonl...")
    saved_ids = load_saved_ids(output_jsonl_path)
    print(f"Found {len(saved_ids)} already saved ids.")

    # Step 3: Process input jsonl
    print("Processing input jsonl...")
    new_entries = []
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Checking examples"):
            example = json.loads(line)
            qid = example.get("id")
            if qid in incorrect_ids and qid not in saved_ids:
                new_entries.append(example)
                saved_ids.add(qid)  # avoid duplicates within this run

    print(f"Found {len(new_entries)} new entries to add.")

    # Step 4: Append new entries to output jsonl
    if new_entries:
        with open(output_jsonl_path, "a", encoding="utf-8") as f:
            for example in new_entries:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"Appended {len(new_entries)} new entries to {output_jsonl_path}.")
    else:
        print("No new entries to add.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and copy incorrect questions from JSONL to another JSONL, based on a TSV file.")
    parser.add_argument("--input_jsonl", required=True, help="Path to input JSONL file.")
    parser.add_argument("--tsv_file", required=True, help="Path to TSV file.")
    parser.add_argument("--output_jsonl", required=True, help="Path to output JSONL file where new incorrect questions will be added.")

    args = parser.parse_args()

    main(args.input_jsonl, args.tsv_file, args.output_jsonl)
