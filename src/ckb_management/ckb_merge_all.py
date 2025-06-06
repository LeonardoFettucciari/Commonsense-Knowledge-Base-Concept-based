import os
import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
from src.utils.io_utils import load_jsonl, save_jsonl

def merge_jsonl_directory(source_dir: str, output_dir: str):
    jsonl_files = sorted(glob(os.path.join(source_dir, "*.jsonl")))

    if len(jsonl_files) == 0:
        logging.error(f"No JSONL files found in {source_dir}")
        return

    logging.info(f"Found {len(jsonl_files)} JSONL files to merge.")

    merged_data = defaultdict(lambda: {"synset_lemma": "", "synset_definition": "", "statements": []})

    for jsonl_path in jsonl_files:
        logging.info(f"Processing {jsonl_path}...")
        data = load_jsonl(jsonl_path)

        for entry in data:
            name = entry["synset_name"]
            if not merged_data[name]["synset_lemma"]:
                merged_data[name]["synset_lemma"] = entry["synset_lemma"]
            if not merged_data[name]["synset_definition"]:
                merged_data[name]["synset_definition"] = entry["synset_definition"]
            merged_data[name]["statements"].extend(entry["statements"])

    # Deduplicate and format
    merged_ckb = []
    for name, data in merged_data.items():
        merged_ckb.append({
            "synset_name": name,
            "synset_lemma": data["synset_lemma"],
            "synset_definition": data["synset_definition"],
            "statements": list(set(data["statements"]))
        })

    # Construct output filename
    output_filename = f"merged_{len(jsonl_files)}files.jsonl"
    output_path = os.path.join(output_dir, output_filename)
    save_jsonl(merged_ckb, output_path)

    logging.info(f"Merged file saved to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser(description="Merge all CKB JSONL files in a directory.")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing JSONL files to merge.")
    parser.add_argument("--output_dir", type=str, required=False, help="Directory to store the merged output.")

    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = args.source_dir

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.info("Merging all JSONL files from source directory...")
    merge_jsonl_directory(args.source_dir, args.output_dir)
