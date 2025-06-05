import os
import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
from src.utils.io_utils import load_jsonl, save_jsonl


def merge_jsonl(source1_path: str, source2_path: str, output_dir: str):
    source1_data = load_jsonl(source1_path)
    source2_data = load_jsonl(source2_path)

    merged_data = defaultdict(lambda: {"synset_lemma": "", "synset_definition": "", "statements": []})

    # First pass: source 1
    for entry in source1_data:
        name = entry["synset_name"]
        merged_data[name]["synset_lemma"] = entry["synset_lemma"]
        merged_data[name]["synset_definition"] = entry["synset_definition"]
        merged_data[name]["statements"].extend(entry["statements"])

    # Second pass: source 2
    for entry in source2_data:
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

    # === Construct output filename ===
    def base_no_ext(path):
        return os.path.splitext(os.path.basename(path))[0]

    base1 = base_no_ext(source1_path)
    base2 = base_no_ext(source2_path)
    output_filename = f"{base1}+{base2}.jsonl"

    output_path = os.path.join(output_dir, output_filename)
    save_jsonl(merged_ckb, output_path)
    logging.info(f"Merged file saved to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Merge two CKB JSONL files.")
    parser.add_argument("--source1_path", type=str, required=True, help="Path to source 1 JSONL file.")
    parser.add_argument("--source2_path", type=str, required=True, help="Path to source 2 JSONL file.")
    parser.add_argument("--output_dir", type=str, required=False, help="Directory to store the merged output.")

    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = os.path.dirname(args.source1_path)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.info("Merging JSONL files from source 1 and source 2...")
    merge_jsonl(args.source1_path, args.source2_path, args.output_dir)
