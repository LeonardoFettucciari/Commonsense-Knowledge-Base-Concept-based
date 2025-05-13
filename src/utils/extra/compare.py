#!/usr/bin/env python3
import os
import sys
import argparse
import glob
from src.utils.io_utils import load_local_file, save_local_file
from src.utils.string_utils import extract_key_value_pairs_from_filename, kwargs_to_filename
from settings.aliases import (
    DATASET_NAME_TO_TAG,
    DATASET_TAG_TO_NAME,
    MODEL_TAG_TO_NAME,
    PROMPT_TYPE_ALIASES,
)
from src.utils.string_utils import extract_base_model_name

import os
import glob

def find_latest_accuracy_file(base_path, prompt):
    """
    Given a base experiment path (e.g., outputs/inference/obqa/llama8B/baseline),
    this function:
      1. Uses base_path/accuracy if it exists
      2. Otherwise, finds the latest-dated subfolder (lexically sorted) under base_path,
         and uses its accuracy/ folder.
    Returns the path to the first TSV file matching the given prompt in its name.
    """
    # 1. Check if base_path/accuracy exists
    direct_accuracy = os.path.join(base_path, "accuracy")
    if os.path.isdir(direct_accuracy):
        candidates = glob.glob(os.path.join(direct_accuracy, f"*{prompt}*.tsv"))
        if candidates:
            return candidates[0]
        else:
            raise FileNotFoundError(f"No TSV file with prompt '{prompt}' found in {direct_accuracy}")

    # 2. Else, find latest dated subfolder and look there
    date_folders = [d for d in glob.glob(os.path.join(base_path, "*/")) if os.path.isdir(d)]
    if not date_folders:
        raise FileNotFoundError(f"No subfolders found in {base_path}, and no accuracy/ folder present.")

    latest_folder = sorted(date_folders)[-1]
    accuracy_dir = os.path.join(latest_folder, "accuracy")
    if not os.path.isdir(accuracy_dir):
        raise FileNotFoundError(f"No accuracy/ folder in latest subfolder: {latest_folder}")

    candidates = glob.glob(os.path.join(accuracy_dir, f"*{prompt}*.tsv"))
    if not candidates:
        raise FileNotFoundError(f"No TSV file with prompt '{prompt}' found in {accuracy_dir}")

    return candidates[0]


def compare(input_path_cot, input_path_with_knowledge, root, exp1, exp2):
    input_cot = load_local_file(input_path_cot)
    input_with_knowledge = load_local_file(input_path_with_knowledge)

    output_data = []
    counter = good_changes = bad_changes = 0

    for row_cot, row_with_knowledge in zip(input_cot, input_with_knowledge):
        counter += 1
        if row_cot["id"] != row_with_knowledge["id"]:
            raise ValueError(f"ID mismatch: {row_cot['id']} != {row_with_knowledge['id']}")
        if int(row_cot['xfinder_extracted_answers_mismatch']) == 1:
            continue
        if int(row_with_knowledge['xfinder_extracted_answers_mismatch']) == 1:
            continue
        if row_cot['xfinder_extracted_answer_llama'] == row_with_knowledge['xfinder_extracted_answer_llama']:
            continue
        if int(row_cot['xfinder_acc_llama']) == 0 and int(row_with_knowledge['xfinder_acc_llama']) == 0:
            continue

        good_change = 1 if int(row_cot['xfinder_acc_llama']) == 0 else 0
        bad_change = 1 if int(row_cot['xfinder_acc_llama']) == 1 else 0
        good_changes += good_change
        bad_changes += bad_change

        new_row = {
            "id": row_cot["id"],
            "question": row_cot["question"],
            "choices": row_cot["choices"],
            "ground_truth": row_cot["ground_truth"],
            "cot_output": row_cot["model_output"],
            "knowledge_output": row_with_knowledge["model_output"],
            "ckb_statements": row_with_knowledge["ckb_statements"],
            "cot_answer": row_cot["xfinder_extracted_answer_llama"],
            "knowledge_answer": row_with_knowledge["xfinder_extracted_answer_llama"],
            "good_change": good_change,
        }
        output_data.append(new_row)

    stats_data = [{
        "total_rows": counter,
        "good_changes_pct": round((good_changes / counter) * 100, 2) if counter > 0 else 0,
        "good_changes": good_changes,
        "bad_changes": bad_changes,
    }]

    base_fname = os.path.splitext(os.path.basename(input_path_cot))[0]
    kb_fname = os.path.splitext(os.path.basename(input_path_with_knowledge))[0]
    base_meta = extract_key_value_pairs_from_filename(base_fname)
    kb_meta = extract_key_value_pairs_from_filename(kb_fname)
    prompt1 = base_meta.get('prompt', 'unknown1')
    prompt2 = kb_meta.get('prompt', 'unknown2')
    prompt_prefix = f"{prompt1}_vs_{prompt2}"

    # Determine experiment names from paths
    experiment_prefix = f"{exp1}_vs_{exp2}"

    # Output folder structure
    out_dir = os.path.join(root, "compare_experiments", experiment_prefix)
    os.makedirs(out_dir, exist_ok=True)

    # Build output filename
    base_meta.pop('prompt', None)
    kb_meta.pop('prompt', None)
    output_filename = f"{prompt_prefix}|{kwargs_to_filename(extension='tsv', **kb_meta)}"
    output_stats_filename = f"stats|{output_filename}"


    save_local_file(output_data, os.path.join(out_dir, output_filename))
    save_local_file(stats_data, os.path.join(out_dir, output_stats_filename))

    print(f"✅ Compared: {os.path.basename(input_path_cot)} vs {os.path.basename(input_path_with_knowledge)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--exp1", required=True)
    parser.add_argument("--prompt1", required=True)
    parser.add_argument("--exp2", required=True)
    parser.add_argument("--prompt2", required=True)
    args = parser.parse_args()

    args.model = MODEL_TAG_TO_NAME.get(args.model, args.model)
    args.model = extract_base_model_name(args.model)

    root = os.path.join("outputs", "inference", args.dataset, args.model)
    path1 = find_latest_accuracy_file(os.path.join(root, args.exp1), args.prompt1)
    path2 = find_latest_accuracy_file(os.path.join(root, args.exp2), args.prompt2)

    compare(path1, path2, root, args.exp1, args.exp2)

if __name__ == "__main__":
    main()
