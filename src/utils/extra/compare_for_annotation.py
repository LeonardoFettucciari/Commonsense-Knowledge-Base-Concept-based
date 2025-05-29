#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import random
from src.utils.io_utils import load_local_file, append_to_local_file
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
            raise FileNotFoundError(f"no TSV file with prompt '{prompt}' found in {direct_accuracy}")

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


def compare_for_annotation(input_path1,
            input_path2,
            root,
            exp1,
            exp2,
            dataset,
            model,
            output_dir,
            n=10,):
    
    input1 = load_local_file(input_path1)
    input2 = load_local_file(input_path2)

    base_fname = os.path.splitext(os.path.basename(input_path1))[0]
    kb_fname = os.path.splitext(os.path.basename(input_path2))[0]
    base_meta = extract_key_value_pairs_from_filename(base_fname)
    kb_meta = extract_key_value_pairs_from_filename(kb_fname)
    prompt1 = base_meta.get('prompt', 'unknown1')
    prompt2 = kb_meta.get('prompt', 'unknown2')
    prompt_prefix = f"{prompt1}_vs_{prompt2}"


    output_data = []
    correct2correct_samples = []
    correct2wrong_samples = []
    wrong2correct_samples = []
    wrong2wrong_samples = []

    counter = 0
    correct2correct = 0
    correct2wrong = 0
    wrong2correct = 0
    wrong2wrong = 0

    for row1, row2 in zip(input1, input2):
        counter += 1

        # Rule out mismatched IDs
        if row1["id"] != row2["id"]:
            raise ValueError(f"ID mismatch: {row1['id']} != {row2['id']}")
        # Rule out mismatched answers on exp1
        if int(row1['xfinder_extracted_answers_mismatch']) == 1:
            continue
        # Rule out mismatched answers on exp2
        if int(row2['xfinder_extracted_answers_mismatch']) == 1:
            continue

        # Prepare new row
        new_row = {
                "dataset": dataset,
                "model": model,
            }
        
        new_row["id"] = row1["id"]
        new_row["question"] = row1["question"]
        new_row["choices"] = row1["choices"]
        new_row["ground_truth"] = row1["ground_truth"]
        new_row[f"output_{prompt1}"] = row1["model_output"]
        new_row[f"output_{prompt2}"] = row2["model_output"]
        new_row["ckb_statements"] = "\n\n".join(row2["ckb_statements"].split("\n"))
        new_row[f"answer_{prompt1}"] = row1["xfinder_extracted_answer_llama"]
        new_row[f"answer_{prompt2}"] = row2["xfinder_extracted_answer_llama"]
        
        # correct2correct
        if int(row1['xfinder_acc_llama']) == 1 and int(row2['xfinder_acc_llama']) == 1:
            correct2correct += 1
            new_row["case"] = "✅✅"
            correct2correct_samples.append(new_row)
            continue

        # correct2wrong
        if int(row1['xfinder_acc_llama']) == 1 and int(row2['xfinder_acc_llama']) == 0:
            correct2wrong += 1
            new_row["case"] = "✅❌"
            correct2wrong_samples.append(new_row)
            continue

        # wrong2correct
        if int(row1['xfinder_acc_llama']) == 0 and int(row2['xfinder_acc_llama']) == 1:
            wrong2correct += 1
            new_row["case"] = "❌✅"
            wrong2correct_samples.append(new_row)
            continue

        # wrong2wrong
        if int(row1['xfinder_acc_llama']) == 0 and int(row2['xfinder_acc_llama']) == 0:
            wrong2wrong += 1
            new_row["case"] = "❌❌"
            wrong2wrong_samples.append(new_row)
            continue
        
        

    for samples in [correct2correct_samples, correct2wrong_samples, wrong2correct_samples, wrong2wrong_samples]:
        random.shuffle(samples)  # Shuffle samples for randomness
        output_data.extend(random.sample(samples, min(n, len(samples))))  # Take up to N samples







    

    

    # Determine experiment names from paths
    experiment_prefix = f"{exp1}_vs_{exp2}"

    # Output folder structure
    out_dir = os.path.join(output_dir, experiment_prefix)
    os.makedirs(out_dir, exist_ok=True)

    # Build output filename
    base_meta.pop('prompt', None)
    kb_meta.pop('prompt', None)
    output_filename = f"{prompt_prefix}|{kwargs_to_filename(extension='tsv', **kb_meta)}"

    print(f"✅ Saved output in: {out_dir}")
    fieldnames = ["dataset", "model", "case", "id", "question", "choices", "ground_truth", f"output_{prompt1}", f"output_{prompt2}", "ckb_statements", f"answer_{prompt1}", f"answer_{prompt2}"]
    append_to_local_file(output_data, os.path.join(out_dir, output_filename), fieldnames=fieldnames)

    print(f"✅ Compared: {os.path.basename(input_path1)} vs {os.path.basename(input_path2)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--exp1", required=True)
    parser.add_argument("--prompt1", required=True)
    parser.add_argument("--exp2", required=True)
    parser.add_argument("--prompt2", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n", required=False, default=10, type=int,
                        help="Number of samples to include in the output for each case (default: 10).")
    args = parser.parse_args()

    args.model = MODEL_TAG_TO_NAME.get(args.model, args.model)
    args.model = extract_base_model_name(args.model)

    root = os.path.join("outputs", "inference", args.dataset, args.model)
    path1 = find_latest_accuracy_file(os.path.join(root, args.exp1), args.prompt1)
    path2 = find_latest_accuracy_file(os.path.join(root, args.exp2), args.prompt2)

    compare_for_annotation(
        input_path1=path1,
        input_path2=path2,
        root=root,
        exp1=args.exp1,
        exp2=args.exp2,
        dataset=args.dataset,
        model=args.model,
        output_dir=args.output_dir,
        n=args.n
    )

if __name__ == "__main__":
    main()
