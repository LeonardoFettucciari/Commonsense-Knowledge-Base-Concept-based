import argparse
import json
import os
import logging

from src.utils.io_utils import load_jsonl, save_jsonl

logger = logging.getLogger(__name__)

def remove_duplicates(data):
    synset_dict = {}
    for entry in data:
        synset_name = entry["synset_name"]
        if synset_name not in synset_dict:
            synset_dict[synset_name] = {
                "synset_lemma": entry["synset_lemma"],
                "synset_definition": entry["synset_definition"],
                "statements": set(entry["statements"])
            }
        else:
            synset_dict[synset_name]["statements"].update(entry["statements"])
    return [
        {
            "synset_name": synset_name, 
            "synset_lemma": details["synset_lemma"],
            "synset_definition": details["synset_definition"], 
            "statements": list(details["statements"])
        }
        for synset_name, details in synset_dict.items()
    ]

def merge_numbered_statements(statements):
    i = 0
    while i < len(statements) - 1:
        current = statements[i].strip()
        next_stmt = statements[i + 1].strip()
        if not current.endswith('.') and re.match(r'^\d', next_stmt):
            statements[i] = current + ' ' + next_stmt
            del statements[i + 1]
        else:
            i += 1

    if len(statements) == 11:
        statements = statements[1:]

    if len(statements) != 10:
        raise ValueError(
            f"Expected 10 statements, but got {len(statements)}:\n{statements}"
        )
    return statements

def merge_statements_step(data):
    filtered_data = []
    for idx, entry in enumerate(data):
        statements = entry.get("statements", [])
        if len(statements) > 10:
            try:
                statements = merge_numbered_statements(statements)
            except ValueError as e:
                logger.warning(f"Skipping entry at index {idx} due to merge error: {e}")
                continue

        if len(statements) < 2:
            logger.warning(f"Skipping entry at index {idx} (has only {len(statements)} statements).")
            continue
        if len(statements) != 10:
            logger.warning(f"Skipping entry at index {idx} (expected 10, got {len(statements)}).")
            continue

        entry["statements"] = statements
        filtered_data.append(entry)
    return filtered_data

# Steps functions mapping
AVAILABLE_STEPS = {
    "remove_duplicates": remove_duplicates,
    "merge_statements": merge_statements_step,
}

def cleanup_pipeline(input_path, output_dir, steps):
    # Load raw knowledge base
    data = load_jsonl(input_path)

    # Extract step functions based on user selection
    step_functions = [AVAILABLE_STEPS[step] for step in steps if step in AVAILABLE_STEPS]

    # Apply cleaning steps in order
    for step_func in step_functions:
        data = step_func(data)

    # Save final output
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    save_jsonl(data, output_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup pipeline for a given Knowledge Base.")
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to the input unfiltered Knowledge Base.'
    )
    parser.add_argument(
        '--output_dir',
        default="data/ckb/cleaned",
        type=str,
        required=False,
        help='Path to the output filtered Knowledge Base.'
    )
    parser.add_argument(
        '--steps',
        type=str,
        required=False,
        default="merge_statements,remove_duplicates",
        help='Comma-separated list of processing steps (e.g., filter_invalid_entries,normalize_text,remove_duplicates)'
    )
    args = parser.parse_args()

    # Convert steps argument into list
    args.steps = args.steps.split(",")
    # Create output_dir if it doesn't exist yet
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cleanup_pipeline(**vars(args))
