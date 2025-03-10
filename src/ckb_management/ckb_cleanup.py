import argparse
import json
import os
from src.utils.io_utils import load_yaml, load_jsonl, save_jsonl

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
            synset_dict[synset_name]["statements"].update(entry["statements"])  # Merge unique statements

    # Convert sets back to lists
    return [
        {"synset_name": synset_name, "synset_lemma": details["synset_lemma"], 
         "synset_definition": details["synset_definition"], "statements": list(details["statements"])}
        for synset_name, details in synset_dict.items()
    ]


# Steps functions mapping
AVAILABLE_STEPS = {
    "remove_duplicates": remove_duplicates,
}

def cleanup_pipeline(input_path, output_dir, steps):
    # Load raw knowledge base
    data = load_jsonl(input_path)

    # Exctract steps functions
    step_functions = [AVAILABLE_STEPS[step] for step in steps if step in AVAILABLE_STEPS]

    # Apply cleaning steps
    for step in step_functions:
        data = step(data)

    # Save final output
    output_path = os.path.join(output_dir, f"cleaned_{len(steps)}_steps_{os.path.basename(input_path)}")
    save_jsonl(data, output_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup pipeline for a given Knowledge Base.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input unfiltered Knowledge Base.')
    parser.add_argument('--output_dir', default="data/ckb/cleaned", type=str, required=False, help='Path to the output filtered Knowledge Base.')
    parser.add_argument('--steps', type=str, required=False, default="remove_duplicates",help='Comma-separated list of processing steps (e.g., filter_invalid_entries,normalize_text,remove_duplicates)')
    args = parser.parse_args()

    # Convert steps argument into list
    args.steps = args.steps.split(",")
    # Create output_dir if it doesn't exist yet
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    cleanup_pipeline(**vars(args))
