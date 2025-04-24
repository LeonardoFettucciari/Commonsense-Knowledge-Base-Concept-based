import os
import json
from collections import defaultdict
from src.datasets.dataset_loader import load_local_dataset

def merge_samples(examples):
    merged_data = {}

    for example in examples:
        qid = example["id"]

        if qid not in merged_data:
            # Initialize new record
            merged_data[qid] = {
                "id": qid,
                "question": example["question"],
                "choices": example["choices"],
                "ground_truth": example["ground_truth"],
                "positives": set(example.get("positives", [])),
                "negatives": set(example.get("negatives", [])),
            }
        else:
            # Merge positives and negatives
            merged_data[qid]["positives"].update(example.get("positives", []))
            merged_data[qid]["negatives"].update(example.get("negatives", []))

    # Ensure no overlap between positives and negatives
    for item in merged_data.values():
        overlap = item["positives"] & item["negatives"]
        if overlap:
            print(f"Removing {len(overlap)} overlapping items from positives for ID {item['id']}")
        item["negatives"] -= overlap  # Remove overlapping items from negatives
        item["positives"] = list(item["positives"])
        item["negatives"] = list(item["negatives"])

    return list(merged_data.values())



def find_and_concatenate_training_data(root_path, output_path):
    output_file = os.path.join(output_path, os.path.basename(root_path) + ".jsonl")
    datasets_list = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        if os.path.basename(dirpath) == "training_data":
            jsonl_files = [f for f in filenames if f.endswith(".jsonl") and not f.startswith("triplets")]

            for jsonl_file in jsonl_files:
                full_path = os.path.join(dirpath, jsonl_file)
                print(f"Loading dataset from: {full_path}")
                dataset = load_local_dataset(full_path)
                datasets_list.append(dataset)

    if datasets_list:
        # Flatten all datasets into a single list of examples
        all_examples = [item for dataset in datasets_list for item in dataset]
        merged_examples = merge_samples(all_examples)

        with open(output_file, "w", encoding="utf-8") as out_f:
            for item in merged_examples:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Merged dataset saved to {output_file}")
    else:
        print("No training_data folders with .jsonl files found.")

def concatenate_training_data(*args):
        
    datasets_list = []

    
    for input_path in args:
        print(f"Loading dataset from: {input_path}")
        dataset = load_local_dataset(input_path)
        datasets_list.append(dataset)

    if datasets_list:
        # Flatten all datasets into a single list of examples
        all_examples = [item for dataset in datasets_list for item in dataset]
        merged_examples = merge_samples(all_examples)

        output_path = os.path.join(os.path.dirname(input_path), "retriever_trainset.jsonl") 
        with open(output_path, "w", encoding="utf-8") as out_f:
            for item in merged_examples:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Merged dataset saved to {output_path}")
    else:
        print("No training_data folders with .jsonl files found.")

# Example usage
find_and_concatenate_training_data("outputs/retriever/training_data/csqa", "outputs/retriever/training_data/final")
find_and_concatenate_training_data("outputs/retriever/training_data/obqa", "outputs/retriever/training_data/final")
find_and_concatenate_training_data("outputs/retriever/training_data/qasc", "outputs/retriever/training_data/final")

concatenate_training_data(
    "outputs/retriever/training_data/final/csqa.jsonl",
    "outputs/retriever/training_data/final/obqa.jsonl",
    "outputs/retriever/training_data/final/qasc.jsonl",
)
