import os
from datasets import concatenate_datasets
from src.datasets.dataset_loader import load_local_dataset

# Assume this function is defined elsewhere
# from your_module import load_local_dataset

def find_and_concatenate_training_data(root_path, output_path):
    output_path = os.path.join(output_path, os.path.basename(root_path) + ".jsonl")
    datasets_list = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        if os.path.basename(dirpath) == "training_data":
            # Find the first .jsonl file inside the training_data folder
            jsonl_files = [f for f in filenames if f.endswith(".jsonl")]
            for jsonl_file in jsonl_files:
                if not jsonl_file.startswith("triplets"): continue

                full_path = os.path.join(dirpath, jsonl_file)
                print(f"Loading dataset from: {full_path}")
                dataset = load_local_dataset(full_path)
                datasets_list.append(dataset)

    if datasets_list:
        combined_dataset = concatenate_datasets(datasets_list)
        combined_dataset.to_json(output_path)
        print(f"Combined dataset saved to {output_path}")
    else:
        print("No training_data folders with .jsonl files found.")

# Example usage
find_and_concatenate_training_data("outputs/retriever/training_data/csqa", "outputs/retriever/training_data/final")
find_and_concatenate_training_data("outputs/retriever/training_data/obqa", "outputs/retriever/training_data/final")
find_and_concatenate_training_data("outputs/retriever/training_data/qasc", "outputs/retriever/training_data/final")
