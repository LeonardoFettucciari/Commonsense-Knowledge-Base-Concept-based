import os
from datasets import concatenate_datasets
from src.datasets.dataset_loader import load_local_dataset


def convert_to_triplets(root_path):

    for dirpath, dirnames, filenames in os.walk(root_path):
        if os.path.basename(dirpath) == "training_data":
            # Find the first .jsonl file inside the training_data folder
            jsonl_files = [f for f in filenames if f.endswith(".jsonl")]
            for jsonl_file in jsonl_files:
                if jsonl_file.startswith('triplets'): continue
                full_path = os.path.join(dirpath, jsonl_file)
                print(f"Loading dataset from: {full_path}")
                dataset = load_local_dataset(full_path)

                # Processing
                triplet_dataset = []
                for sample in dataset:
                    for negative in sample['negative']:
                        triplet_dataset.append({
                            "anchor": sample["anchor"],
                            "positive": sample["positive"],
                            "negative": negative,
                        })
                triplet_dataset = dataset.from_list(triplet_dataset)
                output_path = os.path.join(dirpath, f"triplets_{jsonl_file}")
                triplet_dataset.to_json(output_path)


# Example usage
convert_to_triplets("outputs/retriever/training_data/csqa")