import logging
import os
from argparse import ArgumentParser
from collections import defaultdict
from datasets import Dataset
from src.datasets.dataset_loader import load_local_dataset
from src.utils.io_utils import file_already_processed, mark_file_as_processed

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def extract_positives_negatives(input_path: str, output_path: str) -> None:
    logging.info(f"Loading dataset from: {input_path}")
    dataset = load_local_dataset(input_path)
    transformed_data = []

    grouped_data = defaultdict(list)
    for sample in dataset:
        grouped_data[sample['id']].append(sample)

    logging.info(f"Grouped dataset into {len(grouped_data)} unique IDs")

    for group_id, group in grouped_data.items():
        if len(group) == 20:
            new_sample = {
                'id': group_id,
                'question': group[0]['question'], 
                'choices': group[0]['choices'], 
                'ground_truth': group[0]['ground_truth'], 
                'positives': [],
                'negatives': []
            }

            for sample in group:
                if sample['xfinder_extracted_answers_mismatch'] == 0:
                    if sample['xfinder_acc_llama'] == 1:
                        new_sample['positives'].append(sample['ckb_statements'])
                    elif sample['xfinder_acc_llama'] == 0:
                        new_sample['negatives'].append(sample['ckb_statements'])

            transformed_data.append(new_sample)

    logging.info(f"Extracted {len(transformed_data)} training samples")

    dataset = Dataset.from_list(transformed_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_json(output_path)
    logging.info(f"Saved positives/negatives to: {output_path}")


def anchor_format(input_path: str, output_path: str) -> None:
    logging.info(f"Formatting anchor data from: {input_path}")
    dataset = load_local_dataset(input_path)
    dataset = dataset.select_columns(['question', 'positives', 'negatives'])
    
    triplets = []
    for sample in dataset:
        triplets.extend([
            {
                "anchor": sample["question"],
                "positive": pos,
                "negative": sample["negatives"],
            }
            for pos in sample['positives']
        ])

    dataset = Dataset.from_list(triplets)
    dataset.to_json(output_path)
    logging.info(f"Saved anchor-formatted data to: {output_path}")


def extract_training_data(input_dir: str) -> None:
    logging.info(f"Walking through input directory: {input_dir}")
    for root, dirs, files in os.walk(input_dir):
        if os.path.basename(root) == 'accuracy':
            for filename in files:
                if filename.startswith(("xf", ".")):
                    logging.debug(f"Skipping xFinder stats or cache file: {filename}")
                    continue

                input_path = os.path.join(root, filename)
                if not os.path.isfile(input_path):
                    logging.debug(f"Skipping non-file path: {input_path}")
                    continue

                if file_already_processed(input_path):
                    logging.info(f"Skipping already processed file: {filename}")
                    continue

                logging.info(f"Processing file: {filename}")
                mark_file_as_processed(input_path)

                parent_folder = os.path.dirname(root)
                output_dir = os.path.join(parent_folder, "training_data")
                os.makedirs(output_dir, exist_ok=True)

                # Output file (e.g., filename.csv or filename.json)
                output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")
                anchor_output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")

                try:
                    extract_positives_negatives(input_path, output_path)
                    anchor_format(output_path, anchor_output_path)
                except Exception as e:
                    logging.error(f"Failed to process {filename}: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference script for CKB-based QA tasks.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory to get files to extract positives/negatives from."
    )

    args = parser.parse_args()

    logging.info("Launching extraction script...")
    extract_training_data(**vars(args))
    logging.info("Extraction complete.")
