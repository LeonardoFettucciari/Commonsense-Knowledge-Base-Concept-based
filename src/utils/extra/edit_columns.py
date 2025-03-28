from collections import OrderedDict
from src.utils.io_utils import load_local_file, save_local_file


def remove_column(input_file, output_file, column_name):
    data = load_local_file(input_file)

    for item in data:
        item.pop(column_name, None)
        
    save_local_file(data, output_file)
    

def rename_key_preserve_order(input_file, output_file, old_key, new_key):
    data = load_local_file(input_file)
    updated_data = []

    for item in data:
        new_item = OrderedDict()
        for k, v in item.items():
            if k == old_key:
                new_item[new_key] = v
            else:
                new_item[k] = v
        updated_data.append(new_item)

    save_local_file(updated_data, output_file)


import os

def process_all_files(base_folder, column_to_remove, old_key, new_key):
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith(('.tsv', '.csv', '.jsonl', '.json')):
                input_path = os.path.join(root, file)
                
                # Write to same file or to a new file
                output_path = input_path  # or modify filename if needed

                try:
                    # Step 1: Remove column
                    remove_column(input_path, output_path, column_to_remove)

                    # Step 2: Rename key while preserving order
                    rename_key_preserve_order(output_path, output_path, old_key, new_key)

                    print(f"Processed: {input_path}")

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

process_all_files(
    base_folder="outputs/inference_refine/obqa",
    column_to_remove="prompt",
    old_key="model_output_revised",
    new_key="model_output_refine"
)
