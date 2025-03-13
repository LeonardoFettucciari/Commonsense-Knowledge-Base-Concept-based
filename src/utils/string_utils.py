import logging
import os

def prepare_prompt_output_path(model_output_path, ckb_name, prompt_name, model_name, retriever_type):
    ckb_data_value = extract_value_from_key_in_file_name(filename=ckb_name, key="ckb_data")
    return os.path.join(model_output_path, f"ckb={ckb_data_value}|retriever={retriever_type}|model={extract_model_name(model_name)}|prompt={prompt_name}.tsv")

def prepare_model_output_path(output_dir, dataset_name, model_name):
    return os.path.join(output_dir, dataset_name, extract_model_name(model_name))

def extract_value_from_key_in_file_name(filename, key):
    logging.info(f"FILENAME: {filename}")
    # Split the filename by the '|' separator to get individual key=value parts.
    parts = filename.split('|')
    # Iterate over each part.
    for part in parts:
        # Only process parts that contain '='.
        if '=' in part:
            k, v = part.split('=', 1)  # Split into key and value (only once).
            if k == key:
                logging.info(f"K, V: {k}, {v}")
                return v
    # Return None if key is not found.
    return None

def extract_model_name(model_name):
    if '/' in model_name:
        return model_name.split('/')[-1]
    return model_name
