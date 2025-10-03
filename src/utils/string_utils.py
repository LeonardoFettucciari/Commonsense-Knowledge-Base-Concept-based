import logging
import os
import re
from settings.aliases import FULL_PROMPT_TO_SHORT_PROMPT

def prepare_prompt_output_filename(model_output_path: str, output_data: dict, extension: str = 'tsv', **kwargs) -> str:

    if not kwargs:
        raise ValueError("At least one key-value pair must be provided in kwargs.")

    # If not a knowledge prompt, remove knowledge-related parameters from filename
    if not output_data.get('ckb_statements'):
        kwargs.pop('ckb')
        kwargs.pop('retrieval_strategy')

    output_name = "|".join(f"{key}={value}" for key, value in kwargs.items())
    output_name = f"{output_name}.{extension}"

    return output_name

def kwargs_to_filename(extension: str = 'tsv', **kwargs) -> str:

    if not kwargs:
        raise ValueError("At least one key-value pair must be provided in kwargs.")

    output_name = "|".join(f"{key}={value}" for key, value in kwargs.items())
    output_name = f"{output_name}.{extension}"

    return output_name

def dict_to_filename(dict: dict, extension: str = None) -> str:

    if not dict:
        raise ValueError("Dict is None.")

    output_name = "|".join(f"{key}={value}" for key, value in dict.items())
    output_name = f"{output_name}.{extension}" if extension else output_name

    return output_name

def kwargs_to_path(dir: str, extension: str = 'tsv', **kwargs) -> str:

    if not kwargs:
        raise ValueError("At least one key-value pair must be provided in kwargs.")

    output_name = "|".join(f"{key}={value}" for key, value in kwargs.items())
    output_name = f"*{output_name}.{extension}"

    return os.path.join(dir, output_name)

def extract_value_from_key_in_file_name(filename, key):
    filename, _ = os.path.splitext(filename)
    # Split the filename by the '|' separator to get individual key=value parts.
    parts = filename.split('|')
    # Iterate over each part.
    for part in parts:
        # Only process parts that contain '='.
        if '=' in part:
            k, v = part.split('=', 1)  # Split into key and value (only once).
            if k == key:
                return v
    # Return None if key is not found.
    return None

def extract_key_value_pairs(filename):
    key_value_dict = {}
    
    filename, _ = os.path.splitext(filename)
    # Split the filename by '|' to get individual key=value parts.
    parts = filename.split('|')
    
    for part in parts:
        if '=' in part:
            k, v = part.split('=', 1)  # Split into key and value (only once).
            key_value_dict[k] = v
    
    return key_value_dict

def extract_key_value_pairs_from_filename(filename):
    key_value_dict = {}
    
    # Split the filename by '|' to get individual key=value parts.
    parts = filename.split('|')
    
    for part in parts:
        if '=' in part:
            k, v = part.split('=', 1)  # Split into key and value (only once).
            key_value_dict[k] = v
    
    return key_value_dict

def extract_base_model_name(model_name):
    if '/' in model_name:
        return model_name.split('/')[-1]
    return model_name


def alias_filename(filename: str) -> str:
    
    # Apply each replacement
    for long, short in FULL_PROMPT_TO_SHORT_PROMPT.items():
        filename = filename.replace(long, short)
        

    return filename