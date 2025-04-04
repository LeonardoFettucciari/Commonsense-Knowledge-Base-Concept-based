import logging
import os
import re
from settings.aliases import FULL_PROMPT_TO_SHORT_PROMPT

def prepare_prompt_output_path(model_output_path, extension, **kwargs):
    """
    Constructs an output file path by appending key-value pairs and an extension.

    Args:
        model_output_path (str): The base directory for the output file.
        extension (str): The file extension (e.g., 'txt', 'json').
        **kwargs: Key-value pairs to include in the filename.

    Returns:
        str: The full output file path.
    """
    if not kwargs:
        raise ValueError("At least one key-value pair must be provided in kwargs.")

    output_name = ""
    if kwargs.get('prompt') and "knowledge" not in kwargs['prompt']:
        keys_to_keep = ['model', 'prompt']
        sub_dict = {k: kwargs[k] for k in keys_to_keep if k in kwargs}
        output_name = "|".join(f"{key}={value}" for key, value in sub_dict.items())
        
    else:
        output_name = "|".join(f"{key}={value}" for key, value in kwargs.items())
    
    output_name = f"{output_name}.{extension}"
    return os.path.join(model_output_path, output_name)

def prepare_prompt_output_path_refine(model_output_path, extension, **kwargs):
    """
    Constructs an output file path by appending key-value pairs and an extension.

    Args:
        model_output_path (str): The base directory for the output file.
        extension (str): The file extension (e.g., 'txt', 'json').
        **kwargs: Key-value pairs to include in the filename.

    Returns:
        str: The full output file path.
    """
    if not kwargs:
        raise ValueError("At least one key-value pair must be provided in kwargs.")
        
    output_name = "|".join(f"{key}={value}" for key, value in kwargs.items())
    output_name = f"{output_name}.{extension}"

    return os.path.join(model_output_path, output_name)

def prepare_model_output_path(output_dir, dataset_name, base_model_name):
    return os.path.join(output_dir, dataset_name, base_model_name)

def extract_value_from_key_in_file_name(filename, key):
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
    """
    Extracts all key-value pairs from a given filename string.
    
    Args:
        filename (str): The filename containing key-value pairs separated by '|'.
    
    Returns:
        dict: A dictionary containing extracted key-value pairs.
    """
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

def key_value_pairs_to_filename(key_value_dict, extension=None):
    """
    Converts a dictionary of key-value pairs into a filename string.
    
    Args:
        key_value_dict (dict): A dictionary containing key-value pairs.
    
    Returns:
        str: A filename string containing key-value pairs separated by '|'.
    """
    if extension:
        return "|".join(f"{key}={value}" for key, value in key_value_dict.items()) + f".{extension.strip('.')}"
    return "|".join(f"{key}={value}" for key, value in key_value_dict.items())

def shorten_prompt(prompt):
    for full, short in FULL_PROMPT_TO_SHORT_PROMPT.items():
        # Match pattern like "fewshot_cot_refine_10"
        match_refine = re.match(rf"^{full}_refine_(\d+)$", prompt)
        if match_refine:
            return f"{short}_refine_{match_refine.group(1)}"

        # Match pattern like "fewshot_cot_10"
        match_numbered = re.match(rf"^{full}_(\d+)$", prompt)
        if match_numbered:
            return f"{short}{match_numbered.group(1)}"

        # Exact match like "fewshot_cot"
        if prompt == full:
            return short

    return prompt  # Return original if no match found

