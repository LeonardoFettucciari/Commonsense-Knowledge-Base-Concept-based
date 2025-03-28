import re
from tqdm import tqdm

def merge_ckb(destination_ckb, source_ckb, source_model):
    """
    Merge two CKBs by adding statements from source CKB to destination CKB.
    Specify the source model to extract statements correctly.
    A progress bar is shown during the merge process.
    """

    for destination, source in tqdm(zip(destination_ckb, source_ckb),
                                    total=len(destination_ckb), desc="Merging CKBs"):
        ckb_statements = extract_ckb_statements(source, source_model)
        add_statements_to_ckb(ckb_statements, destination)

    return destination_ckb


def extract_ckb_statements(source, source_model):
    """
    Extract statements from a source CKB entry.
    """
    if source_model.strip().lower() == "gemini":
        return source["statements"]
    elif source_model.strip().lower() == "chatgpt":
        return clean_statement(source['response']['body']['choices'][0]['message']['content'])
    else:
        raise ValueError(f"Invalid source type: {source_model}")
    
def add_statements_to_ckb(ckb_statements, destination):
    """
    Add a list of statements to a destination CKB entry.

    Args:
        ckb_statements (list): List of new statements to add.
        destination (dict): The destination entry in the CKB where statements are to be added.
        destination_type (str): A label or identifier for the type of destination (used in error messages).

    Raises:
        ValueError: If 'statements' key is not found in the destination.
    """
    try:
        destination["statements"] += ckb_statements
    except KeyError:
        raise ValueError(f"Destination ckb is missing a 'statements' field.")

    

def clean_statement(model_output):
    cleaned_statements = [s.strip() for s in re.split(r'\n?\d+\.\s*', model_output) if s.strip()]
    return cleaned_statements