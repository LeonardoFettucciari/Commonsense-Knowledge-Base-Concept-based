import re
from tqdm import tqdm
from typing import List

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
        return clean_statements(source['response']['body']['choices'][0]['message']['content'])
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

    

def split_statements(model_output):
    cleaned_statements = [s.strip() for s in re.split(r'\n+\d+\.\s*', model_output) if s.strip()]
    return cleaned_statements

def clean_statements(statements, expected_count=10):
    if isinstance(statements, str):
        # Split model output string into a list of statements
        statements = split_statements(statements)
    elif isinstance(statements, list) and all(isinstance(item, str) for item in statements):
        # Already have statements in a list
        pass

    # Strip eventual remaining '1.' from first statements
    statements[0] = statements[0].replace('1. ', '', 1)

    # Merge eventually wrongly split sentences
    # e.g. "This has been split in 22.0 lines" > ["This has been split in 2", " lines"]
    if len(statements) > expected_count:
        statements = merge_numbered_statements(statements)

    # Model speaking in first list item
    # e.g. "This is not the correct definition, here are expected_count statements based on the correct definition."
    if len(statements) == expected_count + 1:
        #print(len(statements), statements)
        statements = statements[1:]
    
    # Model censored possible harmful generation, so one item only
    # e.g. "I'm sorry I can't generate anything for that concept."
    elif len(statements) == 1 and expected_count != 1:
        statements = []

    # Check every synset has expected_count statements or 0
    if len(statements) != expected_count and len(statements) != 0:
        raise ValueError(f"{len(statements)} found: {statements}")
    
    return statements

def merge_numbered_statements(statements: List[str]) -> List[str]:
    """
    Merge any statement ending without a period ('.') with a subsequent
    statement starting with a digit.
    
    :param statements: A list of raw statements to be merged if needed.
    :return: The merged list of statements.
    """
    original_statements = statements[:]
    i = 0
    while i < len(statements) - 1:
        current = statements[i].strip()
        next_stmt = statements[i + 1].strip()

        # Only merge if current does not end with '.' and next starts with a digit
        if not current.endswith('.') and re.match(r'^\d', next_stmt):
            statements[i] = current + ' ' + next_stmt
            del statements[i + 1]
        else:
            i += 1
    if len(statements) > 10:
        print("Original statements:")
        print(original_statements)
        print("Final statements:")
        print(statements)
    return statements