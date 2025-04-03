import logging
import re
from typing import List, Dict, Any

from src.utils.io_utils import load_local_file, save_local_file
from src.utils.ckb_utils import clean_statements

logger = logging.getLogger(__name__)

def filter_ckb(
    input_path: str,
    output_path: str,
    model_name: str
) -> None:
    """
    Filter a CKB file by:
      1) Extracting statements based on the model type,
      2) Merging statements if they exceed 10 entries,
      3) Skipping any entry that contains only a single statement,
      4) Ensuring each entry ends up with exactly 10 statements.
    
    The final filtered CKB is then saved to 'output_path'.

    :param input_path: Path to the input CKB JSONL file.
    :param output_path: Path to where the filtered CKB should be saved.
    :param model_name: Model name indicating the statement extraction logic 
                       (“gemini” or “chatgpt”).
    """
    logger.info(f"Loading CKB from {input_path}")
    ckb_data = load_local_file(input_path)

    filtered_ckb = []
    total_synsets = 0
    total_statements = 0

    for idx, entry in enumerate(ckb_data):
        # Determine how to get statements based on model
        statements = entry.get('statements', [])
        statements = clean_statements(statements)

        # Keep track of valid entries
        entry['statements'] = statements
        filtered_ckb.append(entry)
        total_synsets += 1
        total_statements += len(statements)

    # Save results
    logger.info(f"Saving filtered CKB to {output_path}")
    save_local_file(filtered_ckb, output_path)

    logger.info(f"Filtering complete.")
    logger.info(f"Total valid synsets: {total_synsets}")
    logger.info(f"Total statements: {total_statements}")

if __name__ == "__main__":
    # Example usage:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Adjust paths and model names as needed:
    filter_ckb(
        input_path="data/ckb/raw/ckb_data=wordnet|model=gpt-4o-mini.jsonl",
        output_path="data/ckb/filtered/filtered_gpt-4o-mini.jsonl",
        model_name="chatgpt"
    )
    filter_ckb(
        input_path="data/ckb/raw/ckb_data=wordnet|model=gemini-2.0-flash.jsonl",
        output_path="data/ckb/filtered/filtered_gemini-2.0-flash.jsonl",
        model_name="gemini"
    )
