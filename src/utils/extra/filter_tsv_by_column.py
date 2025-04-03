import logging
import os
from src.utils.io_utils import load_local_file, save_local_file
from src.utils.string_utils import extract_key_value_pairs, key_value_pairs_to_filename

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# === Filtering Function ===
def filter_tsv_by_column_value(file_path, column_name, target_value):
    data = load_local_file(file_path)

    # Filter rows
    filtered_data = [row for row in data if row.get(column_name) == target_value]
    logging.info(f"Filtered {len(data)} rows to {len(filtered_data)} rows where {column_name} = '{target_value}'")

    if not filtered_data:
        logging.info(f"No matching rows found for {column_name} = '{target_value}'")
        return file_path

    # Save filtered data to a new TSV file
    output_dir = os.path.dirname(file_path)
    filename, _ = os.path.splitext(os.path.basename(file_path))

    filename_metadata = extract_key_value_pairs(filename)
    filename_metadata[column_name] = target_value
    filename = key_value_pairs_to_filename(filename_metadata, extension="tsv")
    
    output_path = os.path.join(output_dir, filename)

    save_local_file(filtered_data, output_path)
    logging.info(f"Filtered data saved to {output_path}")

    return output_path

# === Example usage ===

o1 = filter_tsv_by_column_value("outputs/retriever/training_data/zeroshot/obqa/Llama-3.1-8B-Instruct/accuracy/model=Llama-3.1-8B-Instruct|prompt=zs.tsv",
                           "xfinder_extracted_answers_mismatch",
                            str(0))
filter_tsv_by_column_value(o1,
                           "xfinder_acc_llama",
                            str(0))