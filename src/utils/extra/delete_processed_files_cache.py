import os
from argparse import ArgumentParser
import logging

def delete_cache(input_dir: str):  
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == ".processed_files_cache.json":
                full_path = os.path.join(root, file)
                os.remove(full_path)
                logging.info(f"Deleted: {full_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Delete cache files.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory."
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Deleting cache...")
    delete_cache(**vars(args))
    logging.info("Cache deleted.")
