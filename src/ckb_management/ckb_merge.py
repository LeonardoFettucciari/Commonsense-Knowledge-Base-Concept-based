import os
import logging
from argparse import ArgumentParser
from src.utils.string_utils import extract_key_value_pairs, key_value_pairs_to_filename
from src.utils.ckb_utils import merge_ckb
from src.utils.io_utils import load_jsonl, save_jsonl


def merge(
        destination_ckb_path: str,
        source_ckb_path: str,
        source_model: str,
        output_dir: str
):
    destination_ckb = load_jsonl(destination_ckb_path)
    source_ckb = load_jsonl(source_ckb_path)

    merged_ckb = merge_ckb(destination_ckb, source_ckb, source_model)

    
    output_filename = "full_ckb.jsonl"
    output_path = os.path.join(output_dir, output_filename)
    save_jsonl(merged_ckb, output_path)



if __name__ == "__main__":
    """
    Parse arguments and launch the inference procedure.
    """
    parser = ArgumentParser(description="Merging script for two CKB json files.")
    parser.add_argument(
        "--destination_ckb_path",
        type=str,
        required=True,
        help="Destination CKB path."
    )
    parser.add_argument(
        "--source_ckb_path",
        type=str,
        required=True,
        help="Source CKB path."
    )
    parser.add_argument(
        "--source_model",
        type=str,
        required=True,
        help="Source model type e.g. Gemini, ChatGPT, etc."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Path to store the output."
    )
    
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = os.path.dirname(args.destination_ckb_path)
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs

    logging.info("Launching merging script...")
    merge(**vars(args))