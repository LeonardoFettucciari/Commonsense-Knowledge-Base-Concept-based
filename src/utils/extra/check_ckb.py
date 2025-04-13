import logging
from src.utils.io_utils import load_local_file

def load_ckb(ckb_name="regular"):
    ckb_path = "data/ckb/cleaned/full_ckb_vera.jsonl" if ckb_name == "vera" else "data/ckb/cleaned/full_ckb.jsonl"
    logging.info(f"CKB in use: VERA") if ckb_name == "vera" else logging.info(f"CKB in use: REGULAR")
    return load_local_file(ckb_path)

def check(statement, ckb):
    for line in ckb:
        if statement in line['statements']:
            logging.info(f"Statement found in synset: {line['synset_name']}")
            return
    logging.info("Statement not found.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ckb_name = input("Enter name of CKB to be used:").strip()
    ckb = load_ckb(ckb_name)  # Load once


    while True:
        user_input = input("Enter a statement (or press Enter to exit): ").strip()
        if not user_input:
            break
        check(user_input, ckb)
