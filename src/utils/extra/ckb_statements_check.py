import logging
from collections import defaultdict
import re
import json
from src.utils.io_utils import load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# input_path = "data/ckb/raw/*ckb_data=wordnet|model=gpt-4o-mini.jsonl"
input_path = "data/ckb/raw/*ckb_data=wordnet_model=gemini-1.5-flash.jsonl"
output_path = "data/ckb/filtered_ckb_over_10.jsonl"

data = load_jsonl(input_path)

with open(output_path, "w", encoding="utf-8") as out_file:
    for sample in data:
        if 'gpt' in input_path:
            ckb_statements = sample["response"]["body"]["choices"][0]["message"]["content"]
            parsed_ckb_statements = [s.strip() for s in re.split(r'^\d+\.\s+', ckb_statements, flags=re.MULTILINE) if s.strip()]
        else:
            ckb_statements = sample["statements"]
            parsed_ckb_statements = ckb_statements

        if len(parsed_ckb_statements) > 10:
            out_line = {
                "synset_name": sample["synset_name"],
                "statements": parsed_ckb_statements
            }
            out_file.write(json.dumps(out_line) + "\n")

logging.info(f"Finished writing filtered samples to {output_path}")
