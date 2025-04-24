from pathlib import Path
import json
from tqdm import tqdm

# Paths
source_path = Path("data/ckb/raw/merged.jsonl")
dest_in_path = Path("data/ckb/cleaned/merged_filtered.jsonl")
dest_out_path = Path("data/ckb/cleaned/merged_with_names.jsonl")

# Count lines (assumes both files have the same # of non-blank lines)
total = sum(1 for _ in source_path.open("r", encoding="utf-8") if _.strip())

with source_path.open("r", encoding="utf-8") as sf, \
     dest_in_path.open("r", encoding="utf-8") as df, \
     dest_out_path.open("w", encoding="utf-8") as of:

    for src_line, dst_line in tqdm(zip(sf, df), total=total, desc="Zipping files", unit="line"):
        # skip blanks
        if not src_line.strip() or not dst_line.strip():
            continue

        src = json.loads(src_line)
        dst = json.loads(dst_line)

        # Copy the name over
        dst["synset_name"] = src.get("synset_name")

        # Write out updated record
        of.write(json.dumps(dst, ensure_ascii=False) + "\n")
