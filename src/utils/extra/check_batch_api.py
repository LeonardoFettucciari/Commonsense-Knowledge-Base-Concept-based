import os
import json
import glob
import csv
import logging
from collections import defaultdict
from datasets import DatasetDict
# Local imports
from settings.aliases import (
    DATASET_NAME_TO_TAG,
    DATASET_TAG_TO_NAME,
    MODEL_TAG_TO_NAME,
    PROMPT_TYPE_ALIASES,
)
from src.datasets.dataset_loader import (
    load_hf_dataset,
    load_local_dataset,
    preprocess_dataset,
)
from src.retriever.retriever import Retriever
from src.utils.io_utils import load_ckb, load_yaml, prepare_output
from src.utils.model_utils import load_model_and_tokenizer, batched_generate_text
from src.utils.prompt_utils import build_prompts, get_prompt_requirements
from src.utils.string_utils import (
    extract_base_model_name,
    prepare_prompt_output_filename,
)
from src.utils.data_utils import concatenate_question_choices



def main(dataset_name: str,
         statement_dir: str
         ) -> None:
    


    config = load_yaml("settings/config.yaml")
    # Step 1 – Load and preprocess dataset
    dataset_tag = DATASET_NAME_TO_TAG[dataset_name]
    logging.info("Loading dataset: %s", dataset_name)
    eval_dataset = load_hf_dataset(config[dataset_tag])
    eval_dataset = preprocess_dataset(eval_dataset, dataset_tag)
    logging.info("Loaded %d samples for evaluation.", len(eval_dataset))

    output_tsv=f"outputs/batches/oracle/{dataset_tag}.tsv"

    # Step 2 – Build metadata dictionary from HF dataset
    metadata = {}
    for ex in eval_dataset:
        qid = ex["id"]
        metadata[qid] = {
            "question": ex["question"],
            "choices": ex["choices"],  # dict with "label" and "text"
            "ground_truth": ex.get("answerKey", ex.get("ground_truth"))
        }

    # Step 3 – Aggregate statements from all JSONL files
    statements_per_id = defaultdict(list)
    jsonl_files = glob.glob(os.path.join(statement_dir, f"*{dataset_tag}*.jsonl"))
    logging.info("Reading %d files from %s", len(jsonl_files), statement_dir)

    for filename in jsonl_files:
        with open(filename, "r") as f:
            for line in f:
                entry = json.loads(line)
                custom_id = entry.get("custom_id").split("question-id-")[-1]
                content = entry.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                if custom_id and content:
                    statements_per_id[custom_id].extend(content.split(" [SEP] "))

    # Step 4 – Write final TSV
    with open(output_tsv, "w", newline='') as out_f:
        tsv_writer = csv.writer(out_f, delimiter='\t')
        tsv_writer.writerow(["id", "question", "choices", "ground_truth", "ckb_statements"])

        for qid, statements in statements_per_id.items():
            if qid in metadata:
                merged_statements = "\n".join(statements)
                row = metadata[qid]
                tsv_writer.writerow([
                    qid,
                    row["question"],
                    json.dumps(row["choices"]),  # Convert choices dict to JSON string
                    row["ground_truth"],
                    merged_statements
                ])
            else:
                logging.warning("custom_id %s not found in HF metadata", qid)

    logging.info("✅ TSV written to: %s", output_tsv)


    output_jsonl = f"outputs/batches/oracle/{dataset_tag}.jsonl"
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for qid, statements in statements_per_id.items():
            if qid in metadata:
                row = metadata[qid]
                example = {
                    "id": qid,
                    "question": row["question"],
                    "choices": row["choices"],  # keep as dict
                    "ground_truth": row["ground_truth"],
                    "ckb_statements": statements  # list of strings
                }
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            else:
                logging.warning("custom_id %s not found in HF metadata", qid)

    logging.info("✅ JSONL written to: %s", output_jsonl)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(dataset_name="csqa",
         statement_dir="outputs/batches/oracle/results",
         )
