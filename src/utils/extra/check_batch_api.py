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


def main(dataset_name: str, statement_dir: str) -> None:
    config = load_yaml("settings/config.yaml")

    # 1) Load & preprocess the evaluation split once
    dataset_tag = DATASET_NAME_TO_TAG[dataset_name]
    logging.info("Loading dataset: %s", dataset_name)
    eval_dataset = preprocess_dataset(
        load_hf_dataset(config[dataset_tag]), dataset_tag
    )
    logging.info("Loaded %d samples for evaluation.", len(eval_dataset))

    # 2) Build metadata map once
    metadata = {
        ex["id"]: {
            "question": ex["question"],
            "choices": ex["choices"],
            "ground_truth": ex.get("answerKey", ex.get("ground_truth")),
        }
        for ex in eval_dataset
    }

    # 3) Walk through the jsonl files *one at a time*
    jsonl_files = glob.glob(os.path.join(statement_dir, f"*{dataset_tag}*.jsonl"))
    logging.info("Found %d jsonl files in %s", len(jsonl_files), statement_dir)

    os.makedirs("outputs/batches", exist_ok=True)

    for filename in jsonl_files:
        batch_tag = os.path.splitext(os.path.basename(filename))[0]  # e.g. qasc-001
        logging.info("Processing file: %s", filename)

        statements_per_id = defaultdict(list)

        # 3a) Collect statements only from this file
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                qid = entry.get("custom_id", "").split("question-id-")[-1]
                content = (
                    entry.get("response", {})
                    .get("body", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if qid and content:
                    statements_per_id[qid].extend(content.split(" [SEP] "))

        

        # 5) Write JSONL for this batch
        output_dir = "outputs/batches/contextual_ckb"
        os.makedirs(output_dir, exist_ok=True)
        out_jsonl = os.path.join(output_dir, f"{batch_tag}.jsonl")
        
        with open(out_jsonl, "w", encoding="utf-8") as out_f:
            for qid, stmts in statements_per_id.items():
                if qid not in metadata:
                    continue
                m = metadata[qid]
                out_f.write(
                    json.dumps(
                        {
                            "id": qid,
                            "question": m["question"],
                            "choices": m["choices"],
                            "ground_truth": m["ground_truth"],
                            "ckb_statements": stmts,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        logging.info("✅ JSONL written to %s", out_jsonl)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(dataset_name="csqa", statement_dir="outputs/batches/results")
