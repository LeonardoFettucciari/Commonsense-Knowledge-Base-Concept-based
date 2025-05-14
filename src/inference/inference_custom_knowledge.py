import csv
import logging
import os
import json
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List
from datetime import datetime

import tqdm

from settings.prompts import (
    SYSTEM_ZEROSHOT_COT,
    SYSTEM_ZEROSHOT_COT_WITH_KNOWLEDGE,
)
from src.prompts.llama_prompt import LlamaPrompt, KnowledgePrompt

# Local imports
from settings.aliases import (
    DATASET_NAME_TO_TAG,
    DATASET_TAG_TO_NAME,
    MODEL_TAG_TO_NAME,
    PROMPT_TYPE_ALIASES,
)
from src.datasets.dataset_loader import load_hf_dataset, load_local_dataset, preprocess_dataset
from src.retriever.retriever import Retriever
from src.utils.io_utils import load_ckb, load_yaml
from src.utils.model_utils import generate_text, load_model_and_tokenizer
from src.utils.prompt_utils import get_prompt_requirements
from src.utils.data_utils import concatenate_question_choices

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _prompt_for_index_or_id(dataset: list[dict]) -> int | None:
    """Ask the user for a sample index or ID value and return the index if found."""
    dataset_size = len(dataset)
    while True:
        raw = input(f"\nEnter a sample index (0 to {dataset_size - 1}) or sample ID (or 'q' to quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        try:
            idx = int(raw)
            if 0 <= idx < dataset_size:
                return idx
            else:
                print("🔍  No sample at that index. Trying as ID...")
        except ValueError:
            pass  # Not an integer, try as ID

        # Try matching by ID
        for i, sample in enumerate(dataset):
            if str(sample.get("id")) == raw:
                return i

        print("❌  No sample found with that index or ID.")



def _prompt_for_custom_knowledge() -> List[str]:
    """Interactively collect custom knowledge statements from the user."""
    print("\n👉  Enter custom knowledge statements one per line. Press Enter on an **empty** line to finish.")
    out: List[str] = []
    while True:
        stmt = input("  > ").strip()
        if stmt == "":
            break
        out.append(stmt)
    return out


def inference(
    model_name: str,
    dataset_name: str,
    config_path: str,
    output_dir: str,
    retrieval_strategy: str,
    ckb_path: str,
    prompt_types: List[str],
    top_k_values: List[int],
    rerank_type: str,
    lambda_: float,
    retriever_model: str,
    diversity_threshold: float,
    run_name: str,
) -> None:
    """Interactive inference loop for a single‑sample, multi‑prompt evaluation."""

    logging.info("Loading configuration from %s", config_path)
    config = load_yaml(config_path)

    # Determine if the prompts will require knowledge / few‑shot
    prompt_requires = get_prompt_requirements(prompt_types)

    # ------------------------------------------------------------------
    # 🗂️  Data
    # ------------------------------------------------------------------
    dataset_tag = DATASET_NAME_TO_TAG[dataset_name]
    logging.info("Loading dataset %s", dataset_name)
    eval_dataset = load_hf_dataset(config[dataset_tag])
    eval_dataset = preprocess_dataset(eval_dataset, dataset_tag)
    logging.info("Dataset loaded (size=%d)", len(eval_dataset))

    # ------------------------------------------------------------------
    # 📚  Retriever (if needed)
    # ------------------------------------------------------------------
    retriever = None
    if prompt_requires["knowledge"]:
        ckb = load_ckb(ckb_path, retrieval_strategy)
        retriever = Retriever(
            model_name_or_path=retriever_model,
            retrieval_strategy=retrieval_strategy,
            ckb=ckb,
            passage_prompt="passage: ",
            query_prompt="query: ",
        )

    # ------------------------------------------------------------------
    # 🤗  Model
    # ------------------------------------------------------------------
    logging.info("Loading model %s", model_name)
    model, tokenizer = load_model_and_tokenizer(model_name)
    logging.info("Model loaded. Ready for interactive inference! ✨")

    # ------------------------------------------------------------------
    # 🔁  Interactive loop
    # ------------------------------------------------------------------
    while True:
        index = _prompt_for_index_or_id(eval_dataset)
        if index is None:
            print("\n👋  Exiting… have a nice day!")
            break

        sample = eval_dataset[index]
        question_choices = concatenate_question_choices(sample)

        # --------------------------------------------------------------
        # 🖨️  Print Q / choices / answer *before* knowledge statements
        # --------------------------------------------------------------
        print("\n" + "=" * 80)
        print(f"📄  SAMPLE {index}")
        print("-" * 80)
        print(f"Question: {sample.get('question', 'N/A')}")
        if "choices" in sample:
            formatted_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])])
            print("Choices :\n" + formatted_choices)
        if "ground_truth" in sample:
            print(f"Answer   : {sample['ground_truth']}")

        # --------------------------------------------------------------
        # 1️⃣  Retrieve top‑5 knowledge statements
        # --------------------------------------------------------------
        retrieved_statements: List[str] = []
        if prompt_requires["knowledge"] and retriever is not None:
            retrieved_statements = retriever.retrieve_top_k(
                question_choices,
                top_k=5,
                diversify=bool(rerank_type),
                re_rank=rerank_type,
                lambda_=lambda_,
                diversity_threshold=diversity_threshold,
            )

        print("\nTop‑5 retrieved knowledge statements:")
        if retrieved_statements:
            for i, stmt in enumerate(retrieved_statements, 1):
                print(f"  {i}. {stmt}")
        else:
            print("  (No statements retrieved.)")

        # --------------------------------------------------------------
        # 2️⃣  Ask user for custom knowledge
        # --------------------------------------------------------------
        custom_statements = _prompt_for_custom_knowledge()

        # --------------------------------------------------------------
        # 3️⃣  Build prompts
        # --------------------------------------------------------------
        prompts: List[LlamaPrompt] = []

        prompts.append(
            LlamaPrompt(
                name="zscot",
                system_instruction=SYSTEM_ZEROSHOT_COT,
                sample=sample,
                cot=True,
            )
        )

        sample_k5 = dict(sample)
        sample_k5["ckb_statements"] = retrieved_statements
        prompts.append(
            LlamaPrompt(
                name="zscot_k5",
                system_instruction=SYSTEM_ZEROSHOT_COT_WITH_KNOWLEDGE,
                sample=sample_k5,
                cot=True,
                top_k=5,
            )
        )

        prompts.append(
            KnowledgePrompt(
                name="zscot_custom",
                system_instruction=SYSTEM_ZEROSHOT_COT_WITH_KNOWLEDGE,
                sample=sample,
                ckb_statements=custom_statements,
            )
        )

        # --------------------------------------------------------------
        # 4️⃣  Generate outputs
        # --------------------------------------------------------------
        outputs: Dict[str, str] = {}
        for prompt in prompts:
            outputs[prompt.name] = generate_text(model, tokenizer, prompt)

        # --------------------------------------------------------------
        # 5️⃣  Display outputs
        # --------------------------------------------------------------
        def _show_output(title: str, text: str, kb: List[str] | None = None):
            print("\n--- " + title + " ---")
            print(text)
            if kb:
                print("\nKnowledge statements used:")
                for i, stmt in enumerate(kb, 1):
                    print(f"  {i}. {stmt}")

        _show_output("Zero‑shot CoT", outputs["zscot"])
        _show_output("Zero‑shot CoT + 5 retrieved", outputs["zscot_k5"], retrieved_statements)
        _show_output("Zero‑shot CoT + custom", outputs["zscot_custom"], custom_statements)
        print("=" * 80)


# =====================================================================================
# 🔨  ENTRY POINT
# =====================================================================================

def main() -> None:
    parser = ArgumentParser(description="Interactive inference script with knowledge support.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--ckb_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--retrieval_strategy", type=str, default="bm25")
    parser.add_argument("--config_path", type=str, default="settings/config.yaml")
    parser.add_argument("--prompt_types", type=str, default="all")
    parser.add_argument("--top_k_values", type=str, default="1,3,5,10,20")
    parser.add_argument("--lambda_", type=float, default=0.0)
    parser.add_argument("--rerank_type", type=str, default=None)
    parser.add_argument("--retriever_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--diversity_threshold", type=float, default=0.7)
    parser.add_argument("--run_name", type=str, required=True)

    args = parser.parse_args()

    # Resolve aliases --------------------------------------------------
    args.model_name = MODEL_TAG_TO_NAME.get(args.model_name, args.model_name)
    args.dataset_name = DATASET_TAG_TO_NAME.get(args.dataset_name, args.dataset_name)

    args.prompt_types = [PROMPT_TYPE_ALIASES.get(t.lower(), t.lower()) for t in args.prompt_types.split(",")]
    args.top_k_values = [int(x) for x in args.top_k_values.split(",")]

    inference(**vars(args))


if __name__ == "__main__":
    main()
