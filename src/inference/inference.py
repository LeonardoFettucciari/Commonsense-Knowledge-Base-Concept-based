import csv
import logging
import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List

import tqdm

# Local imports
from settings.aliases import (
    DATASET_NAME_TO_TAG,
    DATASET_TAG_TO_NAME,
    MODEL_TAG_TO_NAME,
    PROMPT_TYPE_ALIASES,
)
from src.datasets.dataset_loader import QADataset
from src.retriever.retriever import Retriever
from src.utils.data_utils import concatenate_question_choices, synsets_from_samples
from src.utils.io_utils import load_ckb, load_yaml, prepare_output
from src.utils.model_utils import generate_text, load_model_and_tokenizer
from src.utils.prompt_utils import build_prompts, get_prompt_requirements
from src.utils.retriever_utils import (
    add_ckb_statements_to_samples,
    retrieve_top_k_statements,
)
from src.utils.string_utils import (
    extract_base_model_name,
    prepare_model_output_path,
    prepare_prompt_output_path,
    extract_value_from_key_in_file_name,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def inference(
    model_name: str,
    dataset_name: str,
    config_path: str,
    output_dir: str,
    retrieval_scope: str,
    ckb_path: str,
    prompt_types: List[str],
    top_k_values: List[int],
) -> None:
    """
    Run inference on a dataset using a specified model and optionally retrieve
    knowledge base statements to build the prompts.

    :param model_name: Name or path of the model to use (Hugging Face or local).
    :param dataset_name: Name or tag of the dataset to be loaded.
    :param config_path: Path to the YAML configuration file.
    :param output_dir: Directory to save output files.
    :param retrieval_scope: Category or scope for retrieving statements (e.g., 'synset').
    :param ckb_path: Path to the Knowledge Base (CKB) file.
    :param prompt_types: List of prompt types/aliases to use.
    :param top_k_values: List of top-k values for retrieval from the CKB.
    """
    logging.info("Starting inference process...")
    logging.info("Loading configuration from: %s", config_path)
    config = load_yaml(config_path)

    # Resolve prompt type aliases
    resolved_prompt_types = [
        PROMPT_TYPE_ALIASES.get(t.lower(), t.lower()) for t in prompt_types
    ]
    prompt_requires = get_prompt_requirements(resolved_prompt_types)

    # Get dataset tag and load dataset
    dataset_tag = DATASET_NAME_TO_TAG[dataset_name]
    logging.info("Loading dataset: %s", dataset_name)
    eval_dataset = QADataset(config[dataset_tag])
    logging.info("Loaded %d samples for evaluation.", len(eval_dataset.samples))

    # Load few-shot data if required
    fewshot_dataset = None
    if prompt_requires["fewshot"]:
        fewshot_key = f"{dataset_tag}_fewshot"
        logging.info("Loading fewshot examples...")
        fewshot_dataset = QADataset(config[fewshot_key])
        logging.info("Loaded %d fewshot examples.", len(fewshot_dataset.samples))

    # Load knowledge base data if required
    ckb = None
    retriever = None
    if prompt_requires["knowledge"]:
        # Load knowledge base
        ckb = load_ckb(ckb_path, retrieval_scope)

        # Initialize retriever
        retriever = Retriever(retrieval_scope, ckb, config["retriever"])

        # Retrieve statements for few-shot samples if required
        if prompt_requires["fewshot"]:
            for sample in fewshot_dataset.samples:
                fewshot_ckb_statements = retrieve_top_k_statements(
                    retriever, sample, ckb, max(top_k_values), retrieval_scope
                )
                add_ckb_statements_to_samples(sample, fewshot_ckb_statements)

    # Load model and tokenizer
    logging.info("Loading model and tokenizer: %s", model_name)
    model, tokenizer = load_model_and_tokenizer(model_name)
    logging.info("Model and tokenizer loaded successfully.")

    # Prepare for inference
    logging.info("Starting inference...")
    iterator = tqdm.tqdm(
        enumerate(eval_dataset.samples),
        total=len(eval_dataset.samples),
        desc=f"Running inference on {model_name}...",
    )

    ground_truths: List[str] = []
    answers: Dict[str, List[str]] = defaultdict(list)
    outputs: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    # Inference loop
    for index, sample in iterator:
        # Retrieve statements if required
        if prompt_requires["knowledge"]:
            eval_ckb_statements = retrieve_top_k_statements(
                retriever, sample, ckb, max(top_k_values), retrieval_scope
            )
            add_ckb_statements_to_samples(sample, eval_ckb_statements)

        # Build prompts
        prompts = build_prompts(
            sample=sample,
            prompt_types=resolved_prompt_types,
            top_k_values=top_k_values,
            fewshot_examples=fewshot_dataset.samples if fewshot_dataset else None,
        )

        # Generate answers for each prompt
        for prompt in prompts:
            answer_text = generate_text(model, tokenizer, prompt)
            answers[prompt.name].append(answer_text)
            outputs[prompt.name].append(prepare_output(sample, prompt, answer_text))

        # Append ground truth
        ground_truths.append(sample["answerKey"])

    # Save inference results
    model_output_path = prepare_model_output_path(output_dir, dataset_tag, extract_base_model_name(model_name))
    os.makedirs(model_output_path, exist_ok=True)
    logging.info("Saving inference results to: %s", model_output_path)

    for prompt_name, output_data in outputs.items():
        # Prepare file path
        prompt_output_path = prepare_prompt_output_path(
            model_output_path,
            extension="tsv",
            ckb=extract_value_from_key_in_file_name(os.path.basename(ckb_path), "ckb_data"),
            retrieval_scope=retrieval_scope,
            model=extract_base_model_name(model_name),
            prompt=prompt_name,
        )

        # Write data to TSV file
        with open(prompt_output_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=output_data[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(output_data)
        logging.info(
            "Saved results for prompt type '%s' to %s", prompt_name, prompt_output_path
        )

    logging.info("Inference process completed successfully.")


def main() -> None:
    """
    Parse arguments and launch the inference procedure.
    """
    parser = ArgumentParser(description="Inference script for CKB-based QA tasks.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name from Hugging Face."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name from Hugging Face."
    )
    parser.add_argument(
        "--ckb_path",
        type=str,
        required=True,
        help="Path to the Knowledge Base file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to store the outputs."
    )
    parser.add_argument(
        "--retrieval_scope",
        type=str,
        required=False,
        help="Retrieval scope, e.g., statements to use for reranking.",
    )
    parser.add_argument(
        "--config_path",
        default="settings/config.yaml",
        type=str,
        required=False,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--prompt_types",
        default="all",
        type=str,
        required=False,
        help="Comma-separated list of prompt types to use.",
    )
    parser.add_argument(
        "--top_k_values",
        default="1,3,5,10,20",
        type=str,
        required=False,
        help="Comma-separated list of top-k values to use for retrieval.",
    )

    args = parser.parse_args()

    # Replace eventual aliases
    args.model_name = MODEL_TAG_TO_NAME.get(args.model_name, args.model_name)
    args.dataset_name = DATASET_TAG_TO_NAME.get(args.dataset_name, args.dataset_name)

    # Convert prompt types and top_k values into lists
    args.prompt_types = args.prompt_types.split(",")
    args.top_k_values = [int(val) for val in args.top_k_values.split(",")]

    logging.info("Launching inference script...")
    inference(
        **vars(args)
    )


if __name__ == "__main__":
    main()
