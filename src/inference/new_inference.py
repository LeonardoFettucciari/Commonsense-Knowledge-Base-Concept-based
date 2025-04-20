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
from src.datasets.dataset_loader import load_hf_dataset, load_local_dataset, preprocess_dataset
from src.retriever.retriever import Retriever
from src.utils.io_utils import load_ckb, load_yaml, prepare_output
from src.utils.model_utils import generate_text, load_model_and_tokenizer
from src.utils.prompt_utils import build_prompts, get_prompt_requirements
from src.utils.retriever_utils import (
    add_ckb_statements_to_samples,
    retrieve_top_k_statements,
)
from src.utils.string_utils import (
    extract_base_model_name,
    prepare_prompt_output_filename,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def inference(
    model_name: str,
    dataset_name: str,
    config_path: str,
    output_dir: str,
    retrieval_strategy: str,
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
    :param retrieval_strategy: Category or strategy for retrieving statements (e.g., 'synset').
    :param ckb_path: Path to the Knowledge Base (CKB) file.
    :param prompt_types: List of prompt types/aliases to use.
    :param top_k_values: List of top-k values for retrieval from the CKB.
    """
    logging.info("Starting inference process...")
    logging.info("Loading configuration from: %s", config_path)
    config = load_yaml(config_path)

    # Resolve prompt type requirements
    prompt_requires = get_prompt_requirements(prompt_types)

    # Get dataset tag and load dataset
    dataset_tag = DATASET_NAME_TO_TAG[dataset_name]
    logging.info("Loading dataset: %s", dataset_name)
    eval_dataset = load_hf_dataset(config[dataset_tag])
    eval_dataset = preprocess_dataset(eval_dataset, dataset_tag)
    logging.info("Loaded %d samples for evaluation.", len(eval_dataset))

    # Load few-shot data if required
    fewshot_dataset = None
    if prompt_requires["fewshot"]:
        fewshot_tag = f"{dataset_tag}_fewshot"
        logging.info("Loading fewshot examples...")
        fewshot_dataset = load_local_dataset(
            local_path=config[fewshot_tag]['path'],
            max_samples=config[fewshot_tag]['max_samples']
        )
        fewshot_dataset = preprocess_dataset(fewshot_dataset, fewshot_tag)
        logging.info("Loaded %d fewshot examples.", len(fewshot_dataset))

    # Load knowledge base data if required
    ckb = None
    retriever = None
    if prompt_requires["knowledge"]:
        # Load knowledge base
        ckb = load_ckb(ckb_path, retrieval_strategy)

        # Initialize retriever
        retriever = Retriever(
            "intfloat/e5-base-v2",
            retrieval_strategy,
            ckb,
            passage_prompt="passage: ",
            query_prompt="query: ",
        )

        # Retrieve statements for few-shot samples if required
        if prompt_requires["fewshot"]:
            for sample in fewshot_dataset:
                fewshot_ckb_statements = retriever.retrieve(sample)
                
                retrieve_top_k_statements(
                    retriever, sample, ckb, max(top_k_values), retrieval_strategy
                )
                add_ckb_statements_to_samples(sample, fewshot_ckb_statements)

    # Load model and tokenizer
    logging.info("Loading model and tokenizer: %s", model_name)
    model, tokenizer = load_model_and_tokenizer(model_name)
    logging.info("Model and tokenizer loaded successfully.")

    # Prepare for inference
    logging.info("Starting inference...")
    iterator = tqdm.tqdm(
        enumerate(eval_dataset),
        total=len(eval_dataset),
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
                retriever, sample, ckb, max(top_k_values), retrieval_strategy
            )
            add_ckb_statements_to_samples(sample, eval_ckb_statements)

        # Build prompts
        prompts = build_prompts(
            sample=sample,
            prompt_types=prompt_types,
            top_k_values=top_k_values,
            fewshot_examples=fewshot_dataset,
        )

        # Generate answers for each prompt
        for prompt in prompts:
            answer_text = generate_text(model, tokenizer, prompt)
            answers[prompt.name].append(answer_text)
            outputs[prompt.name].append(prepare_output(sample, prompt, answer_text))

        # Append ground truth
        ground_truths.append(sample["ground_truth"])

    # Save inference results
    model_output_dir = os.path.join(output_dir, dataset_tag, extract_base_model_name(model_name))
    os.makedirs(model_output_dir, exist_ok=True)
    logging.info("Saving inference results to: %s", model_output_dir)

    for prompt_name, output_data in outputs.items():
        prompt_output_filename = prepare_prompt_output_filename(
            model_output_dir,
            output_data = output_data[0],
            prompt=prompt_name,
            ckb=os.path.splitext(os.path.basename(ckb_path))[0],
            retrieval_strategy=retrieval_strategy,
        )
        prompt_output_path = os.path.join(model_output_dir, prompt_output_filename)

        # Write data to TSV file
        with open(prompt_output_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=output_data[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(output_data)
        logging.info("Saved results for prompt type '%s' to %s", prompt_name, prompt_output_path)

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
        "--retrieval_strategy",
        type=str,
        required=False,
        help="Retrieval strategy, e.g., statements to use for reranking.",
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

    # Convert prompt types into a list, mapping aliases if any found
    args.prompt_types = [
        PROMPT_TYPE_ALIASES.get(t.lower(), t.lower())
        for t in args.prompt_types.split(",")
    ]

    # Convert top k values into a list
    args.top_k_values = [
        int(val)
        for val in args.top_k_values.split(",")
    ]

    logging.info("Launching inference script...")
    inference(
        **vars(args)
    )


if __name__ == "__main__":
    main()
