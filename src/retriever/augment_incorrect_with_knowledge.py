import csv
import logging
import os
import tqdm
from argparse import ArgumentParser
from typing import Dict, List

# Local imports
from settings.aliases import (
    DATASET_NAME_TO_TAG,
    DATASET_TAG_TO_NAME,
    MODEL_TAG_TO_NAME,
    PROMPT_TYPE_ALIASES,
)
from src.datasets.dataset_loader import load_local_dataset, preprocess_dataset
from src.retriever.retriever import Retriever
from src.utils.io_utils import load_ckb, load_yaml, prepare_output_retriever_training
from src.utils.model_utils import generate_text, load_model_and_tokenizer
from src.utils.prompt_utils import build_prompts_for_retriever_training
from src.utils.retriever_utils import retrieve_top_k_statements
from src.utils.string_utils import (
    extract_base_model_name,
    prepare_prompt_output_filename,
    kwargs_to_path,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def augment_incorrect_with_knowledge(
    input_dir_root: str,
    model_name: str,
    dataset_name: str,
    config_path: str,
    output_dir: str,
    retrieval_strategy: str,
    ckb_path: str,
    top_k: int,
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
    :param top_k: List of top-k values for retrieval from the CKB.
    """
    logging.info("Starting inference process...")
    logging.info("Loading configuration from: %s", config_path)
    config = load_yaml(config_path)

    # Select zs file
    dataset_tag = DATASET_NAME_TO_TAG.get(dataset_name, dataset_name)

    input_dir = os.path.join(
        input_dir_root,
        dataset_tag,
        extract_base_model_name(model_name),
        'accuracy'
    )
    os.makedirs(input_dir, exist_ok=True)

    input_path = kwargs_to_path(
        dir=input_dir,
        extension="tsv",
        prompt="zs"
    )
    zs_dataset = load_local_dataset(input_path)
    zs_dataset = preprocess_dataset(zs_dataset, 'split_choices')


    # Filter on mismatch == 0 and accuracy == 0
    zs_dataset_incorrect_samples = zs_dataset.filter(
        lambda sample: sample['xfinder_extracted_answers_mismatch'] == 0 and sample['xfinder_acc_llama'] == 0
    )

    # TO REMOVE
    #zs_dataset_incorrect_samples = zs_dataset_incorrect_samples.select(range(1)) 

    # Load knowledge base and initialize retriever
    ckb = load_ckb(ckb_path, retrieval_strategy)
    retriever = Retriever(retrieval_strategy, ckb, config["retriever"])

    # Load model and tokenizer
    logging.info("Loading model and tokenizer: %s", model_name)
    model, tokenizer = load_model_and_tokenizer(model_name)
    logging.info("Model and tokenizer loaded successfully.")

    # Prepare for inference
    logging.info("Starting inference...")
    iterator = tqdm.tqdm(
        enumerate(zs_dataset_incorrect_samples),
        total=len(zs_dataset_incorrect_samples),
        desc=f"Running inference on {model_name}...",
    )

    ground_truths: List[str] = []
    answers: List[str] = []
    outputs: List[Dict[str, str]] = []

    # Inference loop
    for index, sample in iterator:
        # Retrieve statements if required
        ckb_statements = retrieve_top_k_statements(
            retriever, sample, ckb, top_k, retrieval_strategy
        )
        ckb_statements = ckb_statements[0] # Extract first and only list
        
        
        # Build prompts
        prompts = build_prompts_for_retriever_training(
            sample=sample,
            ckb_statements=ckb_statements,
            top_k=top_k,
        )

        # Generate answers for each prompt
        for i, prompt in enumerate(prompts):
            answer_text = generate_text(model, tokenizer, prompt)
            answers.append(answer_text)
            outputs.append(prepare_output_retriever_training(sample, prompt, answer_text, i))

        # Append ground truth
        ground_truths.append(sample["ground_truth"])

    # Save inference results
    model_output_dir = os.path.join(output_dir, dataset_tag, extract_base_model_name(model_name))
    os.makedirs(model_output_dir, exist_ok=True)
    logging.info("Saving inference results to: %s", model_output_dir)

    # Prepare file path
    prompt_output_filename = prepare_prompt_output_filename(
        model_output_dir,
        output_data=outputs[0],
        prompt="trainset_retriever",
        ckb=os.path.splitext(os.path.basename(ckb_path))[0],
        retrieval_strategy=retrieval_strategy,
    )
    prompt_output_path = os.path.join(model_output_dir, prompt_output_filename)

    # Write data to TSV file
    with open(prompt_output_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=outputs[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(outputs)

    logging.info("Inference process completed successfully.")



def main() -> None:
    """
    Parse arguments and launch the inference procedure.
    """
    parser = ArgumentParser(description="Inference script for CKB-based QA tasks.")
    parser.add_argument(
        "--input_dir_root",
        type=str,
        required=True,
        help="Root directory for input folders."
    )
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
        "--top_k",
        default="20",
        type=int,
        required=False,
        help="Top k value for statements to retrieve for each sample.",
    )

    args = parser.parse_args()

    # Replace eventual aliases
    args.model_name = MODEL_TAG_TO_NAME.get(args.model_name, args.model_name)
    args.dataset_name = DATASET_TAG_TO_NAME.get(args.dataset_name, args.dataset_name)

    logging.info("Launching augmenting script...")
    augment_incorrect_with_knowledge(
        **vars(args)
    )


if __name__ == "__main__":
    main()