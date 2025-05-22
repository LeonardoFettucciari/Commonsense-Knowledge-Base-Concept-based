import csv
import logging
import os
import datetime
import torch
from argparse import ArgumentParser
from typing import Dict, List

# Local imports
from settings.aliases import (
    DATASET_NAME_TO_TAG,
    DATASET_TAG_TO_NAME,
    MODEL_TAG_TO_NAME,
)
from src.datasets.dataset_loader import load_local_dataset, split_choices
from src.retriever.retriever import Retriever
from src.utils.io_utils import load_ckb, load_yaml, prepare_output_retriever_training
from src.utils.model_utils import generate_text, load_model_and_tokenizer
from src.utils.prompt_utils import build_prompts_for_retriever_training
from src.utils.retriever_utils import retrieve_top_k_statements
from src.utils.data_utils import concatenate_question_choices
from src.utils.string_utils import (
    extract_base_model_name,
    prepare_prompt_output_filename,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_latest_datetime_dir(base_dir: str) -> str:
    """
    Scan direct subdirectories of base_dir for timestamped names
    and return the most recent. If none, returns base_dir.
    """
    dt_dirs = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        for fmt in ("%Y-%m-%d_%H-%M-%S", "%Y-%m-%d"):
            try:
                dt = datetime.datetime.strptime(name, fmt)
                dt_dirs.append((dt, path))
                break
            except ValueError:
                continue
    if not dt_dirs:
        logging.info("No timestamped directories found in %s, using base directory", base_dir)
        return base_dir
    _, latest_path = max(dt_dirs, key=lambda x: x[0])
    logging.info("Latest timestamped directory: %s", latest_path)
    return latest_path


def augment_incorrect_with_knowledge(
    input_dir_root: str,
    input_run_name: str,
    output_run_name: str,
    prompt_name: str,
    model_name: str,
    dataset_name: str,
    config_path: str,
    output_dir: str,
    retrieval_strategy: str,
    ckb_path: str,
    top_k: int,
) -> None:
    # Log start
    logging.info("=== Starting augment_incorrect_with_knowledge ===")
    logging.info("Configuration path: %s", config_path)
    config = load_yaml(config_path)

    # Determine input run directory (structured as input_dir_root/dataset/model/run-name)
    run_dir = os.path.join(
        input_dir_root,
        DATASET_NAME_TO_TAG.get(dataset_name, dataset_name),
        extract_base_model_name(model_name),
        input_run_name
    )
    logging.info("Searching for runs in: %s", run_dir)
    latest_run_dir = get_latest_datetime_dir(run_dir)
    logging.info("Using latest run directory: %s", latest_run_dir)

    # Locate accuracy folder
    accuracy_dir = os.path.join(latest_run_dir, "accuracy")
    logging.info("Verifying accuracy directory: %s", accuracy_dir)
    if not os.path.isdir(accuracy_dir):
        logging.error("Accuracy folder not found: %s", accuracy_dir)
        raise FileNotFoundError(f"Accuracy folder not found in {latest_run_dir}")

    # Select TSV input
    input_path = None
    for fname in os.listdir(accuracy_dir):
        if fname.endswith('.tsv') and prompt_name in fname:
            input_path = os.path.join(accuracy_dir, fname)
            break
    if not input_path:
        logging.error("No TSV file containing '%s' in %s", prompt_name, accuracy_dir)
        raise FileNotFoundError(f"No TSV file containing '{prompt_name}' found in {accuracy_dir}")
    logging.info("Found input TSV: %s", input_path)

    # Load and preprocess dataset
    input_dataset = load_local_dataset(input_path)
    input_dataset = input_dataset.map(
        split_choices,
        remove_columns=["choices"],    # drop the old string column
)

    total_samples = len(input_dataset)
    logging.info("Loaded and preprocessed %d samples", total_samples)

    # Filter for incorrect
    input_dataset_incorrect_samples = input_dataset.filter(
        lambda sample: int(sample['xfinder_extracted_answers_mismatch']) == 0
        and int(sample['xfinder_acc_llama']) == 0
    )
    input_dataset_incorrect_samples = input_dataset_incorrect_samples.select(range(1))
    incorrect_count = len(input_dataset_incorrect_samples)
    logging.info("Filtered to %d incorrect samples", incorrect_count)

    # Initialize CKB and retriever
    logging.info("Loading CKB from %s with strategy %s", ckb_path, retrieval_strategy)
    ckb = load_ckb(ckb_path, retrieval_strategy)
    retriever = Retriever(
            model_name_or_path=config['retriever']['model_name'],
            retrieval_strategy=retrieval_strategy,
            ckb=ckb,
            passage_prompt="passage: ",
            query_prompt="query: ",
        )
    logging.info("Retriever initialized")

    # Load model and tokenizer
    logging.info("Loading model and tokenizer: %s", model_name)
    model, tokenizer = load_model_and_tokenizer(model_name)
    model.eval()
    torch.set_grad_enabled(False)
    logging.info("Model and tokenizer loaded successfully")

    # Run inference loop
    from tqdm import tqdm
    logging.info("Beginning inference loop on %d samples", incorrect_count)
    iterator = tqdm(
        enumerate(input_dataset_incorrect_samples),
        total=incorrect_count,
        desc=f"Inferencing {model_name}",
    )

    outputs: List[Dict[str, str]] = []
    for index, sample in iterator:
        # Retrieve statements
        question_choices = concatenate_question_choices(sample)
        eval_ckb_statements = retriever.retrieve_top_k(
                question_choices,
                top_k=top_k,
                diversify=False,
            )

        # Build prompts
        prompts = build_prompts_for_retriever_training(
            sample=sample,
            ckb_statements=eval_ckb_statements,
            top_k=top_k,
        )
        # Generate answers and collect outputs
        for i, prompt in enumerate(prompts):
            answer_text = generate_text(model, tokenizer, prompt)
            outputs.append(prepare_output_retriever_training(sample, prompt, answer_text, i))
    logging.info("Inference complete: generated %d outputs", len(outputs))

    # Prepare output directory
    model_tag = DATASET_NAME_TO_TAG.get(dataset_name, dataset_name)
    model_base = extract_base_model_name(model_name)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dated_output_dir = os.path.join(
        output_dir, model_tag, model_base, output_run_name, timestamp
    )
    os.makedirs(dated_output_dir, exist_ok=True)
    logging.info("Saving outputs to: %s", dated_output_dir)

    # Write TSV
    prompt_output_filename = prepare_prompt_output_filename(
        dated_output_dir,
        output_data=outputs[0],
        prompt=prompt_name,
        ckb=os.path.splitext(os.path.basename(ckb_path))[0],
        retrieval_strategy=retrieval_strategy,
    )
    prompt_output_path = os.path.join(dated_output_dir, prompt_output_filename)
    with open(prompt_output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=outputs[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(outputs)
    logging.info("Wrote %d rows to %s", len(outputs), prompt_output_path)

    logging.info("=== augment_incorrect_with_knowledge completed ===")


def main() -> None:
    parser = ArgumentParser(description="Inference script for CKB-based QA tasks.")
    parser.add_argument("--input_dir_root", type=str, required=True,
                        help="Root directory containing runs.")
    parser.add_argument("--input_run_name", type=str, required=True,
                        help="Name of the run folder under input_dir_root.")
    parser.add_argument("--output_run_name", type=str, required=True,
                        help="Run name for output storage.")
    parser.add_argument("--prompt_name", type=str, required=True,
                        help="Substring of the prompt TSV filename to select.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name from Hugging Face.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name or tag.")
    parser.add_argument("--ckb_path", type=str, required=True,
                        help="Path to the Knowledge Base file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root path to store outputs.")
    parser.add_argument("--retrieval_strategy", type=str, required=False,
                        help="Retrieval strategy, e.g. 'retriever'.")
    parser.add_argument("--config_path", default="settings/config.yaml",
                        type=str, help="Path to config file.")
    parser.add_argument("--top_k", default=20, type=int,
                        help="Top k retrieval count.")

    args = parser.parse_args()
    args.model_name = MODEL_TAG_TO_NAME.get(args.model_name, args.model_name)
    args.dataset_name = DATASET_TAG_TO_NAME.get(args.dataset_name, args.dataset_name)

    logging.info("Launching augment_incorrect_with_knowledge with args: %s", vars(args))
    augment_incorrect_with_knowledge(
        input_dir_root=args.input_dir_root,
        input_run_name=args.input_run_name,
        output_run_name=args.output_run_name,
        prompt_name=args.prompt_name,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        config_path=args.config_path,
        output_dir=args.output_dir,
        retrieval_strategy=args.retrieval_strategy,
        ckb_path=args.ckb_path,
        top_k=args.top_k,
    )

if __name__ == "__main__":
    main()
