import csv
import logging
import os
import datetime
from argparse import ArgumentParser
from math import ceil
from typing import Dict, List
import torch
from tqdm import tqdm
from transformers import GenerationConfig
from src.datasets.dataset_loader import load_local_dataset, split_choices
from src.retriever.retriever import Retriever
from src.utils.io_utils import load_ckb, load_yaml, prepare_output_retriever_training
from src.utils.model_utils import batched_generate_text, load_model_and_tokenizer
from src.utils.prompt_utils import build_prompts_for_retriever_training
from src.utils.data_utils import concatenate_question_choices
from src.utils.string_utils import (
    extract_base_model_name,
    prepare_prompt_output_filename,
)
from settings.aliases import (
    DATASET_NAME_TO_TAG,
    MODEL_TAG_TO_NAME,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_latest_datetime_dir(base_dir: str) -> str:
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
    output_dir: str,
    retrieval_strategy: str,
    ckb_path: str,
    top_k: int,
    batch_size: int,
    retriever_model: str,
    rerank_type: str = None,
    filter_threshold: float = 0.85,
) -> None:
    
    LIMIT_SAMPLES = 300  # Limit samples for testing

    logging.info("=== Starting augment_incorrect_with_knowledge ===")
    # Load config files
    gen_config = load_yaml("settings/model_config.json")["generation_config"]
    # Remove sampling args if not needed
    if not gen_config.get("do_sample", False):
        gen_config.pop("top_k", None)
        gen_config.pop("top_p", None)
    # Create a new, explicit GenerationConfig
    gen_config = GenerationConfig(**gen_config)

    # Determine input run directory
    run_dir = os.path.join(
        input_dir_root,
        DATASET_NAME_TO_TAG.get(dataset_name, dataset_name),
        extract_base_model_name(model_name),
        input_run_name
    )
    logging.info("Searching for runs in: %s", run_dir)
    latest_run_dir = get_latest_datetime_dir(run_dir)

    # Locate accuracy folder
    accuracy_dir = os.path.join(latest_run_dir, "accuracy")
    logging.info("Verifying accuracy directory: %s", accuracy_dir)
    if not os.path.isdir(accuracy_dir):
        logging.error("Accuracy folder not found: %s", accuracy_dir)
        raise FileNotFoundError(f"Accuracy folder not found in {latest_run_dir}")

    # Select TSV input
    input_path = next(
        (os.path.join(accuracy_dir, f)
         for f in os.listdir(accuracy_dir)
         if f.endswith(".tsv") and prompt_name in f),
        None
    )
    if not input_path:
        logging.error("No TSV file containing '%s' in %s", prompt_name, accuracy_dir)
        raise FileNotFoundError(f"No TSV file containing '{prompt_name}' found in {accuracy_dir}")
    logging.info("Found input TSV: %s", input_path)

    # Load & preprocess dataset
    input_dataset = load_local_dataset(input_path)
    input_dataset = input_dataset.map(split_choices, remove_columns=["choices"])
    total = len(input_dataset)
    logging.info("Loaded and preprocessed %d samples", total)
    input_dataset = input_dataset.filter(
        lambda s: int(s["xfinder_extracted_answers_mismatch"]) == 0
                  and int(s["xfinder_acc_llama"]) == 0
    )
    input_dataset = input_dataset.shuffle()
    input_dataset = input_dataset.select(range(min(len(input_dataset), LIMIT_SAMPLES)))
    incorrect_count = len(input_dataset)
    logging.info("Filtered to %d incorrect samples", incorrect_count)

    # Initialize CKB and retriever
    logging.info("Loading CKB from %s with strategy %s", ckb_path, retrieval_strategy)
    ckb = load_ckb(ckb_path, retrieval_strategy)
    retriever = Retriever(
        model_name_or_path=retriever_model,
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
    logging.info("Model ready for inference")

    outputs: List[Dict[str, str]] = []
    n_batches = ceil(incorrect_count / batch_size) if batch_size else 0
    for batch in tqdm(
        input_dataset.batch(batch_size=batch_size),
        desc=f"Batched inference ({batch_size})",
        total=n_batches
    ):
        # Number of samples in this batch
        batch_len = len(next(iter(batch.values())))

        # Re-create sample dictionaries for this batch
        samples = [{k: batch[k][i] for k in batch} for i in range(batch_len)]

        # Retrieve knowledge
        queries = [concatenate_question_choices(s) for s in samples]
        batch_ckb_lists = retriever.retrieve_top_k(
            queries,
            top_k=top_k,
            batch_size=512,
            re_rank=rerank_type,
            diversity_threshold=filter_threshold,
            pool_size=top_k * 100,
        )

        # Build prompts
        batch_prompts = []
        prompt_meta = []          # (sample_dict, prompt_obj, rank)
        for sample, ckb_stmts in zip(samples, batch_ckb_lists):
            prompts = build_prompts_for_retriever_training(
                sample=sample,
                ckb_statements=ckb_stmts,
                top_k=top_k,
            )
            for rank, pr in enumerate(prompts):
                batch_prompts.append(pr)
                prompt_meta.append((sample, pr, rank))

        # Generate Outputs
        decoded_texts = batched_generate_text(
            model, tokenizer, batch_prompts, gen_config
        )

        # Collect outputs
        for text, (sample, pr, rank) in zip(decoded_texts, prompt_meta):
            outputs.append(
                prepare_output_retriever_training(sample, pr, text, rank)
            )
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
        prompt="zscotk1",
        ckb=os.path.splitext(os.path.basename(ckb_path))[0],
        retrieval_strategy=retrieval_strategy,
    )
    prompt_output_path = os.path.join(dated_output_dir, prompt_output_filename)
    with open(prompt_output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=outputs[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(outputs)
    logging.info("Wrote %d rows to %s", len(outputs), prompt_output_path)

def main() -> None:
    parser = ArgumentParser(description="Batched CKB-based QA inference.")
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
    parser.add_argument("--retriever_model", required=True,
                        type=str, help="Name or path for retriever model.")
    parser.add_argument("--top_k", default=20, type=int,
                        help="Top k retrieval count.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of samples to process per batch.")
    parser.add_argument("--filter_threshold", default=0.85, type=float,
                        help="Filter threshold value.")
    parser.add_argument("--rerank_type", type=str, default=None,
                        help="Reranking strategy, e.g. 'filter', None.")
    args = parser.parse_args()

    # Resolve any aliases
    args.model_name = MODEL_TAG_TO_NAME.get(args.model_name, args.model_name)
    args.dataset_name = DATASET_NAME_TO_TAG.get(args.dataset_name, args.dataset_name)

    logging.info("Launching augment_incorrect_with_knowledge with args: %s", vars(args))
    augment_incorrect_with_knowledge(
        input_dir_root=args.input_dir_root,
        input_run_name=args.input_run_name,
        output_run_name=args.output_run_name,
        prompt_name=args.prompt_name,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        retrieval_strategy=args.retrieval_strategy,
        ckb_path=args.ckb_path,
        top_k=args.top_k,
        batch_size=args.batch_size,
        retriever_model=args.retriever_model,
        rerank_type=args.rerank_type,
        filter_threshold=args.filter_threshold,

    )


if __name__ == "__main__":
    main()
