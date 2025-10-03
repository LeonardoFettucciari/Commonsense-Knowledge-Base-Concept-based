import csv
import logging
import os
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
from transformers import GenerationConfig
import torch
import tqdm
from src.retriever.retriever import Retriever
from src.utils.data_utils import concatenate_question_choices
from src.utils.io_utils import load_ckb, load_yaml, prepare_output
from src.utils.model_utils import load_model_and_tokenizer, batched_generate_text
from src.utils.prompt_utils import build_prompts, get_prompt_requirements
from src.utils.string_utils import (
    extract_base_model_name,
    prepare_prompt_output_filename,
)
from src.datasets.dataset_loader import (
    load_hf_dataset,
    load_local_dataset,
    preprocess_dataset,
)
from settings.aliases import (
    DATASET_NAME_TO_TAG,
    DATASET_TAG_TO_NAME,
    MODEL_TAG_TO_NAME,
    PROMPT_TYPE_ALIASES,
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
    rerank_type: str,
    retriever_model: str,
    diversity_threshold: float,
    run_name: str,
    batch_size: int = 1,
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S"),
) -> None:

    logging.info("Starting inference process...")

    # Load config files
    config = load_yaml(config_path)
    gen_config = load_yaml("settings/model_config.json")["generation_config"]
    # Remove sampling args if not needed
    if not gen_config.get("do_sample", False):
        gen_config.pop("top_k", None)
        gen_config.pop("top_p", None)
    # Create a new, explicit GenerationConfig
    gen_config = GenerationConfig(**gen_config)

    # Determine prompt requirements
    prompt_requires = get_prompt_requirements(prompt_types)

    # Load dataset
    dataset_tag = DATASET_NAME_TO_TAG[dataset_name]
    logging.info("Loading dataset: %s", dataset_name)
    eval_dataset = load_hf_dataset(config[dataset_tag])
    eval_dataset = preprocess_dataset(eval_dataset, dataset_tag)
    logging.info("Loaded %d samples for evaluation.", len(eval_dataset))

    # Load fewshot examples if needed
    fewshot_dataset = None
    if prompt_requires["fewshot"] and prompt_requires["knowledge"]:
        fewshot_tag = f"{dataset_tag}_fscotk"
        fewshot_dataset = load_local_dataset(
            local_path=config[fewshot_tag]['path'],
            max_samples=config[fewshot_tag]['max_samples']
        )
        fewshot_dataset = preprocess_dataset(fewshot_dataset, fewshot_tag)
        logging.info("Loaded %d fewshot examples.", len(fewshot_dataset))
    elif prompt_requires["fewshot"]:
        fewshot_tag = f"{dataset_tag}_fewshot"
        fewshot_dataset = load_local_dataset(
            local_path=config[fewshot_tag]['path'],
            max_samples=config[fewshot_tag]['max_samples']
        )
        fewshot_dataset = preprocess_dataset(fewshot_dataset, fewshot_tag)
        logging.info("Loaded %d fewshot examples.", len(fewshot_dataset))

    # Load knowledge and retrieve
    ckb = None
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
        if prompt_requires["fewshot"]:
            for example in fewshot_dataset:
                qc = concatenate_question_choices(example)
                fs_ckb = example.get("ckb_statements").split("\n")
                if fs_ckb is None:
                    fs_ckb = retriever.retrieve_top_k(
                        qc,
                        top_k=max(top_k_values),
                        pool_size= max(top_k_values) * 2,  # Ensure enough candidates
                        re_rank=rerank_type,
                        diversity_threshold=diversity_threshold,
                    )
                example["ckb_statements"] = fs_ckb

    # Load model/tokenizer
    logging.info("Loading model and tokenizer: %s", model_name)
    model, tokenizer = load_model_and_tokenizer(model_name)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Prepare output containers
    ground_truths: List[str] = []
    answers: Dict[str, List[str]] = defaultdict(list)
    outputs: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    # Progress bar
    for batch in tqdm.tqdm(eval_dataset.batch(batch_size=batch_size), desc=f"Batched inference ({batch_size})"):
        # Batch inference
        batch_prompts: List[tuple] = []  # tuples of (sample, prompt_name, prompt_obj)

        # Retrieve & build prompts per sample
        batch_len = len(next(iter(batch.values())))
        for i in range(batch_len):
            sample = {key: batch[key][i] for key in batch}

            if prompt_requires["knowledge"]:
                qc = concatenate_question_choices(sample)
                ev_ckb = retriever.retrieve_top_k(
                    qc,
                    top_k=max(top_k_values),
                    pool_size= max(top_k_values) * 2,  # Ensure enough candidates
                    re_rank=rerank_type,
                    diversity_threshold=diversity_threshold,
                )
                sample["ckb_statements"] = ev_ckb

            # Build prompts
            prompts = build_prompts(
                sample=sample,
                prompt_types=prompt_types,
                top_k_values=top_k_values,
                fewshot_examples=fewshot_dataset,
            )
            for pr in prompts:
                batch_prompts.append((sample, pr.name, pr))

        # Generate answers in batch
        prompts = [pr for (_, _, pr) in batch_prompts]
        gen_outputs = batched_generate_text(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            gen_config=gen_config,
        )

        # Map outputs back
        for text, (sample, prompt_name, pr) in zip(gen_outputs, batch_prompts):
            answers[prompt_name].append(text)
            outputs[prompt_name].append(prepare_output(sample, pr, text))

        # Ground truths
        for i in range(batch_len):
            sample = {key: batch[key][i] for key in batch}
            ground_truths.append(sample["ground_truth"])


    # Save results
    model_output_dir = os.path.join(
        output_dir,
        dataset_tag,
        extract_base_model_name(model_name),
        run_name,
        timestamp,
    )

    os.makedirs(model_output_dir, exist_ok=True)
    logging.info("Saving inference results to: %s", model_output_dir)

    for prompt_name, out_data in outputs.items():
        filename = prepare_prompt_output_filename(
            model_output_dir,
            output_data=out_data[0],
            prompt=prompt_name,
            ckb=os.path.splitext(os.path.basename(ckb_path))[0],
            retrieval_strategy=retrieval_strategy,
        )
        path = os.path.join(model_output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_data[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(out_data)
        logging.info("Saved results for prompt type '%s' to %s", prompt_name, path)

    logging.info("Inference process completed successfully.")



def main() -> None:
    parser = ArgumentParser(description="Inference script for CKB-based QA tasks.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--ckb_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--retrieval_strategy", type=str, required=False)
    parser.add_argument("--config_path", default="settings/config.yaml", type=str)
    parser.add_argument("--prompt_types", default="all", type=str)
    parser.add_argument("--top_k_values", default="1,3,5,10,20", type=str)
    parser.add_argument("--rerank_type", type=str, default=None)
    parser.add_argument("--retriever_model", type=str)
    parser.add_argument("--diversity_threshold", type=float)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference generation.")
    parser.add_argument("--timestamp", type=str, required=False, help="Global timestamp from the bash script.")
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="A name for this run/experiment."
    )

    args = parser.parse_args()
    args.model_name = MODEL_TAG_TO_NAME.get(args.model_name, args.model_name)
    args.dataset_name = DATASET_TAG_TO_NAME.get(args.dataset_name, args.dataset_name)
    args.prompt_types = [PROMPT_TYPE_ALIASES.get(t.lower(), t.lower()) for t in args.prompt_types.split(",")]
    args.top_k_values = [int(val) for val in args.top_k_values.split(",")]
    args.rerank_type = None if args.rerank_type == "" else args.rerank_type

    # Ensure log directory exists
    os.makedirs("log", exist_ok=True)

    # Log run name and args to a global log file
    log_path = os.path.join("log", "inference_runs.log")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{datetime.now().isoformat()}] Run: {args.run_name}\n")
        for k, v in vars(args).items():
            log_file.write(f"    {k}: {v}\n")
        log_file.write("\n")

    logging.info("Launching inference script...")
    inference(
        **vars(args)
    )

if __name__ == "__main__":
    main()
