from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Optional
import csv
import os
import torch
import random
import transformers
import tqdm
from datasets import load_dataset
from transformers import set_seed

from src.utils.io_utils import load_ckb_statements, prepare_output, load_yaml
from src.retriever.retriever import Retriever
from src.utils.model_utils import get_model_settings, generate_text, load_model_and_tokenizer
from src.datasets.dataset_loader import QADataset
from src.utils.random_utils import set_seed_forall
from src.utils.prompt_utils import build_prompts
from settings.aliases import PROMPT_TYPE_ALIASES, MODEL_NAME_TO_TAG, MODEL_TAG_TO_NAME, DATASET_NAME_TO_TAG, DATASET_TAG_TO_NAME


def inference(
    model_name: str,
    dataset_name: str,
    config_path: str,
    output_dir: str,
    ckb_path: str,
    prompt_types: List[str],
):
    # Load configuration file
    config = load_yaml(config_path)

    # Set seed for repeatable runs
    set_seed_forall(config['seed'])

    # Data loading
    eval_dataset = QADataset(config[f"{DATASET_NAME_TO_TAG[dataset_name]}"])
    fewshot_dataset = QADataset(config[f"{DATASET_NAME_TO_TAG[dataset_name]}_fewshot"])

    # Load knowledge base
    ckb_statements = load_ckb_statements(ckb_path)

    # Retrieve top_k statements for each query (i.e. question + choices)
    retriever = Retriever(ckb_statements, config["retriever"])
    eval_dataset.add_ckb_statements_to_samples(retriever, max(config["prompts"]["top_k_list"]))
    fewshot_dataset.add_ckb_statements_to_samples(retriever, max(config["prompts"]["top_k_list"]))

    # Main
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Iterate over the samples
    iterator = tqdm.tqdm(
        enumerate(eval_dataset.samples),
        total=len(eval_dataset.samples),
        desc=f"Running inference on {model_name}...",
    )
    
    prompt_types = [PROMPT_TYPE_ALIASES.get(t.lower(), t.lower()) for t in prompt_types]
    ground_truths = []
    answers = defaultdict(list)
    outputs = defaultdict(list)
    
    for i, sample in iterator:

        # Build prompts
        prompts = build_prompts(sample, prompt_types, config["prompts"], fewshot_examples=fewshot_dataset.samples)
        
        # Generate answers
        for prompt in prompts:
            answer = generate_text(model, tokenizer, prompt)
            answers[prompt.name].append(answer)

            # Prepare output information for sample
            outputs[prompt.name].append(prepare_output(sample, prompt, answer))

        # Append the ground truth
        ground_truths.append(sample["answerKey"])

    # Save model output
    model_output_path = os.path.join(output_dir, os.path.basename(ckb_path), DATASET_NAME_TO_TAG[dataset_name], model_name.split('/')[1])
    os.makedirs(model_output_path, exist_ok=True)
    for prompt_name, output in outputs.items():
        prompt_results_output_path = os.path.join(model_output_path, f"{prompt_name}.tsv")
        with open(prompt_results_output_path, mode="w", newline="", encoding="utf-8") as file:
            tsv_writer = csv.DictWriter(file, fieldnames=output[0].keys(), delimiter="\t")
            tsv_writer.writeheader()
            tsv_writer.writerows(output)



if __name__ == "__main__":
    parser = ArgumentParser(description="Creation of CKB with Gemini")
    parser.add_argument("--model_name", type=str, required=True, help="Model name from Hugging Face.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name from Hugging Face.")
    parser.add_argument("--ckb_path", type=str, required=True, help="Path to the Knowledge Base file.")
    parser.add_argument("--config_path", default="settings/config.yaml", type=str, required=False, help="Path to the config file.")
    parser.add_argument("--output_dir", default="outputs/inference/", type=str, required=False, help="Path to store the outputs.")
    parser.add_argument("--prompt_types", default=["all"], type=str, nargs='+', required=False, help="List of prompt types to use.")
    args = parser.parse_args()

    # Replace eventual aliases
    model_name = MODEL_TAG_TO_NAME.get(args.model_name, args.model_name)
    dataset_name = DATASET_TAG_TO_NAME.get(args.dataset_name, args.dataset_name)

    # Call main function
    inference(**vars(args))