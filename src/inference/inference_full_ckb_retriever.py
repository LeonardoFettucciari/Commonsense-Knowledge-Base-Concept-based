import logging
from argparse import ArgumentParser
from collections import defaultdict
from typing import List
import csv
import os
import tqdm

from src.utils.io_utils import load_ckb_statements, prepare_output, load_yaml
from src.retriever.retriever import Retriever
from src.utils.model_utils import generate_text, load_model_and_tokenizer
from src.utils.data_utils import concatenate_question_choices
from src.datasets.dataset_loader import QADataset
from src.utils.prompt_utils import build_prompts, get_prompt_settings
from settings.aliases import PROMPT_TYPE_ALIASES, MODEL_TAG_TO_NAME, DATASET_NAME_TO_TAG, DATASET_TAG_TO_NAME
from src.utils.string_utils import prepare_prompt_output_path, prepare_model_output_path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def inference(
    model_name: str,
    dataset_name: str,
    config_path: str,
    output_dir: str,
    ckb_path: str,
    prompt_types: List[str],
    top_k_values: List[str],
):

    logging.info("Starting inference process...")
    logging.info(f"Loading configuration from: {config_path}")
    config = load_yaml(config_path)

    # Analyze prompt types to load only required data
    prompt_types = [PROMPT_TYPE_ALIASES.get(t.lower(), t.lower()) for t in prompt_types]
    # Check if fewshot, cot or knowledge required and set flags
    prompt_settings = get_prompt_settings(prompt_types)


    # Load eval data
    logging.info(f"Loading dataset: {dataset_name}")
    eval_dataset = QADataset(config[f"{DATASET_NAME_TO_TAG[dataset_name]}"])
    logging.info(f"Loaded {len(eval_dataset.samples)} samples for evaluation.")

    # Fewshot data
    fewshot_dataset = None
    if prompt_settings["fewshot"]:
        logging.info(f"Loading fewshot examples: {config[DATASET_NAME_TO_TAG[dataset_name] + '_fewshot']}")
        fewshot_dataset = QADataset(config[f"{DATASET_NAME_TO_TAG[dataset_name]}_fewshot"])
        logging.info(f"Loaded {len(fewshot_dataset.samples)} fewshot examples.")
    

    # Knowledge data
    ckb_statements = None
    if prompt_settings["knowledge"]:
        logging.info(f"Loading knowledge base from: {ckb_path}")
        ckb_statements = load_ckb_statements(ckb_path)
        logging.info(f"Loaded {len(ckb_statements)} knowledge base statements.")

        logging.info("Initializing retriever...")
        retriever = Retriever(ckb_statements, config["retriever"])

        logging.info(f"Retrieveing {max(top_k_values)} statements for evaluation dataset...")
        eval_formatted_questions = concatenate_question_choices(eval_dataset.samples)
        eval_ckb_statements = retriever.retrieve(eval_formatted_questions, max(top_k_values))
        retriever.add_ckb_statements_to_samples(eval_dataset.samples, eval_ckb_statements)

        # Knowledge and fewshot data
        if prompt_settings["fewshot"]:
            logging.info(f"Retrieveing {max(top_k_values)} statements for fewshot examples...")
            fewshot_formatted_questions = concatenate_question_choices(fewshot_dataset.samples)
            fewshot_ckb_statements = retriever.retrieve(fewshot_formatted_questions, max(top_k_values))
            retriever.add_ckb_statements_to_samples(fewshot_dataset.samples, fewshot_ckb_statements)


    # Load model and tokenizer
    logging.info(f"Loading model and tokenizer: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)
    logging.info("Model and tokenizer loaded successfully.")

    # Iterate over samples for inference
    logging.info("Starting inference...")
    iterator = tqdm.tqdm(
        enumerate(eval_dataset.samples),
        total=len(eval_dataset.samples),
        desc=f"Running inference on {model_name}...",
    )
    
    ground_truths = []
    answers = defaultdict(list)
    outputs = defaultdict(list)
    
    for i, sample in iterator:
        logging.debug(f"Processing sample {i+1}/{len(eval_dataset.samples)}")

        # Build prompts
        prompts = build_prompts(
            sample, 
            prompt_types, 
            top_k_values, 
            fewshot_examples=fewshot_dataset.samples if fewshot_dataset is not None else []
        )
        
        # Generate answers
        for prompt in prompts:
            answer = generate_text(model, tokenizer, prompt)
            answers[prompt.name].append(answer)
            outputs[prompt.name].append(prepare_output(sample, prompt, answer))

        # Append ground truth
        ground_truths.append(sample["answerKey"])
    
    # Save model output
    model_output_path = prepare_model_output_path(output_dir, DATASET_NAME_TO_TAG[dataset_name], model_name)
    os.makedirs(model_output_path, exist_ok=True)
    logging.info(f"Saving inference results to: {model_output_path}")
    
    for prompt_name, output in outputs.items():
        prompt_results_output_path = prepare_prompt_output_path(model_output_path, os.path.basename(ckb_path), prompt_name, model_name, retriever_type="full_ckb")
        with open(prompt_results_output_path, mode="w", newline="", encoding="utf-8") as file:
            tsv_writer = csv.DictWriter(file, fieldnames=output[0].keys(), delimiter="\t")
            tsv_writer.writeheader()
            tsv_writer.writerows(output)
        logging.info(f"Saved results for prompt type '{prompt_name}' to {prompt_results_output_path}")

    logging.info("Inference process completed successfully.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Creation of CKB with Gemini")
    parser.add_argument("--model_name", type=str, required=True, help="Model name from Hugging Face.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name from Hugging Face.")
    parser.add_argument("--ckb_path", type=str, required=True, help="Path to the Knowledge Base file.")
    parser.add_argument("--config_path", default="settings/config.yaml", type=str, required=False, help="Path to the config file.")
    parser.add_argument("--output_dir", default="outputs/inference/retriever_on_full_ckb/", type=str, required=False, help="Path to store the outputs.")
    parser.add_argument("--prompt_types", default="all", type=str, required=False, help="Comma-separated list of prompt types to use.")
    parser.add_argument("--top_k_values", default="1,3,5,10,20", type=str, required=False, help="Comma-separated list of prompt types to use.")

    args = parser.parse_args()

    # Replace eventual aliases
    model_name = MODEL_TAG_TO_NAME.get(args.model_name, args.model_name)
    dataset_name = DATASET_TAG_TO_NAME.get(args.dataset_name, args.dataset_name)

    # Convert prompt types into list
    args.prompt_types = args.prompt_types.split(",")
    # Convert top_k_values into list
    args.top_k_values = [int(val) for val in args.top_k_values.split(",")]

    logging.info("Launching inference script...")
    inference(**vars(args))
