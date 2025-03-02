from argparse import ArgumentParser
import gc
from typing import List, Optional
import csv
import os
import torch
import random
import transformers
import torch
import tqdm
from datasets import load_dataset
from transformers import set_seed
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils.stdout_utils import load_kb_statements, csv_to_dict
from src.retriever.retriever import Retriever
from src.utils.prompt_utils import prepare_prompt_input_data
from src.utils.model_utils import get_model_settings, generate_text
from src.utils.metrics_utils import compute_metrics
from src.prompts.llama_prompts import LlamaPrompt
from settings.constants import SEED, MODEL_LIST


# Set seed for repeatable runs
torch.manual_seed(SEED)
random.seed(SEED)
set_seed(SEED)

def run_inference(
    output_dir: str,
    kb_path: str,
    limit_samples: Optional[int] = None,
    top_k_list: Optional[List[int]] = None
):
    print("Current Working Directory:", os.getcwd())

    print("Authenticating with Hugging Face...")
    # Authentication for gated models e.g. LLama
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(hf_token)
    print("Authenticated with Hugging Face.")

    print("Setting device...")
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device set to:", device)

    # Data
    print("Loading data...")
    data = load_dataset("allenai/openbookqa")
    test_data = data['test']
    limit_samples = limit_samples or len(test_data)
    all_samples = test_data.shuffle(seed=SEED).select(range(limit_samples))
    all_examples = csv_to_dict(os.path.join("data", "fewshot_examples.csv"))
    print("Data loaded.")

    print("Loading Knowledge Base...")
    # KB
    kb_statements = load_kb_statements(kb_path, isJson=True)
    print("Knowledge Base loaded.")

    # Retrieve max_k statements for each question
    print("Retrieving statements for questions...")
    max_k = max(top_k_list)
    questions = [sample['question_stem'] for sample in all_samples]
    examples_questions = [example['question_stem'] for example in all_examples]
    retriever = Retriever()
    retriever.initialize(passages=kb_statements)
    all_samples_knowledge = retriever.retrieve(queries=questions, top_k=max_k)
    all_examples_knowledge = retriever.retrieve(queries=examples_questions, top_k=max_k)
    print("Statements retrieved.")

    # Main
    all_metrics_output = []
    for model_index, model_name in enumerate(MODEL_LIST, 1):
        print(f"Using model {model_index}/{len(MODEL_LIST)}: {model_name}")

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        model.model_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, padding_side="left"
            )
        

        # Inizialize lists for computing metrics later
        ground_truths = []
        answers = [] 
        
        # Iterate over the samples
        iterator = tqdm.tqdm(
            enumerate(zip(all_samples, all_samples_knowledge)),
            total=len(all_samples),
            desc="Running inference...",
        )
        
        all_samples_output = []
        for i, (sample, sample_knowledge) in iterator:

            # Build prompts
            prompts = []
            input_data = prepare_prompt_input_data(sample, sample_knowledge, all_examples, all_examples_knowledge, top_k_list)
            prompts.append(LlamaPrompt(input_data=input_data[0], zero_shot=True))
            prompts.append(LlamaPrompt(input_data=input_data[1], few_shot=True))
            prompts.append(LlamaPrompt(input_data=input_data[2], zero_shot=True, cot=True))            
            prompts.append(LlamaPrompt(input_data=input_data[3], few_shot=True, cot=True))
            for j in range(1, 2*len(top_k_list), 2):
                prompts.append(LlamaPrompt(input_data=input_data[3+j], zero_shot=True, knowledge=True))            
                prompts.append(LlamaPrompt(input_data=input_data[3+j+1], few_shot=True, knowledge=True))
            answers.extend([[] for _ in range(len(prompts))])
            
            # Generate answers
            for j, prompt in enumerate(prompts):
                answers[j].append(generate_text(model, tokenizer, prompt, device))

            # Append the ground truth for computing metrics later
            ground_truths.append(sample["answerKey"])

            # Prepare output information for sample
            o = {}
            o['id'] = sample["id"]
            o['question'] = sample["question_stem"]
            o['choices'] = "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])])
            o['gold_truth'] = sample['answerKey']
            o['knowledge'] = "\n".join(sample_knowledge)  
            all_samples_output.extend([[] for _ in range(len(prompts))])
            for j, prompt in enumerate(prompts):
                sample_output = o.copy()
                sample_output[prompt.name] = answers[j][i]
                all_samples_output[j].append(sample_output)  

        print("Freeing up resources...")
        # Free up resources
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    
        print("Resources freed.")
        

        # Save model output
        print("Writing model output...")
        model_output_path = os.path.join(f"{output_dir}",f"{model_name.split('/')[1]}", "obqa")
        os.makedirs(model_output_path, exist_ok=True)
        for sample_output, prompt in zip(all_samples_output, prompts):
            model_results_path = os.path.join(model_output_path,f"{prompt.name}.tsv")
            with open(model_results_path, mode="w", newline="", encoding="utf-8") as file:
                tsv_writer = csv.DictWriter(file, fieldnames=sample_output[0].keys(), delimiter="\t")
                tsv_writer.writeheader()
                tsv_writer.writerows(sample_output)
        print("Model output written.")

        print("Computing model metrics...")
        # Metrics
        metrics_output = {'model_name': model_name}
        metrics = compute_metrics(ground_truths, answers)
        for m, p in zip(metrics, prompts):
            metrics_output[f"accuracy_{p.name}"] = m
        all_metrics_output.append(metrics_output)
        print("Model metrics computed.")

    
    # Write metrics
    print("Writing metrics...")
    metrics_output_path = os.path.join(f"{output_dir}", "metrics.tsv")
    with open(metrics_output_path, mode="w", newline="", encoding="utf-8") as file:
        tsv_writer = csv.DictWriter(file, fieldnames=all_metrics_output[0].keys(), delimiter="\t")
        tsv_writer.writeheader()
        tsv_writer.writerows(all_metrics_output)
    print("Metrics written.")

if __name__ == "__main__":
    # Initialize parser for reading input api-key later
    parser = ArgumentParser(description="Creation of CKB with Gemini")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to store the Knowledge Base.")
    parser.add_argument("--kb_path", type=str, required=True, help="Path to the Knowledge Base file.")
    parser.add_argument("--limit_samples", type=int, required=False, help="Maximum number of samples to consider (default: all).")
    parser.add_argument("--top_k_list", type=int, nargs="+", required=False, default=[1, 3, 5, 10], help="List of values of k, statements to inject in prompts with knowledge. (default: 1 3 5 10).")
    args = parser.parse_args()
    run_inference(**vars(args))
