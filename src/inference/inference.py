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

from src.utils.stdout_utils import read_kb_statements
from src.retriever.retriever import retriever
from src.utils.prompt_utils import prepare_prompt
from src.utils.model_utils import get_answers
from src.utils.metrics_utils import compute_metrics
from settings.constants import SEED, NUM_SAMPLES, TOP_K, MODEL_LIST, NUM_EXAMPLES

# Set seed for repeatable runs
torch.manual_seed(SEED)
random.seed(SEED)
set_seed(SEED)

# Authentication for gated models e.g. LLama
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Train data
train_data = load_dataset("tau/commonsense_qa")['train']
examples = train_data.shuffle(seed=SEED).select(range(NUM_EXAMPLES))
examples_questions = [example['question'] for example in examples]

# Validation data
eval_data = load_dataset("tau/commonsense_qa")['validation']
samples = eval_data.shuffle(seed=SEED).select(range(NUM_SAMPLES))
questions = [sample['question'] for sample in samples]

kb_path = os.path.join("outputs", "gemini-1.5-flash.tsv")
kb_statements = read_kb_statements(kb_path)

# Retrieve top_k statements for each question
all_samples_knowledge = retriever(questions, kb_statements, TOP_K)
all_examples_knowledge = retriever(examples_questions, kb_statements, TOP_K)


# Main
metrics_output = []
for model_index, model_name in enumerate(MODEL_LIST, 1):
    print(f"Using model #{model_index}: {model_name}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left"
        )
    
    # Inizialize lists for computing metrics later
    ground_truths = []
    answers_zeroshot = []
    answers_zeroshot_with_knowledge_k1 = []
    answers_zeroshot_with_knowledge_k3 = []
    answers_zeroshot_with_knowledge_k5 = []
    answers_zeroshot_with_knowledge_k10 = []

    answers_fewshot = []
    answers_fewshot_with_knowledge_k1 = []
    answers_fewshot_with_knowledge_k3 = []
    answers_fewshot_with_knowledge_k5 = []
    answers_fewshot_with_knowledge_k10 = []
    

    # Iterate over the samples
    iterator = tqdm.tqdm(
        enumerate(zip(samples, all_samples_knowledge)),
        total=len(samples),
        desc="Generating Answers...",
    )
    
    full_output = []
    for i, (sample, sample_knowledge) in iterator:

        # Build prompts
        prompt_zeroshot =                   prepare_prompt(sample,
                                                        zero_shot=True)
        
        prompt_zeroshot_with_knowledge_k1 = prepare_prompt(sample,
                                                        knowledge=sample_knowledge,
                                                        zero_shot=True, with_knowledge=True, top_k=1)
        
        prompt_zeroshot_with_knowledge_k3 = prepare_prompt(sample,
                                                        knowledge=sample_knowledge,
                                                        zero_shot=True, with_knowledge=True, top_k=3)
        
        prompt_zeroshot_with_knowledge_k5 = prepare_prompt(sample,
                                                        knowledge=sample_knowledge,
                                                        zero_shot=True, with_knowledge=True, top_k=5)
        
        prompt_zeroshot_with_knowledge_k10 = prepare_prompt(sample,
                                                        knowledge=sample_knowledge,
                                                        zero_shot=True, with_knowledge=True, top_k=10)

        prompt_fewshot =                    prepare_prompt(sample,
                                                        examples,
                                                        few_shot=True)
        
        prompt_fewshot_with_knowledge_k1 =  prepare_prompt(sample,
                                                        examples,
                                                        knowledge=sample_knowledge,
                                                        examples_knowledge=all_examples_knowledge,
                                                        few_shot=True, with_knowledge=True, top_k=1)
        
        prompt_fewshot_with_knowledge_k3 =  prepare_prompt(sample,
                                                        examples,
                                                        knowledge=sample_knowledge,
                                                        examples_knowledge=all_examples_knowledge,
                                                        few_shot=True, with_knowledge=True, top_k=3)
        
        prompt_fewshot_with_knowledge_k5 =  prepare_prompt(sample,
                                                        examples,
                                                        knowledge=sample_knowledge,
                                                        examples_knowledge=all_examples_knowledge,
                                                        few_shot=True, with_knowledge=True, top_k=5)
        
        prompt_fewshot_with_knowledge_k10 =  prepare_prompt(sample,
                                                        examples,
                                                        knowledge=sample_knowledge,
                                                        examples_knowledge=all_examples_knowledge,
                                                        few_shot=True, with_knowledge=True, top_k=10)

        # Generate answers
        _, answer_zeroshot =                    get_answers(model, tokenizer, prompt_zeroshot, model_name, max_new_tokens=1, device=device)
        _, answer_zeroshot_with_knowledge_k1 =  get_answers(model, tokenizer, prompt_zeroshot_with_knowledge_k1, model_name, max_new_tokens=1, device=device)
        _, answer_zeroshot_with_knowledge_k3 =  get_answers(model, tokenizer, prompt_zeroshot_with_knowledge_k3, model_name, max_new_tokens=1, device=device)
        _, answer_zeroshot_with_knowledge_k5 =  get_answers(model, tokenizer, prompt_zeroshot_with_knowledge_k5, model_name, max_new_tokens=1, device=device)
        _, answer_zeroshot_with_knowledge_k10 = get_answers(model, tokenizer, prompt_zeroshot_with_knowledge_k10, model_name, max_new_tokens=1, device=device)

        _, answer_fewshot =                     get_answers(model, tokenizer, prompt_fewshot, model_name, max_new_tokens=1, device=device)
        _, answer_fewshot_with_knowledge_k1 =   get_answers(model, tokenizer, prompt_fewshot_with_knowledge_k1, model_name, max_new_tokens=1, device=device)
        _, answer_fewshot_with_knowledge_k3 =   get_answers(model, tokenizer, prompt_fewshot_with_knowledge_k3, model_name, max_new_tokens=1, device=device)
        _, answer_fewshot_with_knowledge_k5 =   get_answers(model, tokenizer, prompt_fewshot_with_knowledge_k5, model_name, max_new_tokens=1, device=device)
        _, answer_fewshot_with_knowledge_k10 =  get_answers(model, tokenizer, prompt_fewshot_with_knowledge_k10, model_name, max_new_tokens=1, device=device)


        # Append answers for computing metrics later
        answers_zeroshot.append(answer_zeroshot)
        answers_zeroshot_with_knowledge_k1.append(answer_zeroshot_with_knowledge_k1)
        answers_zeroshot_with_knowledge_k3.append(answer_zeroshot_with_knowledge_k3)
        answers_zeroshot_with_knowledge_k5.append(answer_zeroshot_with_knowledge_k5)
        answers_zeroshot_with_knowledge_k10.append(answer_zeroshot_with_knowledge_k10)

        answers_fewshot.append(answer_fewshot)
        answers_fewshot_with_knowledge_k1.append(answer_fewshot_with_knowledge_k1)
        answers_fewshot_with_knowledge_k3.append(answer_fewshot_with_knowledge_k3)
        answers_fewshot_with_knowledge_k5.append(answer_fewshot_with_knowledge_k5)
        answers_fewshot_with_knowledge_k10.append(answer_fewshot_with_knowledge_k10)

        # Append the ground truth for computing metrics later
        ground_truths.append(sample["answerKey"])


        # Get information about the sample
        sample_output = {}
        sample_output['id'] = sample["id"]
        sample_output['question'] = sample["question"]
        sample_output['choices'] = "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])])
        sample_output['gold_truth'] = sample['answerKey']
        sample_output['knowledge'] = "\n".join(sample_knowledge)  

        sample_output['answer_zeroshot'] = answer_zeroshot
        sample_output['answer_zeroshot_with_knowledge_k1'] = answer_zeroshot_with_knowledge_k1
        sample_output['answer_zeroshot_with_knowledge_k3'] = answer_zeroshot_with_knowledge_k3
        sample_output['answer_zeroshot_with_knowledge_k5'] = answer_zeroshot_with_knowledge_k5
        sample_output['answer_zeroshot_with_knowledge_k10'] = answer_zeroshot_with_knowledge_k10

        sample_output['answer_fewshot'] = answer_fewshot
        sample_output['answer_fewshot_with_knowledge_k1'] = answer_fewshot_with_knowledge_k1
        sample_output['answer_fewshot_with_knowledge_k3'] = answer_fewshot_with_knowledge_k3
        sample_output['answer_fewshot_with_knowledge_k5'] = answer_fewshot_with_knowledge_k5
        sample_output['answer_fewshot_with_knowledge_k10'] = answer_fewshot_with_knowledge_k10

        full_output.append(sample_output)        

    # Save output
    model_results_path = os.path.join("outputs",f"{model_name.split('/')[1]}.tsv")

    with open(model_results_path, mode="w", newline="", encoding="utf-8") as file:
        # Re-arrange output columns order as preferred
        fieldnames = ['id', 'question', 'choices',
                      'knowledge',
                      'gold_truth',
                      
                      'answer_zeroshot', 'answer_zeroshot_with_knowledge_k1',
                      'answer_zeroshot_with_knowledge_k3', 'answer_zeroshot_with_knowledge_k5',
                      'answer_zeroshot_with_knowledge_k10',
                      'answer_fewshot', 'answer_fewshot_with_knowledge_k1',
                      'answer_fewshot_with_knowledge_k3', 'answer_fewshot_with_knowledge_k5',
                      'answer_fewshot_with_knowledge_k10',]
        
        tsv_writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        tsv_writer.writeheader()
        tsv_writer.writerows(full_output)

    # Metrics
    metrics = compute_metrics(
        ground_truths,

        answers_zeroshot,
        answers_zeroshot_with_knowledge_k1,
        answers_zeroshot_with_knowledge_k3,
        answers_zeroshot_with_knowledge_k5,
        answers_zeroshot_with_knowledge_k10,

        answers_fewshot,
        answers_fewshot_with_knowledge_k1,
        answers_fewshot_with_knowledge_k3,
        answers_fewshot_with_knowledge_k5,
        answers_fewshot_with_knowledge_k10,
        )
    
    # Free up resources
    del model
    torch.cuda.empty_cache()
    
    model_metrics = {
        'model_name': model_name,
        'accuracy_zeroshot': metrics['accuracy_zeroshot'],
        'accuracy_zeroshot_with_knowledge_k1': metrics['accuracy_zeroshot_with_knowledge_k1'],
        'accuracy_zeroshot_with_knowledge_k3': metrics['accuracy_zeroshot_with_knowledge_k3'],
        'accuracy_zeroshot_with_knowledge_k5': metrics['accuracy_zeroshot_with_knowledge_k5'],
        'accuracy_zeroshot_with_knowledge_k10': metrics['accuracy_zeroshot_with_knowledge_k10'],

        'accuracy_fewshot': metrics['accuracy_fewshot'],
        'accuracy_fewshot_with_knowledge_k1': metrics['accuracy_fewshot_with_knowledge_k1'],
        'accuracy_fewshot_with_knowledge_k3': metrics['accuracy_fewshot_with_knowledge_k3'],
        'accuracy_fewshot_with_knowledge_k5': metrics['accuracy_fewshot_with_knowledge_k5'],
        'accuracy_fewshot_with_knowledge_k10': metrics['accuracy_fewshot_with_knowledge_k10'],
    }
    metrics_output.append(model_metrics)
    
# Write metrics
metrics_output_path = os.path.join("outputs", "metrics.tsv")
with open(metrics_output_path, mode="w", newline="", encoding="utf-8") as file:
    tsv_writer = csv.DictWriter(file, fieldnames=metrics_output[0].keys(), delimiter="\t")
    tsv_writer.writeheader()
    tsv_writer.writerows(metrics_output)