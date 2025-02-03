from datasets import load_dataset
import torch
import os
import argparse
from transformers import set_seed
import random
from utils.model_utils import get_ner_pipeline, load_gemini_model, get_answers
from utils.data_utils import extract_unique_words, wordnet_concepts_extraction
from utils.stdout_utils import save_output_to_file
from settings.prompts import SYSTEM_PROMPT

torch.manual_seed(42)
random.seed(42)
set_seed(42)

# Initialize parser for reading input api-key later
parser = argparse.ArgumentParser()
parser.add_argument("--api_key", type=str, required=True, help="API key for Gemini model.")
args = parser.parse_args()

# Load Data
eval_data = load_dataset("tau/commonsense_qa")['validation']
samples = eval_data.shuffle(seed=42).select(range(100))

# Bundle question + choices together
formatted_samples = []
for sample in samples:
    question = sample['question']
    choices = " ".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])])
    formatted_samples.append(f"{question} {choices}")


# Run NER pipeline
ner_pipeline = get_ner_pipeline("Babelscape/cner-base")
ner_results = ner_pipeline(formatted_samples)


# Extract unique words
unique_words_all_samples = extract_unique_words(ner_results)
# Extract Wordnet synsets and definitions
synsets_all_samples, definitions_all_samples = wordnet_concepts_extraction(unique_words_all_samples)

# Set up model details
api_key = args.api_key # Extracted from input arguments
model_name = "models/gemini-1.5-flash"  
generation_config = {"temperature": 0.0, "max_output_tokens": 8192}
system_instruction = SYSTEM_PROMPT

# Save output to file
if not os.path.exists("outputs"):
    os.makedirs("outputs")

output_path = os.path.join("outputs", f"{model_name.split("/")[1]}.tsv")

# Load the model
llm = load_gemini_model(model_name, generation_config, system_instruction, api_key)

# Get answers
all_outputs = get_answers(llm, synsets_all_samples, definitions_all_samples)

save_output_to_file(output_path,
                    samples,
                    all_outputs,
                    ner_results,
                    unique_words_all_samples)