from datasets import load_dataset
import csv
from nltk.corpus import wordnet
import os
import argparse


from utils.model_utils import get_ner_pipeline, load_gemini_model


# Initialize parser for reading api-key later
parser = argparse.ArgumentParser()
parser.add_argument("--api_key", type=str, required=True, help="API key for Gemini model.")
args = parser.parse_args()


# Load Data
eval_data = load_dataset("tau/commonsense_qa")['validation']
samples = eval_data.shuffle(seed=42).select(range(1))

# Bundle question + choices together
formatted_samples = []
for sample in samples:
    question = sample['question']
    choices = " ".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])])
    formatted_samples.append(f"{question} {choices}")


# Run NER pipeline
ner_pipeline = get_ner_pipeline("Babelscape/cner-base")
ner_results = ner_pipeline(formatted_samples)


# Extract concepts' words
synsets_all_samples = []
definitions_all_samples = []
unique_words_all_samples = []
for sample in ner_results:
    words = [concept["word"] for concept in sample]
    unique_words_sample = set(words)
    for word in words:
        split_words = word.split()
        unique_words_sample.update(split_words)
    unique_words = list(unique_words_sample)
    unique_words_all_samples.append(unique_words)

    # Extract concepts + definitions
    synsets_sample = [wordnet.synsets(word) for word in unique_words]
    definitions_sample = [[synset.definition() for synset in synset_list] for synset_list in synsets_sample]
    synsets_all_samples.append(synsets_sample)
    definitions_all_samples.append(definitions_sample)


# Set up model details
api_key = args.api_key # Extracted from input arguments
model_name = "models/gemini-1.5-flash"  
generation_config = {"temperature": 0.0, "max_output_tokens": 8192}
system_instruction = "You are an expert in commonsense reasoning. Given a concept and its definition, generate 10 useful commonsense statements about the concept."

# Load the model
llm = load_gemini_model(model_name, generation_config, system_instruction, api_key)

from tqdm import tqdm
all_outputs = []
for synsets_list, definitions_list in tqdm(zip(synsets_all_samples, definitions_all_samples)):
    sample_output = []
    for synset, definition in zip(synsets_list, definitions_list):
        for s, d in zip(synset, definition):

            prompt_message = {"role": "user", "parts": [f'Concept: {s.name().split(".")[0]}. Definition: {d}.']}
            

            # Start chat session
            chat_session = llm.start_chat(history=[])

            # Send message and get response
            model_output = chat_session.send_message(prompt_message).text

            # Current sample output
            sample_output.append(model_output)

    # All samples output
    all_outputs.append(sample_output)


# Extract statements' text only
raw_statements = [[statements.split("\n\n1")[1].split("\n",10)[:10] for statements in sample] for sample in all_outputs]
cleaned_statements = [[s.split(". ", 1)[1] for statements in sample for s in statements] for sample in raw_statements]

# Save output to file
relative_path = os.path.join("outputs", f"{model_name.split("/")[1]}.tsv")
with open(relative_path, mode="w", newline="", encoding="utf-8") as file:
    fieldnames = ['id',
                  'question',
                  'choices',

                  'statements',
                  'ner_results',
                  'unique_words',
                  'wordnet_synsets',
                  'wordnet_definitions',

                  'gold_truth',]
        
    tsv_writer = csv.writer(file, delimiter="\t")
    tsv_writer.writerow(fieldnames)

    for sample, statements, ner, synsets_list, definitions_list, unique_words in zip(
        samples,
        cleaned_statements,
        ner_results,
        synsets_all_samples,
        definitions_all_samples,
        unique_words_all_samples,
        ):

        row_list = [sample['id'],
                    sample['question'],
                    "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),

                    "\n".join(s for s in statements),
                    "\n".join(f"{n['word']}@{n['entity_group']}" for n in ner),
                    "\n".join(uw for uw in unique_words),
                    "\n".join(s.name() for synsets in synsets_list for s in synsets),
                    "\n".join(d for definitions in definitions_list for d in definitions),
                    
                    sample['answerKey'],]
        tsv_writer.writerow(row_list)
