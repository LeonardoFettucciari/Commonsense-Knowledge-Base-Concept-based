from datasets import load_dataset
import csv
from nltk.corpus import wordnet
import os


from utils.model_utils import get_ner_pipeline, load_gemini_model

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


# Extract concepts' words
for sample in ner_results:
    words = [concept["word"] for concept in sample]
    unique_words = set(words)
    for word in words:
        split_words = word.split()
        unique_words.update(split_words)
    unique_words_list = list(unique_words)

    # Extract concepts + definitions
    synsets = [wordnet.synsets(word) for word in unique_words_list]
    definitions = [[synset.definition() for synset in synset_list] for synset_list in synsets]


# Set up model details
api_key = "your_api_key_here"  
model_name = "gemini-pro"  
generation_config = {"temperature": 0.0, "max_output_tokens": 8192}
system_instruction = "You are an expert in commonsense reasoning. Given a concept and its definition, generate 10 useful commonsense statements about the concept."

# Load the model
llm = load_gemini_model(model_name, generation_config, system_instruction, api_key)

from tqdm import tqdm
all_outputs = []
for synset, definition in tqdm(zip(synsets, definitions)):
    for s, d in zip(synset, definition):

        prompt_message = [
            {"role": "user", "content": f'Concept: {s.name().split(".")[0]}. Definition: {d}.'},
        ]

        # Start chat session
        chat_session = llm.start_chat(history=[])

        # Send message and get response
        model_output = chat_session.send_message(prompt_message).text

        # Append to output list
        all_outputs.append(model_output)


# Extract statements' text only
raw_statements = [statements.split("\n\n1")[1].split("\n",10)[:10] for statements in all_outputs]
cleaned_statements = [[statement.split(". ", 1)[1] for statement in statements] for statements in raw_statements]

# Save output to file
relative_path = os.path.join("outputs", f"{model_name}.tsv")
with open(relative_path, mode="w", newline="", encoding="utf-8") as file:
    fieldnames = ['id', 'question', 'choices', 'gold_truth', 'statements',]
        
    tsv_writer = csv.writer(file, delimiter="\t")
    tsv_writer.writerow(fieldnames)
    
    for sample in samples:
        row_list = [sample['id'],
                    sample['question'],
                    "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
                    sample['answerKey'],
                    "\n".join(s for statements in cleaned_statements for s in statements),]
        tsv_writer.writerow(row_list)
