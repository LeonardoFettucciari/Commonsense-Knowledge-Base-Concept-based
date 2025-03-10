import os
import random
import jsonlines
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Optional
from datasets import load_dataset
from transformers import set_seed
from src.utils.model_utils import get_ner_pipeline, load_gemini_model, get_model_settings, clean_statement
from src.utils.data_utils import extract_unique_words, get_all_wordnet_synsets, from_words_to_synsets
from src.ckb_creation.gemini_prompts import GeminiPrompt
ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"
POS_LIST = [NOUN, VERB, ADJ, ADV]

torch.manual_seed(42)
random.seed(42)
set_seed(42)

def create_ckb(
    config_path: str,
    api_key: str,
    output_dir: str,
    limit_samples: Optional[int] = None
):
    
    # eval_data = load_dataset("allenai/openbookqa")['test']
    # samples = eval_data.shuffle(seed=42) # .select(range(10))

    # # Bundle question + choices together
    # formatted_samples = []
    # for sample in samples:
    #     question = sample['question_stem']
    #     choices = " ".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])])
    #     formatted_samples.append(f"{question} {choices}")

    # # Run NER pipeline
    # ner_pipeline = get_ner_pipeline("Babelscape/cner-base")
    # ner_results = ner_pipeline(formatted_samples)

    # # Extract unique words
    # unique_words_all_samples = extract_unique_words(ner_results)
    # wordnet_synsets = from_words_to_synsets(unique_words_all_samples)

    # Get model settings
    settings = get_model_settings(config_path)
    model_name = settings["model_name"]
    generation_config = settings["generation_config"]
    system_instruction = settings["system_instruction"]
    
    llm = load_gemini_model(
        model_name, 
        generation_config, 
        system_instruction,
        api_key
    )

    wordnet_synsets = get_all_wordnet_synsets(pos=NOUN)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(
        output_dir, 
        f"ckb|data=wordnet|model={model_name}.jsonl"
    )

    with jsonlines.open(output_path, "w") as fout:
        for i, synset in tqdm(enumerate(wordnet_synsets), total=len(wordnet_synsets)):
            if i == limit_samples:
                break
            synset_name = synset.name()
            synset_definition = synset.definition()
            synset_lemma = synset.lemmas()[0].name()
            input_data = {
                "synset_lemma": synset_lemma,
                "synset_definition": synset_definition
            }
            prompt = GeminiPrompt(input_data=input_data)
            # Start chat session
            chat_session = llm.start_chat(history=[])
            # Send message and get response
            model_output = chat_session.send_message(prompt.messages[0]).text
            # Extract statements' text only and clean them i.e. remove '5.' out of '5. <Statement number 5>'
            cleaned_statements = clean_statement(model_output)
            # Current sample output
            output_dict = {
                "synset_name": synset_name,
                "synset_lemma": synset_lemma,
                "synset_definition": synset_definition,
                "statements": cleaned_statements
            }
            fout.write(output_dict)

if __name__ == "__main__":
    # Initialize parser for reading input api-key later
    parser = ArgumentParser(description="Creation of CKB with Gemini")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the Gemini config.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for Gemini model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to store the Knowledge Base.")
    parser.add_argument("--limit_samples", type=int, required=False, help="Maximum number of synsets to consider.")
    args = parser.parse_args()
    create_ckb(**vars(args))