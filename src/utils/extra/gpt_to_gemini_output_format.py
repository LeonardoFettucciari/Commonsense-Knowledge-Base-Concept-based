import json
import os
from tqdm import tqdm
from nltk.corpus import wordnet as wn

from src.utils.ckb_utils import clean_statements

# Ensure WordNet is available (you can run this once in your environment)
# import nltk
# nltk.download('wordnet')


def format(input_path, output_path):
    model_name = 'gpt' if 'gpt' in os.path.basename(input_path) else 'gemini'

    # Load all synsets in WordNet
    all_synsets = list(wn.all_synsets(pos='n'))

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for idx, line in tqdm(enumerate(infile), total=len(all_synsets), desc="Processing"):
            if idx >= len(all_synsets):
                print(f"Warning: More input rows than synsets. Stopping at {idx}.")
                break

            synset = all_synsets[idx]
            synset_metadata = {
                "synset_name": synset.name(),
                "synset_lemma": synset.lemmas()[0].name(),
                "synset_definition": synset.definition()
            }

            response_content = None
            data = json.loads(line)
            if model_name == 'gpt':
                response_content = data["response"]["body"]["choices"][0]["message"]["content"]
            elif model_name == 'gemini':
                response_content = data['statements']

            statements = clean_statements(response_content)            

            output_obj = {
                **synset_metadata,
                "statements": statements
            }

            outfile.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

format('data/ckb/raw/*ckb_data=wordnet|model=gpt-4o-mini.jsonl',
       'data/ckb/raw/ckb_data=wordnet|model=gpt-4o-mini.jsonl')

format('data/ckb/raw/*ckb_data=wordnet_model=gemini-1.5-flash.jsonl',
       'data/ckb/raw/ckb_data=wordnet_model=gemini-1.5-flash.jsonl')
