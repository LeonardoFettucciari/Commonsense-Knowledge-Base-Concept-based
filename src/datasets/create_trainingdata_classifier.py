import os
import random
import collections
from typing import List, Dict

import nltk
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from src.utils.io_utils import load_local_file, save_local_file

nltk.download("wordnet", quiet=True)

WUP_THRESHOLD = 0.8  # Threshold for Wu-Palmer similarity

def build_ctx_noun(syn_name):
    s = wn.synset(syn_name)
    parts = [
        syn_name,
        s.lemma_names()[0],
        s.definition(),
    ]
    return "\n".join(parts)


def get_negative_synsets(original_synset_name, all_noun_syns, pos_count=6):
    sldm_count = pos_count // 2

    # Parse the original synset
    original_synset = wn.synset(original_synset_name)
    lemma = original_synset.lemma_names()[0]

    # Same Lemma Different Meaning (SLDM)
    candidate_synsets = wn.synsets(lemma, pos=original_synset.pos())
    sldm_synsets = [s.name() for s in candidate_synsets if s.name() != original_synset.name()][:sldm_count]

    # Random synsets
    random_count = pos_count - len(sldm_synsets)
    remaining_synsets = [s for s in all_noun_syns if s != original_synset.name()]
    random_synsets = random.sample(remaining_synsets, k=random_count)

    return sldm_synsets + random_synsets


def main():
    input_path = f"data/ckb/cleaned/merged_filtered.jsonl"
    output_path = f"outputs/classifier/trainset_wup_as_positive_label.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ckb = load_local_file(input_path)
    print(f"Loaded {len(ckb)} samples from {input_path}")

    # Pre-compute lookup tables
    all_noun_syns = [s.name() for s in wn.all_synsets('n')]
    ctx = {s: build_ctx_noun(s) for s in tqdm(all_noun_syns, desc="Generating ctx lookup...")}
    syn2statements = {s['synset_name']: s['statements'] for s in tqdm(ckb, desc="Generating ctx lookup...")}

    # Generate dataset
    dataset = []
    pos_count = 6
    for row in tqdm(ckb, desc="Generating triples..."):  
        syn = row['synset_name']

        # Add negative samples
        for neg in get_negative_synsets(syn, all_noun_syns, pos_count=pos_count):
            try:
                wup_sim = wn.synset(syn).wup_similarity(wn.synset(neg))
            except Exception:
                wup_sim = 0.0

            dataset.append({
                'synset': ctx[syn],
                'statement': random.choice(syn2statements[neg]),
                'label': wup_sim if wup_sim and wup_sim > WUP_THRESHOLD else 0.0, 
            })

        # Add positive samples
        for st in random.sample(row['statements'], k=min(pos_count, len(row['statements']))):
            dataset.append({
                'synset': ctx[syn],
                'statement': st,
                'label': 1.0
            })

    save_local_file(dataset, output_path)
    print(f"Saved {len(dataset)} examples to {output_path}")


if __name__ == "__main__":
    main()
