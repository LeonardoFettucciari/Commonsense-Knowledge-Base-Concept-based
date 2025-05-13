import os
from typing import List, Dict
import nltk
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from src.utils.io_utils import load_local_file, save_local_file
import random, collections
nltk.download("wordnet")


def build_ctx_noun(syn_name):
    s = wn.synset(syn_name)
    lex = s.lexname()                
    parts = [
        syn_name,
        "[LEX]", lex,
        "[LEMMA]", ", ".join(s.lemma_names()[:5]),
        "[DEF]", s.definition(),
    ]
    ex = s.examples()
    if ex: parts.extend(["[EX]", " ; ".join(ex[:3])])
    hyp = s.hypernyms()
    if hyp: parts.extend(["[HYPER]", ", ".join(h.name() for h in hyp[:2])])
    return " ".join(parts)

def sample_same_lex(target):
    lex = wn.synset(target).lexname()
    pool = [s for s in by_lex[lex] if s != target]
    return random.choice(pool) if pool else None

def sample_hypernym_sibling(target):
    hypers = wn.synset(target).hypernyms()
    if not hypers: return None
    sib_pool = []
    for h in hypers:
        sib_pool.extend([c.name() for c in h.hyponyms() if c.name()!=target])
    return random.choice(sib_pool) if sib_pool else None

def negatives_for_noun(target):
    # random noun
    yield random.choice([s for s in all_noun_syns if s!=target])
    # same lexname
    cand = sample_same_lex(target)
    if cand: yield cand
    # hypernym sibling
    sib = sample_hypernym_sibling(target)
    if sib: yield sib
    # same lemma different meaning
    synsets = wn.synsets(wn.synset(target).lemma_names()[0], pos='n')
    if len(synsets) > 1:
        for s in synsets:
            if s.name() != target:
                yield s.name()
                


input_path = f"data/ckb/cleaned/merged_filtered.jsonl"
output_path = f"outputs/classifier/full_gloss.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

ckb = load_local_file(input_path)
print(f"Loaded {len(ckb)} samples from {input_path}")

# Pre-compute lookup tables
all_noun_syns = [s.name() for s in wn.all_synsets('n')]
by_lex        = collections.defaultdict(list)
for s in tqdm(all_noun_syns, desc="Generating lexical lookup..."):        
    by_lex[wn.synset(s).lexname()].append(s)
ctx = {s: build_ctx_noun(s) for s in tqdm(all_noun_syns, desc="Generating ctx lookup...")}

# Generate dataset
dataset = []
for row in tqdm(ckb, desc="Generating pairs"):                    
    syn = row['synset_name']
    statements_added = 0
    
    for st in random.sample(row['statements'], k=min(7, len(row['statements']))):
        dataset.append({
            'statement': st,
            'synset': ctx[syn],
            'label': 1.0
        })
        for neg in negatives_for_noun(syn):
            dataset.append({
                'statement': st,
                'synset': ctx[neg],
                'label': 0.0
            })
        statements_added += 1
        if statements_added >= 7:
            break

save_local_file(
    dataset,
    output_path,
)