from sentence_transformers import CrossEncoder
import torch
import os
import json
from tqdm import tqdm
import torch.nn.functional as F
from src.utils.model_utils import get_ner_pipeline
from src.utils.data_utils import extract_synsets, synsets_from_batch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from nltk.corpus import wordnet as wn
from src.classifier.create_trainingdata_classifier import build_ctx_noun
from src.utils.io_utils import load_local_file
from collections import defaultdict

# Paths
model_path = "models/deberta-v3-classifier_wup/final"
output_path = "data/ckb/raw/contextual_kb.jsonl"

# Load trained model
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
model = DebertaV2ForSequenceClassification.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ner_pipeline = get_ner_pipeline("Babelscape/cner-base")

# Load sentences
sentences = []
for dataset in ["csqa", "obqa", "qasc"]:
    data = load_local_file(f"outputs/batches/oracle/{dataset}.jsonl")
    for item in tqdm(data, desc=f"Loading {dataset} sentences"):
        sentences.extend(item["ckb_statements"])

# Synset extraction
synsets_per_sample = list(tqdm(
    synsets_from_batch(sentences, ner_pipeline),
    desc="Extracting synsets",
    total=len(sentences)
))

# Collect statements for best synset
synset_to_statements = defaultdict(list)

for sentence, synsets in tqdm(zip(sentences, synsets_per_sample),
                              total=len(sentences),
                              desc="Scoring sentences"):

    if not synsets:
        continue

    synset_ctx = [build_ctx_noun(syn.name()) for syn in synsets]
    sentence_batch = [sentence] * len(synsets)

    encodings = tokenizer(
        synset_ctx,
        sentence_batch,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**encodings).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()

    best_synset = synsets[probs.index(max(probs))]
    synset_to_statements[best_synset.name()].append(sentence)

# Write to JSONL with progress bar
with open(output_path, "w", encoding="utf-8") as f:
    for synset_name in tqdm(sorted(synset_to_statements.keys()), desc="Writing JSONL"):
        syn = wn.synset(synset_name)
        row = {
            "synset_name": syn.name(),
            "synset_lemma": syn.lemmas()[0].name(),
            "synset_definition": syn.definition(),
            "statements": synset_to_statements[synset_name]
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"✅ Done. Output saved to: {output_path}")
