import torch
from tqdm import tqdm
from src.utils.model_utils import get_ner_pipeline
from src.utils.data_utils import synsets_from_batch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from nltk.corpus import wordnet as wn
from src.classifier.create_trainingdata_classifier import build_ctx_noun
from src.utils.io_utils import load_local_file

model_path = "models/deberta-v3-classifier_wup/final" 
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
model = DebertaV2ForSequenceClassification.from_pretrained(model_path)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")
ner_pipeline = get_ner_pipeline("Babelscape/cner-base")

# Load oracle sentences
sentences = []
for dataset in ["csqa", "obqa", "qasc"]:
    data = load_local_file(f"outputs/batches/oracle/{dataset}.jsonl")
    for item in tqdm(data, total=len(data), desc=f"Loading {dataset} sentences"):
        sentences.extend(item["ckb_statements"])

sentences = sentences[1500:1510]


# Synset extraction
synsets_per_sample = list(tqdm(
    synsets_from_batch(sentences, ner_pipeline),
    desc="Extracting synsets",
    total=len(sentences)
))


# Inference loop
scores = []
device = model.device
for sentence, synsets in tqdm(zip(sentences, synsets_per_sample),
                              total=len(sentences),
                              desc="Processing sentences"):

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

    scores.append((sentence,
                   [(syn, p) for syn, p in zip(synsets, probs)]))


# Print results
for sentence, syn_score in scores:
    print("=" * 100)
    print(f"{sentence}")
    print("=" * 100)
    for syn, score in sorted(syn_score, key=lambda x: x[1], reverse=True):
        print(f"{syn} -> {score:.4f}")
        print(f"{syn.definition()}\n")
    print("\n")

