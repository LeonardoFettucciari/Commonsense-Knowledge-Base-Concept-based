from sentence_transformers import CrossEncoder
import torch.nn as nn
import torch
import torch.nn.functional as F
from src.utils.model_utils import get_ner_pipeline
from src.utils.data_utils import extract_synsets, synsets_from_samples
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from nltk.corpus import wordnet as wn
from src.datasets.create_trainingdata_crossencoder import build_ctx_noun


# Load the trained DeBERTa model
model_path = "models/deberta-v3-classifier/checkpoint-20529"  # adjust if you use a checkpoint dir
tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
model = DebertaV2ForSequenceClassification.from_pretrained(model_path)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

ner_pipeline = get_ner_pipeline("Babelscape/cner-base")

sentences = [
    "Magnets are attracted to materials that contain iron or certain other metals.",
    "A belt buckle is often made of metal, such as steel or iron.",
    "Wood is not a magnetic material.",
    "Plastic is not a magnetic material.",
    "Paper does not contain metal and is not magnetic.",
    "A wooden table is typically made of organic material and lacks magnetic properties.",
    "A plastic cup is usually made of polymers that are not affected by magnets.",
    "A paper plate is made from compressed fibers and will not attract magnets.",
    "Magnets commonly stick to refrigerator doors, which are made of metal, similar to some belt buckles.",
    "If you try to stick a magnet to a belt buckle, it usually stays attached, indicating magnetic attraction."
]


# Synset extraction
synsets_per_sample = synsets_from_samples(sentences, ner_pipeline)

# Inference loop
scores = []
device = model.device

for sentence, synsets in zip(sentences, synsets_per_sample):
    input_pairs = [(syn.name(), build_ctx_noun(syn.name())) for syn in synsets]
    
    encodings = tokenizer(
        [s for s, _ in input_pairs],
        [ctx for _, ctx in input_pairs],
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**encodings).logits
        probs = torch.sigmoid(logits)[:, 1].cpu().tolist()  # Get class 1 prob

    scores.append((sentence, [(syn, score) for syn, score in zip(synsets, probs)]))

# Print results
for sentence, syn_score in scores:
    print("=" * 100)
    print(f"{sentence}")
    print("=" * 100)
    for syn, score in sorted(syn_score, key=lambda x: x[1], reverse=True):
        print(f"{syn} -> {score:.4f}")
        print(f"{syn.definition()}\n")
    print("\n")

