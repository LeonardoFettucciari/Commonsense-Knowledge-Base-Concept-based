from sentence_transformers import CrossEncoder
import torch.nn as nn
from src.utils.model_utils import get_ner_pipeline
from src.utils.data_utils import extract_synsets, synsets_from_samples
from nltk.corpus import wordnet as wn

def build_ctx_noun(syn):
    lex = syn.lexname()                
    parts = [
        syn.name(),
        "[LEX]", lex,
        "[LEMMA]", ", ".join(syn.lemma_names()[:5]),
        "[DEF]", syn.definition(),
    ]
    ex = syn.examples()
    if ex: parts.extend(["[EX]", " ; ".join(ex[:3])])
    hyp = syn.hypernyms()
    if hyp: parts.extend(["[HYPER]", ", ".join(h.name() for h in hyp[:2])])
    return " ".join(parts)


model = CrossEncoder(
   'models/classifier-ms-marco-MiniLM-L6-v2-full-gloss/final',
)
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


synsets_per_sample = synsets_from_samples(sentences, ner_pipeline)

scores = []
for sentence, synsets in zip(sentences, synsets_per_sample):
    sentence_synsets = [(sentence, build_ctx_noun(syn)) for syn in synsets]
    scores.append((sentence, [(syn, score) for syn, score in zip(synsets, model.predict(sentence_synsets, activation_fn=nn.Sigmoid()))]))

for sentence, syn_score in scores:
    print("=" * 100)
    print(f"{sentence}")
    print("=" * 100)
    for syn, score in sorted(syn_score, key=lambda x: x[1], reverse=True):
        print(f"{syn} -> {score}")
        print(f"{syn.definition()}\n")
    print("\n")

