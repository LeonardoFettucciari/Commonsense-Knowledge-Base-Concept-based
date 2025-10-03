import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

# Load necessary corpora
nltk.download('wordnet')
nltk.download('wordnet_ic')
brown_ic = wordnet_ic.ic('ic-brown.dat')

def semantically_similar_synsets_from_lemma(lemma_str, threshold=0.0):
    synsets = wn.synsets(lemma_str)
    if not synsets:
        return []

    base_syn = synsets[0]
    similar = []

    for cand in synsets:
        if cand.pos() != base_syn.pos():
            continue  # similarity only valid between same part of speech

        scores = {
            "wup": base_syn.wup_similarity(cand),
            "path": base_syn.path_similarity(cand),
            "lch": None,
            "res": None,
            "jcn": None,
            "lin": None
        }

        # Only try LCH if both synsets are in the same tree
        try:
            scores["lch"] = base_syn.lch_similarity(cand)
        except:
            pass

        # IC-based similarities
        try:
            scores["res"] = base_syn.res_similarity(cand, brown_ic)
        except:
            pass
        try:
            scores["jcn"] = base_syn.jcn_similarity(cand, brown_ic)
        except:
            pass
        try:
            scores["lin"] = base_syn.lin_similarity(cand, brown_ic)
        except:
            pass

        if any(v is not None and v >= threshold for v in scores.values()):
            similar.append((cand, scores))

    return similar


if __name__ == "__main__":
    while(True):
        lemma = input("Enter a lemma (e.g., 'bank'): ")
        if not lemma.strip():
            print("Exiting...")
            break
        info = semantically_similar_synsets_from_lemma(lemma)

        for syn, scores in info:
            print(f"\n{syn.name()} – {syn.definition()}")
            for name, score in scores.items():
                print(f"{name}_similarity: {score}")
