import nltk
from nltk.corpus import wordnet as wn

def get_synset_info(synset_name: str):
    try:
        synset = wn.synset(synset_name)
    except nltk.corpus.reader.wordnet.WordNetError:
        return f"Error: Synset '{synset_name}' not found in WordNet."
    
    info = {
        "Name": synset.name(),
        "POS": synset.pos(),
        "Definition": synset.definition(),
        "Examples": synset.examples(),
        "Hypernyms": [hypernym.name() for hypernym in synset.hypernyms()],
        "Hyponyms": [hyponym.name() for hyponym in synset.hyponyms()],
        "Member Holonyms": [holonym.name() for holonym in synset.member_holonyms()],
        "Part Holonyms": [holonym.name() for holonym in synset.part_holonyms()],
        "Substance Holonyms": [holonym.name() for holonym in synset.substance_holonyms()],
        "Member Meronyms": [meronym.name() for meronym in synset.member_meronyms()],
        "Part Meronyms": [meronym.name() for meronym in synset.part_meronyms()],
        "Substance Meronyms": [meronym.name() for meronym in synset.substance_meronyms()],
        "Similar To": [similar.name() for similar in synset.similar_tos()],
        "Entailments": [entailment.name() for entailment in synset.entailments()],
    }
    
    return info

if __name__ == "__main__":
    nltk.download('wordnet')
    synset_name = input("Enter a WordNet synset name (e.g., 'electric_car.n.01'): ")
    info = get_synset_info(synset_name)
    print(info)