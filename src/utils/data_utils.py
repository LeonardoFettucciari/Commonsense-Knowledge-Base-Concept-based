from nltk.corpus import wordnet
import os
import json
import csv
from datasets import load_dataset


def extract_unique_words(ner_results):
    unique_words_all_samples = []   
    for sample in ner_results:
        words = [concept["word"] for concept in sample]
        unique_words_sample = set(words)
        for word in words:
            split_words = word.split()
            unique_words_sample.update(split_words)
        unique_words_all_samples.append(list(unique_words_sample))

    return unique_words_all_samples

def wordnet_concepts_extraction(unique_words_all_samples):
    synsets_all_samples = []
    definitions_all_samples = []

    for unique_words in unique_words_all_samples:
        # Extract concepts + definitions
        synsets_sample = [wordnet.synsets(word) for word in unique_words]
        definitions_sample = [[synset.definition() for synset in synset_list] for synset_list in synsets_sample]
        synsets_all_samples.append(synsets_sample)
        definitions_all_samples.append(definitions_sample)

    return synsets_all_samples, definitions_all_samples

def get_all_wordnet_synsets(pos):
    return list(wordnet.all_synsets(pos=pos))

def from_words_to_synsets(list_of_words):
    flatten_list = [w for words in list_of_words for w in words]
    flatten_list = list(set(flatten_list))
    return [s for word in flatten_list for s in wordnet.synsets(word) if wordnet.synsets(word)]