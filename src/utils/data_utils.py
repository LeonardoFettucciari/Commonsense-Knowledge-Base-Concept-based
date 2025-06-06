from nltk.corpus import wordnet
import os
import json
import csv
import logging
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from datasets import Dataset
import nltk
from typing import Dict, List
from src.utils.model_utils import get_ner_pipeline
from tqdm import tqdm


def extract_unique_words(ner_results):
    words = [concept["word"] for concept in ner_results]
    unique_words_sample = set(words)
    for word in words:
        split_words = word.split()
        unique_words_sample.update(split_words)
    return list(unique_words_sample)

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
    return [synset for word in list_of_words for synset in wordnet.synsets(word) if wordnet.synsets(word) and synset.pos() == 'n']


def synsets_from_batch(samples, ner_pipeline, batch_size=32):
             # generator = iterator
    batch_synsets = []

    for spans in ner_pipeline(samples, batch_size=batch_size):
        batch_synsets.append(
            from_words_to_synsets(extract_unique_words(spans))
        )
    return batch_synsets

# For progress bar
'''
def synsets_from_batch(samples, ner_pipeline, batch_size=32):
    sample_iter = (s for s in samples)          # generator = iterator
    batch_synsets = []

    for spans in tqdm(
            ner_pipeline(sample_iter, batch_size=batch_size),
            total=len(samples),
            desc="Extracting synsets"):
        batch_synsets.append(
            from_words_to_synsets(extract_unique_words(spans))
        )
    return batch_synsets
'''

def extract_synsets(samples, ner_pipeline):
    if isinstance(samples, str):
        samples = [samples]

    # Run NER pipeline
    ner_results = ner_pipeline(samples)

    # Extract unique words for each sample separately
    unique_words_per_sample = [extract_unique_words(sentence) for sentence in ner_results]
    print(unique_words_per_sample)

    synsets_per_sample = []
    for words, sentence in zip(unique_words_per_sample, samples):
        synsets = []
        for word in words:
            syn = lesk(word_tokenize(sentence), word)
            if syn:
                synsets.append(syn)
        synsets_per_sample.append(synsets)

    return synsets_per_sample[0] if len(synsets_per_sample) == 1 else synsets_per_sample


def concatenate_question_choices(samples):
    samples = [samples] if not isinstance(samples, list) else samples # Reads single string or list of strings
    queries = []
    for s in samples:
        question = s["question"]
        choices = "\n".join([f"{label}. {choice}" for label, choice in zip(s['choices']['label'], s['choices']['text'])])
        query = f"{question}\n{choices}" # Query is question + choices
        queries.append(query)
    return queries[0] if len(queries) == 1 else queries # Returns single string or list of strings

def filter_by_column_value(data: dict, column_name: str, target_value: str):
    filtered_data = [row for row in data if row.get(column_name) == target_value]
    logging.info(f"Filtered {len(data)} rows to {len(filtered_data)} rows where {column_name} = '{target_value}'")

    if not filtered_data:
        logging.info(f"No matching rows found for {column_name} = '{target_value}'")
        return data
    
    return filtered_data