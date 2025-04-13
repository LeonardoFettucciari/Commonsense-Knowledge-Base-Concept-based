from nltk.corpus import wordnet
import os
import json
import csv
import logging
from src.utils.model_utils import get_ner_pipeline


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
    flatten_list = [word for word in list_of_words]
    return [synset for word in flatten_list for synset in wordnet.synsets(word) if wordnet.synsets(word) and synset.pos() == 'n']

def synsets_from_samples(samples):
    samples = [samples] if not isinstance(samples, list) else samples # Reads single string or list of strings

    # Run NER pipeline
    ner_pipeline = get_ner_pipeline("Babelscape/cner-base")
    ner_results = ner_pipeline(samples)

    # Extract unique words for each sample separately
    unique_words_per_sample = [extract_unique_words(ner_result) for ner_result in ner_results]

    # Convert words to synsets for each sample separately
    synsets = [from_words_to_synsets(unique_words) for unique_words in unique_words_per_sample]

    return synsets[0] if len(synsets) == 1 else synsets


def concatenate_question_choices(samples):
    samples = [samples] if not isinstance(samples, list) else samples # Reads single string or list of strings
    queries = []
    for s in samples:
        question = s["question"]
        choices = " ".join([f"{label}. {choice}" for label, choice in zip(s['choices']['label'], s['choices']['text'])])
        query = f"{question} {choices}" # Query is question + choices
        queries.append(query)
    return queries[0] if len(queries) == 1 else queries # Returns single string or list of strings

def filter_by_column_value(data: dict, column_name: str, target_value: str):
    filtered_data = [row for row in data if row.get(column_name) == target_value]
    logging.info(f"Filtered {len(data)} rows to {len(filtered_data)} rows where {column_name} = '{target_value}'")

    if not filtered_data:
        logging.info(f"No matching rows found for {column_name} = '{target_value}'")
        return data
    
    return filtered_data