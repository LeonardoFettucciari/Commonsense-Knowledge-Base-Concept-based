import random
import os, csv, json
import re
from datasets import load_dataset, Dataset
from tqdm import tqdm
from src.utils.io_utils import load_local_file

def split_choices(example):
    """
    Turns a flat `choices` string like
      "A. foo\nB. bar\nC. baz\nD. qux"
    into a dict:
      {"text": ["foo","bar","baz","qux"], "label": ["A","B","C","D"]}
    """
    matches = re.findall(r'([A-Z])\.\s*(.+)', example["choices"])
    return {
        "choices": {
            "text": [text for _, text in matches],
            "label": [lbl  for lbl, _  in matches]
        }
    }

def load_hf_dataset(config: dict) -> Dataset:
    """
    Loads a Hugging Face dataset using configuration settings.

    Args:
        config (dict): Configuration dictionary containing:
            - 'path' (str): Path or name of the dataset.
            - 'subset' (str, optional): Subset or configuration name.
            - 'split' (str, optional): Which split to load (e.g., 'train', 'test').
            - 'max_samples' (int, optional): Maximum number of samples to select.

    Returns:
        Dataset: A Hugging Face Dataset object with the selected split and
                 optionally limited to 'max_samples' items.
    """
    dataset = load_dataset(
        config.get('path'),
        config.get('subset'),
        split=config.get('split')
    )

    max_samples = config.get('max_samples') if config.get('max_samples') else len(dataset)

    max_samples = min(max_samples, len(dataset))
    dataset = dataset.select(range(max_samples))

    return dataset


def load_local_dataset(local_path: str, max_samples: int | None = None) -> Dataset:
    ext = os.path.splitext(local_path)[-1].lower()
    match ext:
        case '.json':
            dataset = load_dataset('json', data_files=local_path)['train']
        case '.jsonl':
            dataset = load_dataset('json', data_files=local_path)['train']
        case '.csv':
            dataset = load_dataset('csv', data_files=local_path)['train']
        case '.tsv':
            dataset = load_dataset('csv', data_files=local_path, delimiter='\t')['train']
        case _ :
            raise ValueError(f"Can't read dataset from {local_path}, extension not supported.")
        
    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return dataset

def preprocess_dataset(dataset: Dataset, dataset_name: str = "default") -> Dataset:
    preprocessor = get_preprocessor(dataset_name)
    if preprocessor is None:
        raise ValueError(f"No preprocessor registered for dataset: {dataset_name}")
    return dataset.map(preprocessor, remove_columns=dataset.column_names, load_from_cache_file=False)

# Preprocessing decorators

PREPROCESSORS = {}

def register_preprocessor(name):
    def decorator(fn):
        PREPROCESSORS[name] = fn
        return fn
    return decorator

def get_preprocessor(name):
    if 'fewshot' in name or 'fs' in name:
        return PREPROCESSORS.get(name, preprocess_default_fewshot)
    return PREPROCESSORS.get(name, preprocess_default)


# Preprocessing functions

def preprocess_default(sample):
    return {
        "id": sample["id"],
        "question": sample["question"],
        "choices": sample["choices"],
        "ground_truth": sample["answerKey"]
    }

def preprocess_default_fewshot(sample):
    return {
        "id": sample["id"],
        "question": sample["question"],
        "choices": json.loads(sample["choices"]),
        "ground_truth": sample["answerKey"],
        "cot": sample["cot"],
        "ckb_statements": sample.get("ckb_statements", "")
    }

@register_preprocessor("obqa")
def preprocess_obqa(sample):
    
    return {
        "id": sample["id"],
        "question": sample["question_stem"],
        "choices": sample["choices"],
        "ground_truth": sample["answerKey"],
    }

@register_preprocessor("obqa_fewshot")
def preprocess_obqa_fewshot(sample):
    
    return {
        "id": sample["id"],
        "question": sample["question_stem"],
        "choices": json.loads(sample["choices"]),
        "ground_truth": sample["answerKey"],
        "cot": sample["cot"],
        "ckb_statements": sample.get("ckb_statements", "")
    }

@register_preprocessor("obqa_fscotk")
def preprocess_obqa_fewshot(sample):
    
    return {
        "id": sample["id"],
        "question": sample["question_stem"],
        "choices": json.loads(sample["choices"]),
        "ground_truth": sample["answerKey"],
        "cot": sample["cot"],
        "ckb_statements": sample.get("ckb_statements", "")
    }

@register_preprocessor("split_choices")
def preprocess_split_choices(sample):

    matches = re.findall(r'([A-Z])\.\s*(.+)', sample["choices"])

    sample["choices"] = {
        "text": [choice for _, choice in matches],
        "label": [label for label, _ in matches]
    }

    return sample