import random
import os, csv, json
import re
from datasets import load_dataset, Dataset
from tqdm import tqdm
from src.utils.io_utils import load_local_file


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
    return dataset.map(preprocessor, remove_columns=dataset.column_names)

# Preprocessing decorators

PREPROCESSORS = {}

def register_preprocessor(name):
    def decorator(fn):
        PREPROCESSORS[name] = fn
        return fn
    return decorator

def get_preprocessor(name):
    if 'fewshot' in name:
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
        "choices": sample["choices"],
        "ground_truth": sample["answerKey"],
        "cot": sample["cot"]
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
        "cot": sample["cot"]
    }

@register_preprocessor("split_choices")
def preprocess_split_choices(sample):

    matches = re.findall(r'([A-Z])\.\s*(.+)', sample["choices"])

    sample["choices"] = {
        "text": [choice for _, choice in matches],
        "label": [label for label, _ in matches]
    }

    return sample






class QADataset(Dataset):
    def __init__(self, dataset_config):
        self.dc = dataset_config
        self.dataset_name = self.dc["dataset_name"]
        self.dataset_path = self.dc["dataset_path"]
        self.max_samples = self.dc.get("max_samples", None)
        
        self.dataset = self._load_dataset_from_path()
        self.samples = self._extract_samples_data()


    def _load_dataset_from_path(self):
        # Try to load from local file
        data = None
        if os.path.exists(self.dataset_path):
            try:
                data = load_local_file(self.dataset_path)
                if(self.dc['shuffle']):
                    random.shuffle(data)
                if(self.max_samples):
                    data = data[:self.max_samples]
            except Exception as e:
                raise ValueError(f"Failed to load local file: {e}")
        else:
            # Try to load from Hugging Face
            try:
                data = self._load_hf_dataset()
                if(self.dc['shuffle']):
                    data.shuffle()
                if(self.max_samples):
                    data = data.select(range(min(self.max_samples, len(data))))

            except Exception as e:
                raise ValueError(f"Failed to load Hugging Face dataset: {e}")
        
        return data

    def _load_hf_dataset(self):
        
        # Known datasets
        if self.dataset_name == "obqa":
            return load_dataset(self.dataset_path, self.dc['subset'], split=self.dc['split'])
        # Generic dataset
        else:
            try:
                return load_dataset(self.dataset_path, split=self.dc['split'])
            except Exception as e:
                raise ValueError(f"Failed to load Hugging Face dataset: {e}")
        
    def _extract_samples_data(self):
        # Extract questions, choices and answers from samples
        samples = []
        if(self.dataset_name == "obqa"):
            for sample in self.dataset:
                id = sample["id"]
                question = sample["question_stem"]
                choices_texts = sample["choices"]["text"]
                choices_labels = sample["choices"]["label"]
                answerKey = sample["answerKey"]

                samples.append({
                    "id": id,
                    "question": question,
                    "choices": {"text": choices_texts, "label": choices_labels},
                    "answerKey": answerKey,
                    })
        elif(self.dataset_name == "obqa_fewshot"):
            for sample in self.dataset:
                id = sample["id"]
                question = sample["question_stem"]
                choices = json.loads(sample["choices"])
                choices_texts = choices["text"]
                choices_labels = choices["label"]
                answerKey = sample["answerKey"]
                cot = sample["cot"]

                samples.append({
                    "id": id,
                    "question": question,
                    "choices": {"text": choices_texts, "label": choices_labels},
                    "answerKey": answerKey,
                    "cot": cot,
                    })
        elif(self.dataset_name == "csqa"):
            for sample in self.dataset:
                id = sample["id"]
                question = sample["question"]
                choices = sample["choices"]
                answerKey = sample["answerKey"]

                samples.append({
                    "id": id,
                    "question": question,
                    "choices": choices,
                    "answerKey": answerKey,
                    })
        elif(self.dataset_name == "csqa_fewshot"):
            for sample in self.dataset:
                id = sample["id"]
                question = sample["question"]
                choices = json.loads(sample["choices"])
                choices_texts = choices["text"]
                choices_labels = choices["label"]
                answerKey = sample["answerKey"]
                cot = sample["cot"]

                samples.append({
                    "id": id,
                    "question": question,
                    "choices": {"text": choices_texts, "label": choices_labels},
                    "answerKey": answerKey,
                    "cot": cot,
                    })
        elif(self.dataset_name == "qasc"):
            for sample in self.dataset:
                id = sample["id"]
                question = sample["question"]
                choices = sample["choices"]
                answerKey = sample["answerKey"]

                samples.append({
                    "id": id,
                    "question": question,
                    "choices": choices,
                    "answerKey": answerKey,
                    })
        elif(self.dataset_name == "qasc_fewshot"):
            for sample in self.dataset:
                id = sample["id"]
                question = sample["question"]
                choices = json.loads(sample["choices"])
                choices_texts = choices["text"]
                choices_labels = choices["label"]
                answerKey = sample["answerKey"]
                cot = sample["cot"]

                samples.append({
                    "id": id,
                    "question": question,
                    "choices": {"text": choices_texts, "label": choices_labels},
                    "answerKey": answerKey,
                    "cot": cot,
                    })
        else:
            try:
                for sample in self.dataset:
                    id = sample["id"]
                    question = sample["question"]
                    choices = sample["choices"]
                    answerKey = sample["ground_truth"]

                    lines = choices.split('\n')
                    choices_labels = []
                    choices_texts = []
                    for line in lines:
                        label, txt = line.split('. ', 1)
                        choices_labels.append(label.strip())
                        choices_texts.append(txt.strip())

                    samples.append({
                        "id": id,
                        "question": question,
                        "choices": {"text": choices_texts, "label": choices_labels},
                        "answerKey": answerKey,
                        })
            except Exception as e:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}") from e
                
        return samples