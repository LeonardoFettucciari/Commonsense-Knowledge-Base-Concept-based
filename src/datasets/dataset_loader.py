import random
from torch.utils.data import Dataset
import os, csv, json
from datasets import load_dataset
from tqdm import tqdm
from src.utils.io_utils import load_local_file
from src.retriever.retriever import Retriever

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
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
                
        return samples