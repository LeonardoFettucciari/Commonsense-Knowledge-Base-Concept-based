import csv
import json
import re
import yaml


def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)
    
def load_json(path):
    with open(path, "r") as file:
        return json.load(file)
    
def load_local_file(file_path):
        if file_path.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_path.endswith(".jsonl"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        elif file_path.endswith(".csv"):
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=',')
                data = list(reader)
        elif file_path.endswith(".tsv"):
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                data = list(reader)
        else:
            raise ValueError(f"Unsupported file extension: {file_path}")
        return data


def save_output_to_file(relative_path,
                        samples,
                        all_outputs,
                        ner_results,
                        unique_words_all_samples,
                        ):

    with open(relative_path, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = [
            'id',
            'question',
            'choices',
            'ground_truth',
            'ner_results',
            'unique_words',
            'wordnet_synsets',
            'wordnet_definitions',
            'statements',
        ]
            
        tsv_writer = csv.writer(file, delimiter="\t")
        tsv_writer.writerow(fieldnames)

        for sample, outputs, ner, unique_words in zip(samples, all_outputs, ner_results, unique_words_all_samples):
            row_list = [
                sample['id'],
                sample['question'],
                "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
                sample['answerKey'],
                "\n".join(f"{n['word']}@{n['entity_group']}" for n in ner),
                "\n".join(uw for uw in unique_words),
                "\n".join(o["synset"] for o in outputs),
                "\n".join(o["definition"] for o in outputs),
                "\n".join("\n".join(o["statements"]) for o in outputs),
            ]
            tsv_writer.writerow(row_list)

def load_kb_statements(config_kb):
    kb_statements = []
    kb_path = config_kb['kb_path']
    kb_name = config_kb['kb_name']

    # Load kb from any file (json, csv, etc)
    kb = load_local_file(kb_path)

    # obqa_test_gemini-1.5-flash
    if(kb_name == "obqa_test_gemini-1.5-flash"):
        for line in kb:
            kb_statements.extend(line['statements'])
    # gemini-1.5-flash
    if(kb_name == "gemini-1.5-flash"):
        with open(kb_path, newline='', encoding='utf-8') as file:
            for line in kb:
                kb.extend([s for s in re.split(r'\n+', line['statements']) if len(s.strip()) > 0])
    # General case
    else:
        for line in kb:
            kb_statements.extend(line['statements'])
    
    return kb_statements

def csv_to_dict(path):
    with open(path, mode="r", newline="") as file:
        csv_reader = csv.DictReader(file)  
        return [row for row in csv_reader]

def prepare_output(sample, prompt, answer):
    output = {
        "id": sample["id"],
        "question": sample["question"],
        "choices": "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
        "gold_truth": sample['answerKey'],
        f"{prompt.name}": answer,
    }

    if(prompt.top_k):
        output['knowledge'] = "\n".join(sample['kb_statements'][:prompt.top_k])
        
    return output