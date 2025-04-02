import csv
import json
import re
from natsort import natsorted
import yaml
import os
import logging

def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)
    
def load_json(path):
    with open(path, "r") as file:
        return json.load(file)
    
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

    
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


def save_local_file(data, file_path):
    if file_path.endswith(".json"):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    elif file_path.endswith(".jsonl"):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    elif file_path.endswith(".csv"):
        if not data:
            raise ValueError("No data to write to CSV.")
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    elif file_path.endswith(".tsv"):
        if not data:
            raise ValueError("No data to write to TSV.")
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys(), delimiter='\t')
            writer.writeheader()
            writer.writerows(data)

    else:
        raise ValueError(f"Unsupported file extension: {file_path}")



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

def load_ckb(ckb_path: str, retrieval_scope: str):
    """
    Load the knowledge base from `ckb_path`.
    Depending on retrieval_scope, we might load a dict or a list, etc.
    """
    if retrieval_scope == "cner_filter":
        logging.info("Loading CKB as a dictionary for synset-based retrieval.")
        ckb_dict = load_kb_as_dict(ckb_path)
        return ckb_dict

    elif retrieval_scope == "full_ckb":
        logging.info("Loading CKB as a list of statements.")
        ckb_list = load_ckb_statements(ckb_path)
        return ckb_list

    else:
        logging.warning(f"Unknown retrieval_scope '{retrieval_scope}'. Defaulting to empty.")
        return []


def load_ckb_statements(ckb_path):
    # Load ckb from path
    ckb = load_local_file(ckb_path)
    # Load statements from unknown generic ckb file
    ckb_statements = []
    for line in ckb:
        ckb_statements.extend(line['statements'])
    return ckb_statements

def load_kb_as_dict(jsonl_path):
    kb_dict = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            kb_dict[data["synset_name"]] = data["statements"] # Given a synset, return its statements
    return kb_dict

def prepare_output(sample, prompt, answer):
    output = {
        "id": sample["id"],
        "question": sample["question"],
        "choices": "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
        "ground_truth": sample['answerKey'],
        "model_output": answer,
    }
    if(prompt.top_k):
        output['ckb_statements'] = "\n".join(sample['ckb_statements'][:prompt.top_k])
    return output

def prepare_output_retriever_training(sample, prompt, answer, top_k_index):
    output = {
        "id": sample["id"],
        "top_k_index": top_k_index,
        "question": sample["question"],
        "choices": "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
        "ground_truth": sample['answerKey'],
        "model_output": answer,
        "ckb_statements": prompt.ckb_statements,
    }
    return output

def prepare_output_refine(sample, prompt, original_answer, refine_answer):
    return {
        "id": sample["id"],
        "question": sample["question"],
        "choices": "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
        "ground_truth": sample['answerKey'],
        "ckb_statements": "\n".join(sample['ckb_statements'][:prompt.top_k]),
        "model_output": original_answer,
        "model_output_refine": refine_answer,
    }

def csv_to_dict(path):
    with open(path, mode="r", newline="") as file:
        csv_reader = csv.DictReader(file)  
        return [row for row in csv_reader]

def jsonl_to_tsv(jsonl_file, tsv_file=None):

    tsv_file = tsv_file if tsv_file else os.path.splitext(jsonl_file)[0] + ".tsv"
    
    with open(jsonl_file, "r", encoding="utf-8") as infile, open(tsv_file, "w", encoding="utf-8", newline="") as outfile:
        # Read the first line to get field names (keys)
        first_line = infile.readline().strip()
        if not first_line:
            print("Empty JSONL file.")
            return
        
        first_record = json.loads(first_line)
        fieldnames = list(first_record.keys())  # Get keys from first record

        # Write header and first line
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(first_record)

        # Process remaining lines
        for line in infile:
            writer.writerow(json.loads(line.strip()))

def write_accuracy_summary(input_dir):
    bundle_json_by_prefix(input_dir)
    all_jsonl_to_tsv(input_dir)

def bundle_json_by_prefix(input_dir):
    files = natsorted(os.listdir(input_dir))
    prefix_dict = {}

    # Group files by prefix
    for file in files:
        if file.endswith(".json"):
            prefix_part = file.split("prompt=")[0] + "prompt="
            if prefix_part not in prefix_dict:
                prefix_dict[prefix_part] = []
            prefix_dict[prefix_part].append(file)

    # Process each prefix group
    for prefix, grouped_files in prefix_dict.items():
        output_name = "xFinder_accuracy|" + prefix.replace("|prompt=", ".jsonl")
        output_file = os.path.join(input_dir, output_name)

        if os.path.exists(output_file):
            os.remove(output_file)

        for file in grouped_files:
            input_file = os.path.join(input_dir, file)
            if os.path.isfile(input_file):
                with open(input_file, "r", encoding="utf-8") as infile:
                    data = json.load(infile)
                    with open(output_file, "a", encoding="utf-8") as outfile:
                        outfile.write(json.dumps(data) + "\n")
                os.remove(input_file)

def all_jsonl_to_tsv(input_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".jsonl"):  
            file_path = os.path.join(input_dir, file)  
            jsonl_to_tsv(file_path) 

def save_jsonl(data, file_path):
    """Save a list of dictionaries into a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
