import csv
import json
import re

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

def load_kb_statements(input_path, isJson=False):
    kb = []
    if isJson:
        with open(input_path, "r", encoding="utf-8") as file:
            for line in file:
                kb.extend(json.loads(line)['statements'])

    else:
        with open(input_path, newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                kb.extend([s for s in re.split(r'\n+', row['statements']) if len(s.strip()) > 0])
    return kb

def csv_to_dict(path):
    with open(path, mode="r", newline="") as file:
        csv_reader = csv.DictReader(file)  
        return [row for row in csv_reader]
