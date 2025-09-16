import json
import os
import csv
from collections import Counter
from datetime import datetime

def percentage(part, total):
    if total == 0:
        return "0.00%"
    return f"{round(part / total * 100, 2):.2f}%"

# Inputs
kb_file = 'data/ckb/raw/merged_8files.jsonl'
datasets = ['obqa', 'csqa', 'qasc']
models = ['Llama-3.1-8B-Instruct', 'Llama-3.2-3B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-7B-Instruct']
run_name = "tr_fscotk5_it2_newkb"

# Load KB statements
kb_statements = set()
with open(kb_file, 'r') as f:
    for line in f:
        obj = json.loads(line)
        kb_statements.update(obj.get('statements', []))

kb_total = len(kb_statements)

# Iterate over dataset-model combinations
for dataset in datasets:
    for model in models:
        base_path = f'outputs/inference/{dataset}/{model}/{run_name}'

        # Find latest timestamp folder
        subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        if not subfolders:
            print(f"No folders found for {dataset}/{model}")
            continue
        latest_folder = max(subfolders, key=lambda x: datetime.strptime(x, "%Y-%m-%d_%H-%M-%S"))
        latest_path = os.path.join(base_path, latest_folder, "accuracy")

        # Find TSV file starting with "prompt"
        tsv_files = [f for f in os.listdir(latest_path) if f.startswith("prompt") and f.endswith(".tsv")]
        if not tsv_files:
            print(f"No prompt TSV file found for {dataset}/{model}")
            continue
        
        tsv_file = os.path.join(latest_path, tsv_files[0])

        # Counters
        total_counter = Counter()
        correct_counter = Counter()
        incorrect_counter = Counter()

        # Read TSV and process
        rows = []
        with open(tsv_file, 'r', encoding='utf-8') as fin:
            reader = csv.DictReader(fin, delimiter='\t')
            fieldnames = reader.fieldnames + ['contains_contextual_ckb', 'is_correct']
            for row in reader:
                raw_ckb_field = row['ckb_statements']

                # Safely parse statements list by splitting on literal \n
                ckb_statements_field = raw_ckb_field.replace('\\n', '\n').split('\n')
                if len(ckb_statements_field) != 5:
                    print(f"{row['id']} has {len(ckb_statements_field)} statements.")
                    print(ckb_statements_field)
                
                matched = sum(1 for stmt in ckb_statements_field if stmt in kb_statements)
                
                total_counter['samples_matched'] += 1 if matched > 0 else 0
                total_counter['samples_total'] += 1
                total_counter['matched'] += matched
                total_counter['total'] += len(ckb_statements_field)

                # check correctness condition
                mismatch = int(row.get('xfinder_extracted_answers_mismatch', 1))
                acc_llama = int(row.get('xfinder_acc_llama', 0))

                if mismatch == 0:
                    if acc_llama == 1:
                        correct_counter['samples_matched'] += 1 if matched > 0 else 0
                        correct_counter['samples_total'] += 1
                        correct_counter['matched'] += matched
                        correct_counter['total'] += len(ckb_statements_field)
                        row['is_correct'] = '✅'
                    elif acc_llama == 0:
                        incorrect_counter['samples_matched'] += 1 if matched > 0 else 0
                        incorrect_counter['samples_total'] += 1
                        incorrect_counter['matched'] += matched
                        incorrect_counter['total'] += len(ckb_statements_field)
                        row['is_correct'] = '❌'
                    else:
                        row['is_correct'] = '🤔'
                else:
                    row['is_correct'] = '🤔'

                highlighted_statements = []
                for stmt in ckb_statements_field:
                    if stmt in kb_statements:
                        highlighted_statements.append(f"**{stmt}**")
                    else:
                        highlighted_statements.append(stmt)
                row['ckb_statements'] = '\n\n'.join(highlighted_statements)

                row['contains_contextual_ckb'] = 1 if matched > 0 else 0
                rows.append(row)

        # Write modified TSV
        output_path = f'outputs/inference/{dataset}/{model}/{run_name}_contextual_counter'
        os.makedirs(output_path, exist_ok=True)
        modified_tsv = os.path.join(output_path, f"{run_name}_modified.tsv")
        with open(modified_tsv, 'w', encoding='utf-8', newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        # Prepare output2 json
        output = {
            'dataset': dataset,
            'model': model,
            'overall': {
                'samples_matched': total_counter['samples_matched'],
                'samples_total': total_counter['samples_total'],
                'samples_percentage': percentage(total_counter['samples_matched'],total_counter['samples_total']),
                'statements_matched': total_counter['matched'],
                'statements_total': total_counter['total'],
                'statements_percentage': percentage(total_counter['matched'],total_counter['total']),
            },
            'correct': {
                'samples_matched': correct_counter['samples_matched'],
                'samples_total': correct_counter['samples_total'],
                'samples_percentage': percentage(correct_counter['samples_matched'],correct_counter['samples_total']),
                'statements_matched': correct_counter['matched'],
                'statements_total': correct_counter['total'],
                'statements_percentage': percentage(correct_counter['matched'],correct_counter['total']),
            },
            'incorrect': {
                'samples_matched': incorrect_counter['samples_matched'],
                'samples_total': incorrect_counter['samples_total'],
                'samples_percentage': percentage(incorrect_counter['samples_matched'],incorrect_counter['samples_total']),
                'statements_matched': incorrect_counter['matched'],
                'statements_total': incorrect_counter['total'],
                'statements_percentage': percentage(incorrect_counter['matched'],incorrect_counter['total']),
            }
        }
        
        # Write output2 JSONL
        output2_file = os.path.join(output_path, f"{run_name}.jsonl")
        with open(output2_file, 'w') as fout:
            json.dump(output, fout, indent=2)

        # Write output2 as TSV
        output2_tsv = os.path.join(output_path, f"{run_name}.tsv")
        with open(output2_tsv, 'w', encoding='utf-8', newline='') as fout:
            tsv_writer = csv.writer(fout, delimiter='\t')
            header = [
                'dataset', 'model',
                'overall_samples_matched', 'overall_samples_total', 'overall_samples_percentage',
                'overall_statements_matched', 'overall_statements_total', 'overall_statements_percentage',
                'correct_samples_matched', 'correct_samples_total', 'correct_samples_percentage',
                'correct_statements_matched', 'correct_statements_total', 'correct_statements_percentage',
                'incorrect_samples_matched', 'incorrect_samples_total', 'incorrect_samples_percentage',
                'incorrect_statements_matched', 'incorrect_statements_total', 'incorrect_statements_percentage'
            ]
            tsv_writer.writerow(header)
            row_out = [
                output['dataset'], output['model'],
                output['overall']['samples_matched'], output['overall']['samples_total'], output['overall']['samples_percentage'],
                output['overall']['statements_matched'], output['overall']['statements_total'], output['overall']['statements_percentage'],
                output['correct']['samples_matched'], output['correct']['samples_total'], output['correct']['samples_percentage'],
                output['correct']['statements_matched'], output['correct']['statements_total'], output['correct']['statements_percentage'],
                output['incorrect']['samples_matched'], output['incorrect']['samples_total'], output['incorrect']['samples_percentage'],
                output['incorrect']['statements_matched'], output['incorrect']['statements_total'], output['incorrect']['statements_percentage']
            ]
            tsv_writer.writerow(row_out)

        print(f"Done for {dataset}/{model}. Outputs saved to {modified_tsv}, {output2_file}, and {output2_tsv}")
