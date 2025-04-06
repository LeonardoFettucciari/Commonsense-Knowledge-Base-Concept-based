import os
from src.utils.io_utils import load_local_file, save_local_file
from src.utils.string_utils import extract_key_value_pairs, key_value_pairs_to_filename

def compare(input_path_zs, input_path_with_knowledge):
    """
    Compare two TSV files line-by-line. Perform checks and optionally add columns 
    to new rows in an output TSV file.
    """
    input_zs = load_local_file(input_path_zs)
    input_with_knowledge = load_local_file(input_path_with_knowledge)

    output_data = []
    output_data_jsonl = []

    # Stats
    output_samples = 0
    positives_count = 0
    negatives_count = 0


    chunk_size = 20
    for i, row_zs in enumerate(input_zs):
        if row_zs['xfinder_extracted_answers_mismatch'] == 1:
                continue
        output_samples += 1

        start = i * chunk_size
        end = start + chunk_size
        chunk = input_with_knowledge[start:end]
        
        positives = []
        negatives = []
        for row_with_knowledge in chunk:
            # Compare row_zs with row_with_knowledge
            if row_zs["id"] != row_with_knowledge["id"]:
                raise ValueError(f"ID mismatch: {row_zs['id']} != {row_with_knowledge['id']}")
            
            if int(row_with_knowledge['xfinder_extracted_answers_mismatch']) == 1:
                continue
            
            if int(row_with_knowledge['xfinder_acc_llama']) == 1:
                positives.append(row_with_knowledge['ckb_statements'])
                positives_count += 1
            else:
                negatives.append(row_with_knowledge['ckb_statements'])
                negatives_count += 1

        positives_string = '\n'.join(positives)
        negatives_string = '\n'.join(negatives)
        new_row = {
            "id": row_zs["id"],
            "question": row_zs["question"],
            "choices": row_zs["choices"],
            "ground_truth": row_zs["ground_truth"],

            "positives": positives_string,
            "negatives": negatives_string,
        }
        output_data.append(new_row)


        choices = [choice.split('. ', 1) for choice in row_zs["choices"].split('\n')]
        choices = {
            'label': [c[0] for c in choices],
            'text': [c[1] for c in choices]
        }

        new_row_jsonl = {
            "id": row_zs["id"],
            "question": row_zs["question"],
            "choices": choices,
            "ground_truth": row_zs["ground_truth"],

            "positives": positives,
            "negatives": negatives,
        }
        output_data_jsonl.append(new_row_jsonl)


    stats_data = [{
            "samples_produced": output_samples,
            "positives_count": positives_count,
            "negatives_count": negatives_count,
        }]


    output_dir = os.path.dirname(input_path_with_knowledge)
    filename, extension = os.path.splitext(os.path.basename(input_path_with_knowledge))
    filename_metadata = extract_key_value_pairs(filename)
    prefix = "zs_vs_" + filename_metadata['prompt']
    filename_metadata.pop('prompt', None)
    output_filename = f"{prefix}|{key_value_pairs_to_filename(filename_metadata, extension)}"
    output_filename_jsonl = f"{prefix}|{key_value_pairs_to_filename(filename_metadata, 'jsonl')}"
    output_path = os.path.join(output_dir, output_filename)
    output_path_jsonl = os.path.join(output_dir, output_filename_jsonl)

    output_stats_filename = f"stats|{output_filename}"
    output_stats_path = os.path.join(output_dir, output_stats_filename)
    save_local_file(output_data, output_path)
    save_local_file(output_data_jsonl, output_path_jsonl)
    save_local_file(stats_data, output_stats_path)

datasets = ['csqa', 'obqa', 'qasc']
models = ['Llama-3.1-8B-Instruct', 'Llama-3.2-3B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-7B-Instruct']
retrieval_strategies = ['cner_filter', 'full_ckb']

k = {'csqa': {
        'Llama-3.1-8B-Instruct': {
            'full_ckb': 20,
            'cner_filter': 20
        },
        'Llama-3.2-3B-Instruct': {
            'full_ckb': 5,
            'cner_filter': 5
        },
        'Qwen2.5-1.5B-Instruct': {
            'full_ckb': 1,
            'cner_filter': 1
        },
        'Qwen2.5-7B-Instruct': {
            'full_ckb': 10,
            'cner_filter': 1
        }
    },

    'obqa': {
        'Llama-3.1-8B-Instruct': {
            'full_ckb': 10,
            'cner_filter': 10
        },
        'Llama-3.2-3B-Instruct': {
            'full_ckb': 3,
            'cner_filter': 5
        },
        'Qwen2.5-1.5B-Instruct': {
            'full_ckb': 3,
            'cner_filter': 5
        },
        'Qwen2.5-7B-Instruct': {
            'full_ckb': 3,
            'cner_filter': 3
        }
    },

    'qasc': {
        'Llama-3.1-8B-Instruct': {
            'full_ckb': 10,
            'cner_filter': 10
        },
        'Llama-3.2-3B-Instruct': {
            'full_ckb': 3,
            'cner_filter': 3
        },
        'Qwen2.5-1.5B-Instruct': {
            'full_ckb': 3,
            'cner_filter': 5
        },
        'Qwen2.5-7B-Instruct': {
            'full_ckb': 5,
            'cner_filter': 10
        }
    }
}

kvera = {'csqa': {
        'Llama-3.1-8B-Instruct': {
            'full_ckb': 20,
            'cner_filter': 20
        },
        'Llama-3.2-3B-Instruct': {
            'full_ckb': 5,
            'cner_filter': 5
        },
        'Qwen2.5-1.5B-Instruct': {
            'full_ckb': 1,
            'cner_filter': 1
        },
        'Qwen2.5-7B-Instruct': {
            'full_ckb': 20,
            'cner_filter': 20
        }
    },

    'obqa': {
        'Llama-3.1-8B-Instruct': {
            'full_ckb': 20,
            'cner_filter': 10
        },
        'Llama-3.2-3B-Instruct': {
            'full_ckb': 3,
            'cner_filter': 5
        },
        'Qwen2.5-1.5B-Instruct': {
            'full_ckb': 3,
            'cner_filter': 10
        },
        'Qwen2.5-7B-Instruct': {
            'full_ckb': 3,
            'cner_filter': 3
        }
    },

    'qasc': {
        'Llama-3.1-8B-Instruct': {
            'full_ckb': 20,
            'cner_filter': 5
        },
        'Llama-3.2-3B-Instruct': {
            'full_ckb': 3,
            'cner_filter': 3
        },
        'Qwen2.5-1.5B-Instruct': {
            'full_ckb': 3,
            'cner_filter': 5
        },
        'Qwen2.5-7B-Instruct': {
            'full_ckb': 1,
            'cner_filter': 10
        }
    }
}
'''
for dataset in datasets:
    for model in models:
        for retrieval_strategy in retrieval_strategies:

            compare(f"outputs/inference/{dataset}/{model}/accuracy/model={model}|prompt=zszs.tsv",
                    f"outputs/inference/{dataset}/{model}/accuracy/retrieval_strategy={retrieval_strategy}|model={model}|prompt=zsk{k[dataset][model][retrieval_strategy]}.tsv")

            compare(f"outputs/inference/{dataset}/{model}/accuracy/model={model}|prompt=zszs.tsv",
                    f"outputs/inference_vera/{dataset}/{model}/accuracy/ckb=vera_final_ckb|retrieval_strategy={retrieval_strategy}|model={model}|prompt=zsk{kvera[dataset][model][retrieval_strategy]}.tsv")
'''

compare('outputs/retriever/training_data/zeroshot/obqa/Llama-3.1-8B-Instruct/accuracy/model=Llama-3.1-8B-Instruct|prompt=zs|xfinder_extracted_answers_mismatch=0|xfinder_acc_llama=0.tsv',
        'outputs/retriever/training_data/obqa_retriever_train/Llama-3.1-8B-Instruct/accuracy/ckb=full_ckb|retrieval_strategy=cner_filter|model=Llama-3.1-8B-Instruct|prompt=zsk1.tsv')