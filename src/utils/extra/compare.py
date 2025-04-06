import os
from src.utils.io_utils import load_local_file, save_local_file, load_yaml
from src.utils.string_utils import extract_key_value_pairs, key_value_pairs_to_filename


def compare(input_path_cot, input_path_with_knowledge):
    """
    Compare two TSV files line-by-line. Perform checks and optionally add columns 
    to new rows in an output TSV file.
    """
    input_cot = load_local_file(input_path_cot)
    input_with_knowledge = load_local_file(input_path_with_knowledge)

    output_data = []

    # Stats
    counter = 0
    good_changes = 0
    bad_changes = 0
    both_wrong_equal = 0
    both_wrong_different = 0
    both_correct = 0
    first_file_mismatch = 0
    second_file_mismatch = 0

    # We assume both files have the same length. 
    # If they don't, you might need to handle that (e.g., check lengths or iterate safely).
    for row_cot, row_with_knowledge in zip(input_cot, input_with_knowledge):
        # -------------------------------------------------------
        # 1. Perform the checks on the current lines as needed.
        #    For example, let's pretend we only want to keep rows 
        #    where the 'id' fields match and the 'some_value' is above 10.
        # -------------------------------------------------------
        
        # Count the number of rows processed
        counter += 1

        if row_cot["id"] != row_with_knowledge["id"]:
            raise ValueError(f"ID mismatch: {row_cot['id']} != {row_with_knowledge['id']}")
        
        if int(row_cot['xfinder_extracted_answers_mismatch']) == 1:
            first_file_mismatch += 1
            continue
        if int(row_with_knowledge['xfinder_extracted_answers_mismatch']) == 1:
            second_file_mismatch += 1
            continue

        if row_cot['xfinder_extracted_answer_llama'] == row_with_knowledge['xfinder_extracted_answer_llama']:
            if int(row_cot['xfinder_acc_llama']) == 0:
                both_wrong_equal += 1
            else:
                both_correct += 1
            continue

        if int(row_cot['xfinder_acc_llama']) == 0 and int(row_with_knowledge['xfinder_acc_llama']) == 0:
            both_wrong_different += 1
            continue

        good_change = 1 if int(row_cot['xfinder_acc_llama']) == 0 else 0
        bad_change = 1 if int(row_cot['xfinder_acc_llama']) == 1 else 0

        good_changes += good_change
        bad_changes += bad_change

        # -------------------------------------------------------
        # 2. Construct an output row. You can combine fields from
        #    both input rows, rename them, or add new columns.
        # -------------------------------------------------------
        new_row = {
            "id": row_cot["id"],
            "question": row_cot["question"],
            "choices": row_cot["choices"],
            "ground_truth": row_cot["ground_truth"],
            "cot_output": row_cot["model_output"],
            "knowledge_output": row_with_knowledge["model_output"],
            "ckb_statements": row_with_knowledge["ckb_statements"],
            "cot_answer": row_cot["xfinder_extracted_answer_llama"],
            "knowledge_answer": row_with_knowledge["xfinder_extracted_answer_llama"],
            
            "good_change": good_change,
        }

        # -------------------------------------------------------
        # 3. Append to output_data if you passed the checks.
        # -------------------------------------------------------
        output_data.append(new_row)

    # -------------------------------------------------------
    # 4. Append global stats.
    # -------------------------------------------------------

    good_changes_pct = round((good_changes / counter) * 100, 2) if counter > 0 else 0
    total_check = good_changes + bad_changes + both_wrong_equal + both_wrong_different + both_correct + first_file_mismatch + second_file_mismatch
    stats_data = [{
            "total_rows": counter,
            "good_changes_pct": good_changes_pct,
            "good_changes": good_changes,
            "bad_changes": bad_changes,
            "both_wrong_equal": both_wrong_equal,
            "both_wrong_different": both_wrong_different,
            "both_correct": both_correct,
            "first_file_mismatch": first_file_mismatch,
            "second_file_mismatch": second_file_mismatch,
            "total_check": total_check,
        }]

    # -------------------------------------------------------
    # 5. Save the collected rows into a new TSV file.
    # -------------------------------------------------------
    filename_no_knowledge = os.path.splitext(os.path.basename(input_path_cot))[0]
    filename_no_knowledge_metadata = extract_key_value_pairs(filename_no_knowledge)

    filename, extension = os.path.splitext(os.path.basename(input_path_with_knowledge))
    filename_metadata = extract_key_value_pairs(filename)
    prefix = filename_no_knowledge_metadata['prompt'] + "_vs_" + filename_metadata['prompt']
    output_dir = os.path.join(os.path.dirname(input_path_cot), "prompt_vs_knowledge", f"{filename_no_knowledge_metadata['prompt']}_vs_knowledge")
    os.makedirs(output_dir, exist_ok=True)
    filename_metadata.pop('prompt', None)
    output_filename = f"{prefix}|{key_value_pairs_to_filename(filename_metadata, extension)}"
    output_path = os.path.join(output_dir, output_filename)

    output_stats_filename = f"stats|{output_filename}"
    output_stats_path = os.path.join(output_dir, output_stats_filename)
    save_local_file(output_data, output_path)
    save_local_file(stats_data, output_stats_path)


best_knowledge = load_yaml("settings/best_knowledge_top_k.yaml")

datasets = ['csqa', 'obqa', 'qasc']
models = ['Llama-3.1-8B-Instruct', 'Llama-3.2-3B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-7B-Instruct']
retrieval_strategies = ['cner_filter', 'full_ckb']
prompt_types = ['zs', 'zscot', 'fs', 'fscot']

for dataset in datasets:
    for model in models:
        for retrieval_strategy in retrieval_strategies:
            for prompt_type in prompt_types:
                case = "zs" if "zs" in prompt_type else "fs"
                compare(f"outputs/inference/{dataset}/{model}/accuracy/model={model}|prompt={prompt_type}.tsv",
                        f"outputs/inference/{dataset}/{model}/accuracy/retrieval_strategy={retrieval_strategy}|model={model}|prompt={case}k{best_knowledge[dataset][model]['regular'][retrieval_strategy][case]}.tsv")

                compare(f"outputs/inference/{dataset}/{model}/accuracy/model={model}|prompt={prompt_type}.tsv",
                        f"outputs/inference_vera/{dataset}/{model}/accuracy/ckb=vera_final_ckb|retrieval_strategy={retrieval_strategy}|model={model}|prompt={case}k{best_knowledge[dataset][model]['vera'][retrieval_strategy][case]}.tsv")