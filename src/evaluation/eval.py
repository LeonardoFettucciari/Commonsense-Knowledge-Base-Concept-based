import os
import datetime
import csv
import json
import argparse
import statistics
import logging
from natsort import natsorted
from xfinder.eval import Evaluator
from tqdm import tqdm
from huggingface_hub import snapshot_download
from src.utils.io_utils import write_accuracy_summary, file_already_processed, mark_file_as_processed
from src.utils.string_utils import extract_key_value_pairs, alias_filename, dict_to_filename, extract_value_from_key_in_file_name


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def xfinder_setup(model_name, model_dir):
    model_path = f"{model_dir}/IAAR-Shanghai/{model_name}"
    logging.info(f"Setting up xFinder model: {model_name} at {model_path}")
    
    # Ensure the model is fully downloaded or resume if incomplete
    snapshot_download(
        repo_id=f"IAAR-Shanghai/{model_name}",
        resume_download=True,
        local_dir=model_path
    )

    evaluator = Evaluator(
        model_name=model_name,
        inference_mode="local",
        model_path_or_url=model_path,
    )
    return evaluator

def evaluate(input_file, output_path, output_path_json, xfinder_evaluator_llama: Evaluator =None, xfinder_evaluator_qwen: Evaluator=None):
    logging.info(f"Evaluating file: {input_file}")
    with open(input_file) as csvfile, open(output_path, "w") as f_out, open(output_path_json, "w") as f_out_json:
        reader = csv.DictReader(csvfile, delimiter='\t')
        fieldnames = reader.fieldnames  # Dynamically get column names

        # Add xFinder output columns
        extended_fieldnames = fieldnames + [
            "xfinder_extracted_answer_llama", 
            "xfinder_acc_llama", 
            "xfinder_extracted_answer_qwen", 
            "xfinder_acc_qwen",

            "xfinder_extracted_answers_mismatch",
        ]
        
        writer = csv.DictWriter(f_out, fieldnames=extended_fieldnames, delimiter='\t')
        writer.writeheader()
        
        xfinder_accuracies_llama = []
        xfinder_accuracies_qwen = []
        xfinder_extracted_answers_mismatches = []
        
        for row in tqdm(reader, desc=f"Processing {input_file}"):
            try:
                question = row.get("question", None)
                choices = row.get("choices", None)
                model_output = row.get("model_output", None)
                ground_truth = row.get("ground_truth", None)

            except Exception as e:
                logging.info(f"Error processing row: {e}")
                continue  # Skip this row and move to the next one

            # Use xFinder formatting for the question and answer choices
            answer_range = [elem.split(". ") for elem in choices.split('\n')]
            
            formatted_choices = " ".join([f"({item[0]}) {item[1]}" for item in answer_range])
            formatted_question = f"{question} Answer Choices: {formatted_choices}"

            # Original output
            result_llama = xfinder_evaluator_llama.evaluate_single_item(
                question=formatted_question,
                llm_output=model_output,
                answer_range=answer_range,
                answer_type="alphabet_option",
                correct_answer=ground_truth,
            )
            
            result_qwen = xfinder_evaluator_qwen.evaluate_single_item(
                question=formatted_question,
                llm_output=model_output,
                answer_range=answer_range,
                answer_type="alphabet_option",
                correct_answer=ground_truth,
            )
            
            xfinder_accuracies_llama.append(result_llama[-1])
            xfinder_accuracies_qwen.append(result_qwen[-1])

            # Check if the extracted answers are different
            extracted_answer_llama = result_llama[-3]
            extracted_answer_qwen = result_qwen[-3]
            xfinder_extracted_answers_mismatch = 1 if extracted_answer_llama != extracted_answer_qwen else 0
            xfinder_extracted_answers_mismatches.append(xfinder_extracted_answers_mismatch)

            row.update({
                "xfinder_extracted_answer_llama": result_llama[-3],
                "xfinder_acc_llama": result_llama[-1],
                "xfinder_extracted_answer_qwen": result_qwen[-3],
                "xfinder_acc_qwen": result_qwen[-1],

                "xfinder_extracted_answers_mismatch": xfinder_extracted_answers_mismatch,
            })

            # Write row to output file
            row = {k: v for k, v in row.items() if k is not None}
            writer.writerow(row)
        
        # Write average accuracies
        avg_accuracy_llama = statistics.mean(xfinder_accuracies_llama)
        avg_accuracy_qwen = statistics.mean(xfinder_accuracies_qwen)
        logging.info(f"Average xFinder accuracy for LLaMA (original): {avg_accuracy_llama}")
        logging.info(f"Average xFinder accuracy for Qwen (original): {avg_accuracy_qwen}")

        # Compute mismateches
        total_mismatches = sum(xfinder_extracted_answers_mismatches)
        total_rows = len(xfinder_extracted_answers_mismatches)
        logging.info(f"Total mismatches: {total_mismatches} out of {total_rows} rows")

        # Write temporary JSON file that will be used later for summary
        json.dump({
            "prompt_type": extract_value_from_key_in_file_name(os.path.basename(input_file), "prompt"),

            "xfinder_avg_accuracy_llama": avg_accuracy_llama,
            "xfinder_avg_accuracy_qwen": avg_accuracy_qwen,
            "xfinder_average_accuracy": round((avg_accuracy_llama + avg_accuracy_qwen) / 2, 4),

            "xfinder_mismatches_percentage": round((total_mismatches / total_rows) * 100, 2),
            "xfinder_total_mismatches": total_mismatches,
            "xfinder_total_rows": total_rows,
        }, f_out_json)


import os
import datetime
import logging

def get_latest_datetime_dir(base_dir):
    """
    Scan direct subdirectories of base_dir, parse those whose names are
    in the format YYYY-MM-DD_HH-MM-SS, and return the path to the most recent one.
    If none found, return None.
    """
    dt_dirs = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            dt = datetime.datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            continue

        dt_dirs.append((dt, path))

    if not dt_dirs:
        return None

    latest_dt, latest_path = max(dt_dirs, key=lambda x: x[0])
    '''
    if len(dt_dirs) >= 2:
        sorted_dirs = sorted(dt_dirs, key=lambda x: x[0], reverse=True)
        second_dt, second_path = sorted_dirs[1]
    else:
        second_dt, second_path = None, None  # or raise an error
    '''
    return latest_path

def main(args):
    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1) Pick the latest timestamped folder if available
    base_dir = args.input_dir
    selected_dir = get_latest_datetime_dir(base_dir) or base_dir
    logging.info(f"Using input directory: {selected_dir}")


    # 2) Build list of files to process
    files_to_process = []
    for name in os.listdir(selected_dir):
        path = os.path.join(selected_dir, name)
        if not name.startswith(".") and os.path.isfile(path) and not file_already_processed(path):
            files_to_process.append(path)

    if not files_to_process:
        return

    # 3) Load models now that we have work
    logging.info("Loading models…")
    model_dir = "models"
    xfinder_evaluator_qwen = xfinder_setup("xFinder-qwen1505", model_dir)
    xfinder_evaluator_llama = xfinder_setup("xFinder-llama38it", model_dir)

    # 4) Prepare output
    output_dir = os.path.join(selected_dir, "accuracy")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving results to: {output_dir}")

    # 5) Process each queued file
    for file_path in files_to_process:
        name = os.path.basename(file_path)
        logging.info(f"Processing file: {name}")

        base, _ = os.path.splitext(name)
        metadata = extract_key_value_pairs(base)
        if metadata.get('prompt'):
            metadata['prompt'] = alias_filename(metadata['prompt'])
        clean_name = dict_to_filename(metadata)

        tsv_out = os.path.join(output_dir, f"{clean_name}.tsv")
        json_out = os.path.join(output_dir, f"{clean_name}.json")

        evaluate(
            file_path,
            tsv_out,
            json_out,
            xfinder_evaluator_llama,
            xfinder_evaluator_qwen,
        )

        mark_file_as_processed(file_path)
        logging.info(f"Finished processing: {name}")

    # 6) Final summary
    write_accuracy_summary(output_dir)
    logging.info("Evaluation process completed.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on MCQA with xFinder")
    parser.add_argument('--input_dir', required=True, help='Directory containing the output of the models on the MCQA task.')
    args = parser.parse_args()

    main(args)