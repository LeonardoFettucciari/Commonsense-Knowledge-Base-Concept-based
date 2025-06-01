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


def evaluate(input_file, output_path, output_path_json, xfinder_evaluator_llama: Evaluator = None, xfinder_evaluator_qwen: Evaluator = None):
    logging.info(f"Evaluating file: {input_file}")
    with open(input_file) as csvfile, open(output_path, "w") as f_out, open(output_path_json, "w") as f_out_json:
        reader = csv.DictReader(csvfile, delimiter='\t')
        fieldnames = reader.fieldnames

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
                continue

            answer_range = [elem.split(". ") for elem in choices.split('\n')]
            formatted_choices = " ".join([f"({item[0]}) {item[1]}" for item in answer_range])
            formatted_question = f"{question} Answer Choices: {formatted_choices}"

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

            row = {k: v for k, v in row.items() if k is not None}
            writer.writerow(row)
        
        avg_accuracy_llama = statistics.mean(xfinder_accuracies_llama)
        avg_accuracy_qwen = statistics.mean(xfinder_accuracies_qwen)
        total_mismatches = sum(xfinder_extracted_answers_mismatches)
        total_rows = len(xfinder_extracted_answers_mismatches)

        logging.info(f"Average xFinder accuracy for LLaMA: {avg_accuracy_llama}")
        logging.info(f"Average xFinder accuracy for Qwen: {avg_accuracy_qwen}")
        logging.info(f"Total mismatches: {total_mismatches} out of {total_rows} rows")

        json.dump({
            "prompt_type": extract_value_from_key_in_file_name(os.path.basename(input_file), "prompt"),
            "xfinder_avg_accuracy_llama": avg_accuracy_llama,
            "xfinder_avg_accuracy_qwen": avg_accuracy_qwen,
            "xfinder_average_accuracy": round((avg_accuracy_llama + avg_accuracy_qwen) / 2, 4),
            "xfinder_mismatches_percentage": round((total_mismatches / total_rows) * 100, 2),
            "xfinder_total_mismatches": total_mismatches,
            "xfinder_total_rows": total_rows,
        }, f_out_json)


def get_latest_datetime_dir(base_dir):
    """
    Returns the latest subdirectory with format YYYY-MM-DD_HH-MM-SS.
    """
    dt_dirs = []

    try:
        entries = os.listdir(base_dir)
    except FileNotFoundError:
        logging.warning(f"Directory not found: {base_dir}. Skipping.")
        return None
    except NotADirectoryError:
        logging.warning(f"Not a directory: {base_dir}. Skipping.")
        return None

    for name in entries:
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            dt = datetime.datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
            dt_dirs.append((dt, path))
        except ValueError:
            continue

    if not dt_dirs:
        return None

    _, latest_path = max(dt_dirs, key=lambda x: x[0])
    return latest_path


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    base_dir = args.input_dir

    if not os.path.exists(base_dir):
        logging.warning(f"Input directory does not exist: {base_dir}. Exiting.")
        return

    selected_dir = get_latest_datetime_dir(base_dir) or base_dir
    logging.info(f"Using input directory: {selected_dir}")

    # Build list of files to process
    files_to_process = []
    try:
        for name in os.listdir(selected_dir):
            path = os.path.join(selected_dir, name)
            if not name.startswith(".") and os.path.isfile(path) and not file_already_processed(path):
                files_to_process.append(path)
    except FileNotFoundError:
        logging.warning(f"No files found in directory: {selected_dir}. Skipping.")
        return
    except NotADirectoryError:
        logging.warning(f"{selected_dir} is not a directory. Skipping.")
        return

    if not files_to_process:
        logging.info("No unprocessed files found.")
        return

    # Load models
    logging.info("Loading models…")
    model_dir = "models"
    xfinder_evaluator_qwen = xfinder_setup("xFinder-qwen1505", model_dir)
    xfinder_evaluator_llama = xfinder_setup("xFinder-llama38it", model_dir)

    # Prepare output
    output_dir = os.path.join(selected_dir, "accuracy")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving results to: {output_dir}")

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

    write_accuracy_summary(output_dir)
    logging.info("Evaluation process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on MCQA with xFinder")
    parser.add_argument('--input_dir', required=True, help='Directory containing the output of the models on the MCQA task.')
    args = parser.parse_args()
    main(args)
