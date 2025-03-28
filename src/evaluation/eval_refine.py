import os
import csv
import json
import argparse
import statistics
import logging
from natsort import natsorted
from xfinder.eval import Evaluator
from tqdm import tqdm
from huggingface_hub import snapshot_download
from src.utils.io_utils import write_accuracy_summary
from src.utils.string_utils import extract_key_value_pairs, shorten_prompt, key_value_pairs_to_filename

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

def evaluate(input_file, output_path, output_path_json, xfinder_evaluator_llama=None, xfinder_evaluator_qwen=None):
    logging.info(f"Evaluating file: {input_file}")
    with open(input_file) as csvfile, open(output_path, "w") as f_out, open(output_path_json, "w") as f_out_json:
        reader = csv.DictReader(csvfile, delimiter='\t')
        fieldnames = reader.fieldnames

        # Add xFinder output columns
        extended_fieldnames = fieldnames + [
            "xfinder_extracted_answer_llama", 
            "xfinder_acc_llama", 
            "xfinder_extracted_answer_qwen", 
            "xfinder_acc_qwen",
            "xfinder_extracted_answer_llama_refine", 
            "xfinder_acc_llama_refine", 
            "xfinder_extracted_answer_qwen_refine", 
            "xfinder_acc_qwen_refine",
            "good_answer_change_llama",
            "bad_answer_change_llama",
            "good_answer_change_qwen",
            "bad_answer_change_qwen",
            "xfinder_extracted_answers_mismatch",
            "xfinder_extracted_answers_mismatch_refine",
        ]

        writer = csv.DictWriter(f_out, fieldnames=extended_fieldnames, delimiter='\t')
        writer.writeheader()

        # Accuracy lists
        xfinder_accuracies_llama = []
        xfinder_accuracies_qwen = []
        xfinder_accuracies_llama_refine = []
        xfinder_accuracies_qwen_refine = []

        # Change tracking lists
        good_answer_change_llama_list = []
        bad_answer_change_llama_list = []
        good_answer_change_qwen_list = []
        bad_answer_change_qwen_list = []

        # Mismatch tracking
        xfinder_extracted_answers_mismatches = []
        xfinder_extracted_answers_mismatches_refine = []

        for row in tqdm(reader, desc=f"Processing {input_file}"):
            try:
                question = row.get("question")
                choices = row.get("choices")
                model_output = row.get("model_output")
                model_output_refine = row.get("model_output_refine")
                ground_truth = row.get("ground_truth")

                answer_range = [elem.split(". ") for elem in choices.split('\n')]
                formatted_choices = " ".join([f"({item[0]}) {item[1]}" for item in answer_range])
                formatted_question = f"{question} Answer Choices: {formatted_choices}"
            except Exception as e:
                logging.info(f"Error processing row: {e}")
                continue

            # Evaluate original
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

            row.update({
                "xfinder_extracted_answer_llama": result_llama[-3],
                "xfinder_acc_llama": result_llama[-1],
                "xfinder_extracted_answer_qwen": result_qwen[-3],
                "xfinder_acc_qwen": result_qwen[-1],
            })

            # Evaluate refined
            result_llama_refine = xfinder_evaluator_llama.evaluate_single_item(
                question=formatted_question,
                llm_output=model_output_refine,
                answer_range=answer_range,
                answer_type="alphabet_option",
                correct_answer=ground_truth,
            )
            result_qwen_refine = xfinder_evaluator_qwen.evaluate_single_item(
                question=formatted_question,
                llm_output=model_output_refine,
                answer_range=answer_range,
                answer_type="alphabet_option",
                correct_answer=ground_truth,
            )

            xfinder_accuracies_llama_refine.append(result_llama_refine[-1])
            xfinder_accuracies_qwen_refine.append(result_qwen_refine[-1])

            row.update({
                "xfinder_extracted_answer_llama_refine": result_llama_refine[-3],
                "xfinder_acc_llama_refine": result_llama_refine[-1],
                "xfinder_extracted_answer_qwen_refine": result_qwen_refine[-3],
                "xfinder_acc_qwen_refine": result_qwen_refine[-1],
            })

            # Compute mismatches
            mismatch_orig = 1 if result_llama[-3] != result_qwen[-3] else 0
            mismatch_refine = 1 if result_llama_refine[-3] != result_qwen_refine[-3] else 0
            xfinder_extracted_answers_mismatches.append(mismatch_orig)
            xfinder_extracted_answers_mismatches_refine.append(mismatch_refine)

            row.update({
                "xfinder_extracted_answers_mismatch": mismatch_orig,
                "xfinder_extracted_answers_mismatch_refine": mismatch_refine,
            })

            # Track changes
            good_llama = 1 if result_llama[-3] != result_llama_refine[-3] and result_llama_refine[-1] == 1 else 0
            bad_llama = 1 if result_llama[-3] != result_llama_refine[-3] and result_llama_refine[-1] == 0 else 0
            good_qwen = 1 if result_qwen[-3] != result_qwen_refine[-3] and result_qwen_refine[-1] == 1 else 0
            bad_qwen = 1 if result_qwen[-3] != result_qwen_refine[-3] and result_qwen_refine[-1] == 0 else 0

            good_answer_change_llama_list.append(good_llama)
            bad_answer_change_llama_list.append(bad_llama)
            good_answer_change_qwen_list.append(good_qwen)
            bad_answer_change_qwen_list.append(bad_qwen)

            row.update({
                "good_answer_change_llama": good_llama,
                "bad_answer_change_llama": bad_llama,
                "good_answer_change_qwen": good_qwen,
                "bad_answer_change_qwen": bad_qwen,
            })

            row = {k: v for k, v in row.items() if k is not None}
            writer.writerow(row)

        # Averages
        avg_llama = statistics.mean(xfinder_accuracies_llama)
        avg_qwen = statistics.mean(xfinder_accuracies_qwen)
        avg_llama_refine = statistics.mean(xfinder_accuracies_llama_refine)
        avg_qwen_refine = statistics.mean(xfinder_accuracies_qwen_refine)

        total_mismatch = sum(xfinder_extracted_answers_mismatches)
        total_mismatch_refine = sum(xfinder_extracted_answers_mismatches_refine)
        total_rows = len(xfinder_extracted_answers_mismatches)

        # Summary JSON
        json.dump({
            "prompt_type": os.path.splitext(os.path.basename(input_file))[0].split("prompt=")[1],

            "xfinder_avg_accuracy_llama": avg_llama,
            "xfinder_avg_accuracy_qwen": avg_qwen,
            "xfinder_average_accuracy": round((avg_llama + avg_qwen) / 2, 4),

            "xfinder_avg_accuracy_llama_refine": avg_llama_refine,
            "xfinder_avg_accuracy_qwen_refine": avg_qwen_refine,
            "xfinder_average_accuracy_refine": round((avg_llama_refine + avg_qwen_refine) / 2, 4),

            "xfinder_total_mismatches": total_mismatch,
            "xfinder_total_mismatches_refine": total_mismatch_refine,
            "xfinder_mismatches_percentage": round((total_mismatch / total_rows) * 100, 2),
            "xfinder_mismatches_percentage_refine": round((total_mismatch_refine / total_rows) * 100, 2),
            "xfinder_total_rows": total_rows,

            "good_answer_changes_llama": sum(good_answer_change_llama_list),
            "bad_answer_changes_llama": sum(bad_answer_change_llama_list),
            "good_answer_changes_qwen": sum(good_answer_change_qwen_list),
            "bad_answer_changes_qwen": sum(bad_answer_change_qwen_list),
        }, f_out_json)


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    model_dir = args.model_dir
    overwrite = args.overwrite
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.info(f"Processing input directory: {input_dir}")
    logging.info(f"Saving results to: {output_dir}")

    # Download and setup models
    xfinder_evaluator_qwen = xfinder_setup("xFinder-qwen1505", model_dir)
    xfinder_evaluator_llama = xfinder_setup("xFinder-llama38it", model_dir)
    
    # Scan all files inside input_dir
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)  
        if os.path.isfile(file_path):
            logging.info(f"Processing file: {file}")
            filename, _ = os.path.splitext(os.path.basename(file))
            
            # Shorten filename for better readability
            filename_metadata = extract_key_value_pairs(filename)
            filename_metadata['prompt'] = shorten_prompt(filename_metadata['prompt'])
            filename = key_value_pairs_to_filename(filename_metadata)

            output_path = os.path.join(output_dir, f"{filename}.tsv")
            output_path_json = os.path.join(output_dir, f"{filename}.json")
            
            # Check for the final results file (xFinder_accuracy|FILE_PREFIX.jsonl)
            prefix_part = file.split("prompt=")[0] + "prompt="
            final_results_path = os.path.join(output_dir, f"xFinder_accuracy|{prefix_part}.jsonl")
            if os.path.exists(final_results_path) and not overwrite:
                logging.info(f"Skipping {file} as final result file {final_results_path} already exists. Use --overwrite to force re-computation.")
                continue

            # Run evaluation on the selected file            
            evaluate(file_path,
                     output_path,
                     output_path_json,
                     xfinder_evaluator_llama,
                     xfinder_evaluator_qwen,
            )
        
    # Write summary file for accuracy of all prompt types (e.g., zeroshot-accuracy, etc.)
    write_accuracy_summary(output_dir)
    logging.info("Evaluation process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on MCQA with xFinder")
    parser.add_argument('--input_dir', required=True, help='Directory containing the output of the models on the MCQA task.')
    parser.add_argument('--output_dir', required=False, help='Directory to save the evaluation results.')
    parser.add_argument('--model_dir', default='models', required=False, help='Directory to save the evaluation models.')
    parser.add_argument('--overwrite', action='store_true', help='Force re-computation even if final results already exist.')
    args = parser.parse_args()

    # Default value for --output_dir
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "accuracy")

    main(args)
