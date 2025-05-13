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

def main(args):
    input_dir = args.input_dir
    output_dir = os.path.join(input_dir, "accuracy")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.info(f"Processing input directory: {input_dir}")
    logging.info(f"Saving results to: {output_dir}")

    # Download and setup models
    model_dir = "models"
    xfinder_evaluator_qwen = xfinder_setup("xFinder-qwen1505", model_dir)
    xfinder_evaluator_llama = xfinder_setup("xFinder-llama38it", model_dir)
    
    # Scan all files inside input_dir
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        
        # If not a file
        if not os.path.isfile(file_path): continue

        # If file already processed, skip it
        if file_already_processed(file_path): continue
        if file.startswith("."): continue

        # Otherwise process it and mark it as processed
        mark_file_as_processed(file_path)
        logging.info(f"Processing file: {file}")

        filename, _ = os.path.splitext(os.path.basename(file))

        # Shorten filename for better readability
        filename_metadata = extract_key_value_pairs(filename)
        if filename_metadata.get('prompt'):
            filename_metadata['prompt'] = alias_filename(filename_metadata['prompt'])
        filename = dict_to_filename(filename_metadata)
        
        # Define temporary output files
        output_path = os.path.join(output_dir, f"{filename}.tsv")
        output_path_json = os.path.join(output_dir, f"{filename}.json")

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
    args = parser.parse_args()

    main(args)