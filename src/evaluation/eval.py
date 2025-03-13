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
        fieldnames = reader.fieldnames  # Dynamically get column names

        # Remove prompt field from the output file
        # it is only used for computing metrics in xFinder
        if "prompt" in fieldnames:
            fieldnames.remove("prompt")

        # Add xFinder output columns
        extended_fieldnames = fieldnames + [
            "xfinder_extracted_answer_llama", 
            "xfinder_acc_llama", 
            "xfinder_extracted_answer_qwen", 
            "xfinder_acc_qwen"
        ]
        
        writer = csv.DictWriter(f_out, fieldnames=extended_fieldnames, delimiter='\t')
        writer.writeheader()
        
        xfinder_accuracies_llama = []
        xfinder_accuracies_qwen = []
        
        for row in tqdm(reader, desc=f"Processing {input_file}"):
            prompt = row.get("prompt", None)
            choices = row.get("choices", None)
            model_output = row.get("model_output", None)
            gold_truth = row.get("gold_truth", None)
            
            if not choices:
                logging.warning(f"Skipping row due to missing choices: {row}")
                continue
            
            answer_range = [elem.split(". ") for elem in choices.split('\n')]
            
            result_llama = xfinder_evaluator_llama.evaluate_single_item(
                question=prompt,
                llm_output=model_output,
                answer_range=answer_range,
                answer_type="alphabet_option",
                correct_answer=gold_truth,
            )
            
            result_qwen = xfinder_evaluator_qwen.evaluate_single_item(
                question=prompt,
                llm_output=model_output,
                answer_range=answer_range,
                answer_type="alphabet_option",
                correct_answer=gold_truth,
            )
            
            xfinder_accuracies_llama.append(result_llama[-1])
            xfinder_accuracies_qwen.append(result_qwen[-1])

            row.update({
                "xfinder_extracted_answer_llama": result_llama[-3],
                "xfinder_acc_llama": result_llama[-1],
                "xfinder_extracted_answer_qwen": result_qwen[-3],
                "xfinder_acc_qwen": result_qwen[-1]
            })

            writer.writerow(row)
        
        avg_accuracy_llama = statistics.mean(xfinder_accuracies_llama)
        avg_accuracy_qwen = statistics.mean(xfinder_accuracies_qwen)
        logging.info(f"Average xFinder accuracy for LLaMA: {avg_accuracy_llama}")
        logging.info(f"Average xFinder accuracy for Qwen: {avg_accuracy_qwen}")
        
        # Write temporary JSON file that will be used later for summary
        json.dump({
            "prompt_type": os.path.splitext(os.path.basename(input_file))[0].split("prompt=")[1],
            "xfinder_avg_accuracy_llama": avg_accuracy_llama,
            "xfinder_avg_accuracy_qwen": avg_accuracy_qwen
        }, f_out_json)

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    model_dir = args.model_dir    
    
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
            
            # Define temporary output files
            output_path = os.path.join(output_dir, f"xFinder|{filename}.tsv")
            output_path_json = os.path.join(output_dir, f"xFinder|{filename}.json")
            
            # Check for the final results file (xFinder_accuracy|FILE_PREFIX.jsonl)
            prefix_part = file.split("prompt=")[0] + "prompt="
            final_results_path = os.path.join(output_dir, f"xFinder_accuracy|{prefix_part}.jsonl")
            if os.path.exists(final_results_path) and not args.overwrite:
                logging.info(f"Skipping {file} as final result file {final_results_path} already exists. Use --overwrite to force re-computation.")
                continue

            # Run evaluation on the selected file            
            evaluate(file_path,
                     output_path,
                     output_path_json,
                     xfinder_evaluator_llama,
                     xfinder_evaluator_qwen)
        
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
