import os
import csv
import json
import argparse
import statistics
from natsort import natsorted
from xfinder.eval import Evaluator
from tqdm import tqdm
from huggingface_hub import snapshot_download
from src.utils.io_utils import jsonl_to_tsv, bundle_json

def xfinder_setup(model_name, model_dir):
    model_path = f"{model_dir}/IAAR-Shanghai/{model_name}"
    
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
    with open(input_file) as csvfile, open(output_path, "w") as f_out, open(output_path_json, "w") as f_out_json:
        reader = csv.DictReader(csvfile, delimiter='\t')
        fieldnames = reader.fieldnames  # Dynamically get column names
        
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
        
        for row in tqdm(reader):
            prompt = row.get("prompt", None)
            choices = row.get("choices", None)
            model_output = row.get("model_output", None)
            gold_truth = row.get("gold_truth", None)
            
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
            
            row.update({
                "xfinder_extracted_answer_llama": result_llama[-3],
                "xfinder_acc_llama": result_llama[-1],
                "xfinder_extracted_answer_qwen": result_qwen[-3],
                "xfinder_acc_qwen": result_qwen[-1]
            })
            
            xfinder_accuracies_llama.append(result_llama[-1])
            xfinder_accuracies_qwen.append(result_qwen[-1])
            writer.writerow(row)
        
        json.dump({
            "file_name": input_file.split('/')[-1],
            "xfinder_avg_accuracy_llama": statistics.mean(xfinder_accuracies_llama),
            "xfinder_avg_accuracy_qwen": statistics.mean(xfinder_accuracies_qwen)
        }, f_out_json)


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    model_dir = args.model_dir    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download and setup models
    xfinder_evaluator_qwen = xfinder_setup("xFinder-qwen1505", model_dir)
    xfinder_evaluator_llama = xfinder_setup("xFinder-llama38it", model_dir)
    
    # Scan all files inside input_dir
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)  
        if os.path.isfile(file_path):  
            filename, _ = os.path.splitext(os.path.basename(file))
            output_path = os.path.join(output_dir, f"xFinder|{filename}.tsv")
            output_path_json = os.path.join(output_dir, f"xFinder|{filename}.json")
            # Run evaluation on selected file            
            evaluate(os.path.join(input_dir, file),
                     output_path,
                     output_path_json,
                     xfinder_evaluator_llama,
                     xfinder_evaluator_qwen)
        
    # Write summary file for accuracy of all prompts types i.e. zeroshot-accuracy, etc.
    accuracy_summary_output_name = f"xFinder|accuracies.jsonl"
    bundle_json(output_dir, accuracy_summary_output_name)
    jsonl_to_tsv(os.path.join(output_dir, accuracy_summary_output_name))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on MCQA with xFinder")
    parser.add_argument('--input_dir', help='Directory containing the output of the models on the MCQA task.')
    parser.add_argument('--output_dir', help='Directory to save the evaluation results.')
    parser.add_argument('--model_dir', default='models', help='Directory to save the evaluation models.')
    args = parser.parse_args()

    # Default value for --output_dir
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "accuracy")

    main(args)
