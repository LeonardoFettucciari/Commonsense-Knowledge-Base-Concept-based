import os
import yaml
import jsonlines
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Optional, List
from transformers import AutoTokenizer
from datasets import load_dataset


MAX_BATCH_TOKENS = 1000000

full2short = {
        "allenai/openbookqa": "obqa",
        "allenai/qasc": "qasc",
        "tau/commonsense_qa": "csqa",
    }

short2full = {
        "obqa": "allenai/openbookqa",
        "qasc": "allenai/qasc",
        "csqa": "tau/commonsense_qa",
    }


def create_gpt_prompt(
        question: str,
        choices: str,
        answer: str,
        model_name: str,
        system_instruction: str,
    ) -> List[dict]:

    user_content = (
        f'Question:\n{question}\n\n'
        f'Choices:\n{choices}\n\n'
        f'Answer:\n{answer}\n\n'
    )

    if "o1" in model_name:
        prompt_messages = [{"role": "user", "content": user_content}]
    else:
        prompt_messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]
    return prompt_messages


def load_config(config_path: str):
    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_batch(
    config_path: str,
    dataset_path: str,
    output_dir: str,
    limit_samples: Optional[int] = None,
):
    config = load_config(config_path=config_path)

    model_name = config["model_name"]
    generation_config = config["generation_config"]
    system_instruction = config["system_instruction"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")

    dataset_name = full2short.get(dataset_path)
    if dataset_name is None:
        raise ValueError(f"Unknown dataset alias for: {dataset_path}")

    dataset_cfg = config.get(dataset_name)
    print(f"Loading dataset: {dataset_cfg}")
    
    all_samples = load_dataset(
        dataset_cfg.get('path'),
        dataset_cfg.get('subset'),
        split=dataset_cfg.get('split')
    )
    print(f"Loaded {len(all_samples)} samples from dataset {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    sample_buffer = []
    token_buffer, start, end = 0, 0, 0
    for i, line in tqdm(enumerate(all_samples), total=len(all_samples)):
        question_id = line["id"]
        question = line.get("question") or line.get("question_stem")
        choices = " ".join([f"{label}. {choice}" for label, choice in zip(line['choices']['label'], line['choices']['text'])])
        answer = line["answerKey"]
        messages = create_gpt_prompt(
            question=question,
            choices=choices,
            answer=answer,
            model_name=model_name,
            system_instruction=system_instruction
        )
        request_id = f'question-id-{question_id}'

        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")[0]
        token_count = len(input_ids)
        token_buffer += token_count

        print(f"[{i}] Request ID: {request_id}, Tokens: {token_count}, Token buffer: {token_buffer}")

        if "o1" in model_name or "o3" in model_name:
            request = {
                "custom_id": request_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": messages,
                    "temperature": generation_config["temperature"],
                    "max_completion_tokens": generation_config["max_completion_tokens"]
                }
            }
        else:
            request = {
                "custom_id": request_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": messages,
                    "temperature": generation_config["temperature"],
                    "max_tokens": generation_config["max_tokens"]
                }
            }

        sample_buffer.append(request)
        end = i

        if token_buffer > MAX_BATCH_TOKENS:
            print(f"Writing batch: start={start}, end={end}, buffer size={len(sample_buffer)}, tokens={token_buffer}")
            output_path = os.path.join(
                output_dir, 
                f"batch_api_data_{dataset_name}_range_{start}_{end}_model_{model_name}.jsonl"
            )
            with jsonlines.open(output_path, "w") as fout:
                for sample in sample_buffer:
                    fout.write(sample)
            print(f"Wrote file: {output_path}")
            start = end
            sample_buffer = []
            token_buffer = 0

        if i == limit_samples:
            print(f"Reached limit_samples: {limit_samples}")
            if sample_buffer:
                output_path = os.path.join(
                    output_dir, 
                    f"batch_api_data_{dataset_name}_range_{start}_{end}_model_{model_name}.jsonl"
                )
                with jsonlines.open(output_path, "w") as fout:
                    for sample in sample_buffer:
                        fout.write(sample)
                print(f"Wrote final limited batch: {output_path}")
            else:
                print("Warning: limit reached but sample_buffer is empty.")
            break

    if sample_buffer:
        output_path = os.path.join(
            output_dir, 
            f"batch_api_data_{dataset_name}_range_{start}_{end}_model_{model_name}.jsonl"
        )
        with jsonlines.open(output_path, "w") as fout:
            for sample in sample_buffer:
                fout.write(sample)
        print(f"Wrote final batch: {output_path}")
    else:
        print("No remaining samples to write at end of processing.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Create batches for GPT APIs.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the GPT config.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path containing the input data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the batches.")
    parser.add_argument("--limit_samples", type=int, required=False, help="Maximum number of elements to consider.")
    args = parser.parse_args()
    args.dataset_path = short2full.get(args.dataset_path, args.dataset_path)
    create_batch(**vars(args))
