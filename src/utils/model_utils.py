from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM, pipeline
import google.generativeai as genai
from tqdm import tqdm
import re
import json
import torch
from src.utils.io_utils import load_json
from typing import List, Dict, Any




def get_model_settings(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return json.load(f)


import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

def load_model_and_tokenizer(
    model_name: str,
    device: torch.device = None,
    max_retries: int = 5,
    retry_wait: int = 30
):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for attempt in range(1, max_retries + 1):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left",
            )
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            return model, tokenizer
        except (HfHubHTTPError, OSError, RepositoryNotFoundError) as e:
            print(f"[Attempt {attempt}] Download failed: {e}")
            if attempt == max_retries:
                raise
            print(f"Retrying in {retry_wait} seconds...")
            time.sleep(retry_wait)



def get_ner_pipeline(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )


def load_gemini_model(
    model_name: str,
    generation_config: Dict[str, Any],
    system_instruction: str,
    api_key: str,
):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        system_instruction=system_instruction,
    )


def clean_statement(model_output: str) -> List[str]:
    parts = re.split(r'^\d+\.\s+', model_output, flags=re.MULTILINE)
    return [s.strip() for s in parts if s.strip()]


def get_statement(
    llm, synsets_list: Any, definitions_list: Any
) -> List[Dict[str, Any]]:
    all_outputs = []
    for synset, definition in zip(synsets_list, definitions_list):
        for s, d in zip(synset, definition):
            prompt_message = {"role": "user", "parts": [f'Concept: {s.name().split(".")[0]}. Definition: {d}.']}
            chat_session = llm.start_chat(history=[])
            model_output = chat_session.send_message(prompt_message).text
            cleaned = clean_statement(model_output)
            all_outputs.append({
                "synset": s.name(),
                "definition": d,
                "statements": cleaned,
            })
    return all_outputs


def get_statements(llm, synsets_all_samples: Any, definitions_all_samples: Any) -> Any:
    all_batches = []
    for synsets_list, definitions_list in tqdm(zip(synsets_all_samples, definitions_all_samples)):
        all_batches.append(get_statement(llm, synsets_list, definitions_list))
    return all_batches


def generate_text(model,
                  tokenizer,
                  prompt,
                  config_path="settings/model_config.json",
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Load the model's default generation config
    gen_config = model.generation_config
    # Check and override top_p
    if hasattr(gen_config, "top_p"):
        gen_config.top_p = None  # or remove it entirely
        
    # Retrieve model configuration parameters
    config = load_json(config_path)["generation_config"]

    # Convert to chat template
    inputs = tokenizer.apply_chat_template(prompt.messages, return_tensors="pt", add_generation_prompt=True).to(device)
    #inputs_text = tokenizer.apply_chat_template(prompt.messages, add_generation_prompt=True, tokenize=False)
    
    attention_mask = (inputs != tokenizer.pad_token_id).long().to(device)
    
    # Generate text using the model.
    with torch.no_grad():
        model_outputs = model.generate(
            inputs,
            attention_mask          =attention_mask,
            pad_token_id            =tokenizer.pad_token_id,
            max_new_tokens          =config.get("max_new_tokens", 512),
            do_sample               =config.get("do_sample", False),
            temperature             =config.get("temperature", 0.0),
            output_scores           =config.get("output_scores", False),
            return_dict_in_generate =config.get("return_dict_in_generate", True),
        )

    # If return_dict_in_generate=True, 'model_outputs' is a dictionary with 'sequences'
    # If return_dict_in_generate=False, 'model_outputs' is a tensor directly.
    if config.get("return_dict_in_generate", False):
        sequences = model_outputs.sequences
    else:
        sequences = model_outputs

    # Decode only the newly generated tokens after the input prompt
    # Shape: [batch_size, seq_length]
    generated_text = tokenizer.decode(
        sequences[0][inputs.shape[1]:],
        skip_special_tokens=True,
    )

    return generated_text




def batched_generate_text(
    model,
    tokenizer,
    prompts: List[Any],  # prompts must have a .messages attribute
    gen_config: Dict[str, Any],
    device: torch.device = None,
) -> List[str]:
    """
    Batch all prompts through model.generate in one call, with a debug check
    to ensure prompt_lengths matches single-prompt tokenization.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Batch-tokenize with chat template (may return dict or Tensor)
    batch = tokenizer.apply_chat_template(
        [p.messages for p in prompts],
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
    )
    if isinstance(batch, dict):
        input_ids = batch["input_ids"].to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

    else:
        input_ids = batch.to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)


    # 4) Generate in one go
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            generation_config=gen_config,
        )

    # 5) Extract sequences
    sequences = outputs.sequences if gen_config.return_dict_in_generate else outputs

    

    # 6) Slice and decode
    decoded_texts: List[str] = []
    for i, seq in enumerate(sequences):
        prompt_len = input_ids.shape[1]  # total length of the input prompt
        new_tokens = seq[prompt_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        decoded_texts.append(decoded)

    return decoded_texts

