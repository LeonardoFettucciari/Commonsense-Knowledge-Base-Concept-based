from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import google.generativeai as genai
from tqdm import tqdm
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.io_utils import load_json


# Load model settings
def get_model_settings(config_path):
    with open(config_path) as f:
        settings = json.load(f)
    return settings

def load_model_and_tokenizer(model_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
    
    # Ensure pad_token_id is set
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


# NER model
def get_ner_pipeline(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# Gemini
def load_gemini_model(model_name, generation_config, system_instruction, api_key):
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        system_instruction=system_instruction,
    )

    return model

def clean_statement(model_output):
    cleaned_statements = [s.strip() for s in re.split(r'\n?\d+\.\s*', model_output) if s.strip()]
    return cleaned_statements

def get_statement(llm, synsets_list, definitions_list):
    sample_output = []
    for synset, definition in zip(synsets_list, definitions_list):
        for s, d in zip(synset, definition):
            prompt_message = {"role": "user", "parts": [f'Concept: {s.name().split(".")[0]}. Definition: {d}.']}
            # Start chat session
            chat_session = llm.start_chat(history=[])
            # Send message and get response
            model_output = chat_session.send_message(prompt_message).text
            # Extract statements' text only and clean them i.e. remove '5.' out of '5. <Statement number 5>'
            cleaned_statements = clean_statement(model_output)
            # Current sample output
            output_dict = {
                "synset": s.name(),
                "definition": d,
                "statements": cleaned_statements
            }
            sample_output.append(output_dict)
    return sample_output

def get_statements(llm, synsets_all_samples, definitions_all_samples):
    all_outputs = []
    for synsets_list, definitions_list in tqdm(zip(synsets_all_samples, definitions_all_samples)):
        sample_output = get_statement(llm, synsets_list, definitions_list)
        all_outputs.append(sample_output)
    return all_outputs
    
# Generate text
def generate_text(model,
                  tokenizer,
                  prompt,
                  config_path="settings/model_config.json",
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    # Retrieve model configuration parameters

    config = load_json(config_path)["generation_config"]

    # Convert to chat template
    inputs = tokenizer.apply_chat_template(prompt.messages, return_tensors="pt", add_generation_prompt=True).to(device)

    attention_mask = torch.ones_like(inputs).to(device)
    
    # Generate text using the model.
    model_outputs = model.generate(
        inputs,
        attention_mask          =attention_mask,
        pad_token_id            =tokenizer.pad_token_id,
        max_new_tokens          =config.get("max_new_tokens", 512),
        do_sample               =config.get("do_sample", False),
        temperature             =config.get("temperature", 0.0),
        output_scores           =config.get("output_scores", False),
        return_dict_in_generate =config.get("return_dict_in_generate", False),
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