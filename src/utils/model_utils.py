from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import google.generativeai as genai
from tqdm import tqdm
import re
import json
import torch

# Load model settings
def get_model_settings(config_path):
    with open(config_path) as f:
        settings = json.load(f)
    return settings

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

def truncate_inputs(inputs, model_name):
    if model_name == "meta-llama/Llama-3.2-3B-Instruct":
        return inputs[:, :-1]
    elif model_name == "meta-llama/Llama-3.1-8B-Instruct":
        return inputs[:, :-1]
    elif model_name == "Qwen/Qwen2.5-1.5B-Instruct":
        return inputs[:, :-2]
    elif model_name == "Qwen/Qwen2.5-7B-Instruct":
        return inputs[:, :-2]
    else:
        return inputs[:, :-1]
    
# Generate text
def generate_text(model,
                  tokenizer,
                  prompt,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    # Retrieve model configuration parameters
    settings = get_model_settings("settings/generic_llm_config.json")["generation_config"]

    # Convert to chat template
    inputs = tokenizer.apply_chat_template(prompt.messages, return_tensors="pt", add_generation_prompt=True).to(device)

    # Truncate depending on model used
    #inputs = truncate_inputs(inputs, model.model_name)
    
    # Create the attention mask.
    attention_mask = torch.ones_like(inputs).to(device)

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Generate text using the model.
    model_outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=settings["max_new_tokens"],
        do_sample=settings["do_sample"],
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        num_beams=settings["num_beams"],
        output_scores=settings["output_scores"],
        return_dict_in_generate=settings["return_dict_in_generate"],
    )

    # Decode the generated text.
    generated_text = tokenizer.decode(
        model_outputs.sequences[0][inputs[0].shape[0] :],
        skip_special_tokens=True,
    )

    return generated_text