from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import google.generativeai as genai
from tqdm import tqdm
import re
import json

def get_model_settings(config_path):
    with open(config_path) as f:
        setting = json.load(f)
    return setting

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

# Retriever

import torch
import re
def generate_text(model,
                  tokenizer,
                  prompt,
                  model_name,
                  max_new_tokens=512,
                  system=True,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Convert to chat template
    messages = []
    if system:
        messages.append({"role": "system", "content": prompt.pop(0)})
        
    for turn_id, turn in enumerate(prompt):
        if turn_id % 2 == 0:
            messages.append({"role": "user", "content": turn})
        else:
            messages.append({"role": "assistant", "content": turn})

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    # Truncate depending on model used
    if model_name == "meta-llama/Llama-3.2-3B-Instruct":
        inputs = inputs[:, :-1]
    elif model_name == "meta-llama/Llama-3.1-8B-Instruct":
        inputs = inputs[:, :-1]
    elif model_name == "Qwen/Qwen2.5-1.5B-Instruct":
        inputs = inputs[:, :-2]
    elif model_name == "Qwen/Qwen2.5-7B-Instruct":
        inputs = inputs[:, :-2]
    else:
        inputs = inputs[:, :-1]


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
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # Decode the generated text.
    generated_text = tokenizer.decode(
        model_outputs.sequences[0][inputs[0].shape[0] :],
        skip_special_tokens=True,
    )

    return model_outputs, generated_text


def get_answers(model,
                tokenizer,
                prompt,
                model_name,
                max_new_tokens=512,
                system=True,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    # Generate alternative labels with whitespaces in front.
    labels = ['A', 'B', 'C', 'D', 'E']
    labels.extend([f" {label}" for label in labels])

    # Generate text using the model.
    model_outputs, raw_generated_answer = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        system=system,
        device=device
    )

    # Get the probabilities of the first token.
    probabilities = torch.log_softmax(model_outputs.scores[-1], dim=-1)[0]

    # Check that the labels are in the tokenizer's vocabulary.
    labels = [label for label in labels if len(tokenizer.tokenize(label)) == 1]

    # Get the label IDs.
    label_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]

    # Get the probability of each label (A, B, C, D, E) and its variants.
    answer = [probabilities[label_id].item() for label_id in label_ids]


    # Get the label with the highest probability.
    answer = labels[answer.index(max(answer))]
    answer = answer.lstrip()

    return raw_generated_answer, answer