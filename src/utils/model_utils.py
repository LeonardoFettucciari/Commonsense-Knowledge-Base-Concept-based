from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import google.generativeai as genai
from tqdm import tqdm


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


def get_answer(llm, synsets_list, definitions_list):
    sample_output = []
    for synset, definition in zip(synsets_list, definitions_list):
        for s, d in zip(synset, definition):

            prompt_message = {"role": "user", "parts": [f'Concept: {s.name().split(".")[0]}. Definition: {d}.']}
            

            # Start chat session
            chat_session = llm.start_chat(history=[])

            # Send message and get response
            model_output = chat_session.send_message(prompt_message).text

            # Current sample output
            sample_output.append(model_output)
    return sample_output

def get_answers(llm, synsets_all_samples, definitions_all_samples):
    all_outputs = []
    for synsets_list, definitions_list in tqdm(zip(synsets_all_samples, definitions_all_samples)):
        sample_output = get_answer(llm, synsets_list, definitions_list)
        all_outputs.append(sample_output)
    return all_outputs

