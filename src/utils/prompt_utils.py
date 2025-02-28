from settings.prompts import (SYSTEM_ZEROSHOT, SYSTEM_ZEROSHOT_WITH_KNOWLEDGE, SYSTEM_ZEROSHOT_COT,
                              SYSTEM_FEWSHOT, SYSTEM_FEWSHOT_WITH_KNOWLEDGE, SYSTEM_FEWSHOT_COT)

def prepare_prompt_input_data(sample, sample_knowledge, fewshot_examples, fewshot_knowledge):
  input_data = []
  input_data.append({
                    "system_instruction": SYSTEM_ZEROSHOT,
                    "question": sample["question_stem"],
                    "choices": "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
                })
  
  input_data.append({
                    "system_instruction": SYSTEM_FEWSHOT,
                    "question": sample["question_stem"],
                    "choices": "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
                    "fewshot_examples": fewshot_examples
                })
  input_data.append({
                    "system_instruction": SYSTEM_ZEROSHOT_COT,
                    "question": sample["question_stem"],
                    "choices": "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
                })
  input_data.append({
                    "system_instruction": SYSTEM_FEWSHOT_COT,
                    "question": sample["question_stem"],
                    "choices": "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
                    "fewshot_examples": fewshot_examples
                })
  input_data.append({
                    "system_instruction": SYSTEM_ZEROSHOT_WITH_KNOWLEDGE,
                    "question": sample["question_stem"],
                    "choices": "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
                    "knowledge": sample_knowledge
                })
  input_data.append({
                    "system_instruction": SYSTEM_FEWSHOT_WITH_KNOWLEDGE,
                    "question": sample["question_stem"],
                    "choices": "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
                    "knowledge": sample_knowledge,
                    "fewshot_examples": fewshot_examples,
                    "fewshot_knowledge": fewshot_knowledge,
                    
                })
  return input_data