from settings.prompts import (SYSTEM_ZEROSHOT, SYSTEM_ZEROSHOT_WITH_KNOWLEDGE, SYSTEM_ZEROSHOT_COT,
                              SYSTEM_FEWSHOT, SYSTEM_FEWSHOT_WITH_KNOWLEDGE, SYSTEM_FEWSHOT_COT)

def prepare_prompt_input_data(sample, sample_knowledge, fewshot_examples, fewshot_knowledge, top_k_list):
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
  for k in top_k_list:
    input_data.append({
                        "prompt_name": f"zeroshot_with_knowledge_top_{k}",
                        "system_instruction": SYSTEM_ZEROSHOT_WITH_KNOWLEDGE,
                        "question": sample["question_stem"],
                        "choices": "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
                        "knowledge": sample_knowledge[:k]
                    })
    input_data.append({
                        "prompt_name": f"fewshot_with_knowledge_top_{k}",
                        "system_instruction": SYSTEM_FEWSHOT_WITH_KNOWLEDGE,
                        "question": sample["question_stem"],
                        "choices": "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])]),
                        "knowledge": sample_knowledge[:k],
                        "fewshot_examples": fewshot_examples,
                        "fewshot_knowledge": [inner[:k] for inner in fewshot_knowledge],
                    })
  return input_data