# Models
MODEL_NAME_TO_TAG = {
    "meta-llama/Llama-3.2-3B-Instruct": "llama3B",
    "meta-llama/Llama-3.1-8B-Instruct": "llama8B",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen5B",
    "Qwen/Qwen2.5-7B-Instruct": "qwen7B",
}

MODEL_TAG_TO_NAME = {
    "llama3B": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8B": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen7B": "Qwen/Qwen2.5-7B-Instruct",
}

# Datasets
DATASET_NAME_TO_TAG = {
    "allenai/openbookqa": "obqa",
    "tau/commonsense_qa": "csqa",
    "allenai/qasc": "qasc",
}

DATASET_TAG_TO_NAME = {
    "obqa": "allenai/openbookqa",
    "csqa": "tau/commonsense_qa",
    "qasc": "allenai/qasc",
}

# Prompts
PROMPT_TYPE_ALIASES = {
    "zs": "zeroshot",
    "zero-shot": "zeroshot",
    "zero_shot": "zeroshot",
    "zeroshot": "zeroshot",

    "zscot": "zeroshot_cot",
    "zs_cot": "zeroshot_cot",
    "zero-shot_cot": "zeroshot_cot",
    "zero_shot_cot": "zeroshot_cot",
    "zeroshot_cot": "zeroshot_cot",

    "zsk": "zeroshot_with_knowledge",
    "zero-shot_with_knowledge": "zeroshot_with_knowledge",
    "zero_shot_with_knowledge": "zeroshot_with_knowledge",
    "zeroshot_with_knowledge": "zeroshot_with_knowledge",

    "zscotk": "zeroshot_cot_with_knowledge",
    "zs_cot_k": "zeroshot_cot_with_knowledge",
    "zero-shot_cot_with_knowledge": "zeroshot_cot_with_knowledge",
    "zeroshot_cot_with_knowledge": "zeroshot_cot_with_knowledge",


    "fs": "fewshot",
    "few-shot": "fewshot",
    "few_shot": "fewshot",
    "fewshot": "fewshot",

    "fscot": "fewshot_cot",
    "fs_cot": "fewshot_cot",
    "few-shot_cot": "fewshot_cot",
    "few_shot_cot": "fewshot_cot",
    "fewshot_cot": "fewshot_cot",

    "fsk": "fewshot_with_knowledge",
    "few-shot_with_knowledge": "fewshot_with_knowledge",
    "few_shot_with_knowledge": "fewshot_with_knowledge",
    "fewshot_with_knowledge": "fewshot_with_knowledge",

}

FULL_PROMPT_TO_SHORT_PROMPT = {
    "zeroshot_with_knowledge": "zsk",
    "zeroshot_cot": "zscot",
    "zeroshot": "zs",
    "fewshot_with_knowledge": "fsk",
    "fewshot_cot": "fscot",
    "fewshot": "fs",    
}