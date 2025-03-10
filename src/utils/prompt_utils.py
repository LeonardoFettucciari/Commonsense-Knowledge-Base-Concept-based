from settings.prompts import (SYSTEM_ZEROSHOT, SYSTEM_ZEROSHOT_WITH_KNOWLEDGE, SYSTEM_ZEROSHOT_COT,
                              SYSTEM_FEWSHOT, SYSTEM_FEWSHOT_WITH_KNOWLEDGE, SYSTEM_FEWSHOT_COT)
from src.prompts.llama_prompt import LlamaPrompt

def build_prompts(sample, prompt_types, top_k_values, fewshot_examples=[]):
    prompts = []

    for p_type in prompt_types:
        if p_type == "zeroshot" or p_type == "all":
            prompts.append(LlamaPrompt( name="zeroshot",
                                        system_instruction=SYSTEM_ZEROSHOT,
                                        sample=sample,
                                        ))
            
        if p_type == "zeroshot_cot" or p_type == "all":
            prompts.append(LlamaPrompt( name="zeroshot_cot",
                                        system_instruction=SYSTEM_ZEROSHOT_COT,
                                        sample=sample,
                                        cot=True,
                                        ))
            
        if p_type == "zeroshot_with_knowledge" or p_type == "all":
            for k in top_k_values:
                prompts.append(LlamaPrompt( name=f"zeroshot_with_knowledge_{k}",
                                            system_instruction=SYSTEM_ZEROSHOT_WITH_KNOWLEDGE,
                                            sample=sample,
                                            top_k=k,
                                            ))
            
        if p_type == "fewshot" or p_type == "all":
            prompts.append(LlamaPrompt( name="fewshot",
                                        system_instruction=SYSTEM_FEWSHOT,
                                        sample=sample,
                                        fewshot_examples=fewshot_examples,
                                        ))
            
        if p_type == "fewshot_cot" or p_type == "all":
            prompts.append(LlamaPrompt( name="fewshot_cot",
                                        system_instruction=SYSTEM_FEWSHOT_COT,
                                        sample=sample,
                                        fewshot_examples=fewshot_examples,
                                        cot=True,
                                        ))
            
        if p_type == "fewshot_with_knowledge" or p_type == "all":
            for k in top_k_values:
                prompts.append(LlamaPrompt( name=f"fewshot_with_knowledge_{k}",
                                            system_instruction=SYSTEM_FEWSHOT_WITH_KNOWLEDGE,
                                            sample=sample,
                                            fewshot_examples=fewshot_examples,
                                            top_k=k,
                                            ))
    return prompts