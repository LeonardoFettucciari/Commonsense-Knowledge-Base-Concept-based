import copy
from settings.prompts import (SYSTEM_ZEROSHOT, SYSTEM_ZEROSHOT_WITH_KNOWLEDGE, SYSTEM_ZEROSHOT_COT,
                              SYSTEM_ZEROSHOT_COT_WITH_KNOWLEDGE,
                              SYSTEM_ZEROSHOT_COT_WITH_KNOWLEDGE_1,
                              SYSTEM_ZEROSHOT_COT_WITH_KNOWLEDGE_2,
                              SYSTEM_FEWSHOT, SYSTEM_FEWSHOT_WITH_KNOWLEDGE, SYSTEM_FEWSHOT_COT)
from src.prompts.llama_prompt import LlamaPrompt, KnowledgePrompt

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
                
        if p_type == "zeroshot_cot_with_knowledge" or p_type == "all":
            for k in top_k_values:
                prompts.append(LlamaPrompt( name=f"zeroshot_cot_with_knowledge_{k}",
                                            system_instruction=SYSTEM_ZEROSHOT_COT_WITH_KNOWLEDGE_1,
                                            sample=sample,
                                            cot=True,
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

def build_prompts_for_retriever_training(sample, ckb_statements, top_k):
    prompts = []

    for k in range(top_k):
        prompts.append(KnowledgePrompt(name=f"zeroshot_with_knowledge_{k}",
                                       system_instruction=SYSTEM_ZEROSHOT_WITH_KNOWLEDGE,
                                       sample=sample,
                                       ckb_statements=ckb_statements[k]
                                       ))
    return prompts

def get_prompt_requirements(prompt_types):
    knowledge = any("knowledge" in s.lower() or "all" in s.lower() for s in prompt_types)
    fewshot = any("fewshot" in s.lower() or "all" in s.lower() for s in prompt_types)
    cot = any("cot" in s.lower() or "all" in s.lower() for s in prompt_types)
    
    return {
        "knowledge": knowledge,
        "fewshot": fewshot,
        "cot": cot
    }

def extend_prompt_with_knowledge(prompt, sample, answer_text, top_k_values):
    extended_prompts = []
    for top_k in top_k_values:
        extended_prompt_k = copy.deepcopy(prompt)
        extended_prompt_k.top_k = top_k
        extended_prompt_k.name = f"{prompt.name}_refine_{top_k}"
        extended_prompt_k.append_messages({"role": "assistant", "content": answer_text})
        ckb_statements = "\n".join(sample["ckb_statements"][:top_k])
        extended_prompt_k.append_messages({"role": "user", "content": f"Given the following knowledge statements, refine your answer.\n\nKnowledge:\n{ckb_statements}"})
        extended_prompts.append(extended_prompt_k)
    return extended_prompts
