from typing import List, Union
from abc import ABC, abstractmethod

class Prompt(ABC):
    
    @abstractmethod
    def format_input(
        self, 
        synset_name: str, 
        synset_definition: str, 
    ) -> List[dict]:
        
        raise NotImplementedError

class GeminiPrompt(Prompt):
    def __init__(self, input_data: dict = None):
        self.messages = []
        if input_data:
            self.format_input(input_data)
    
    def format_input(self, input_data: dict):
        synset_lemma = input_data["synset_lemma"]
        synset_definition = input_data["synset_definition"]
        user_string = f'Concept: {synset_lemma}\n\nDefinition: {synset_definition}'
        self.messages.append({"role": "user", "parts": user_string})

class LlamaPrompt(Prompt):
    def __init__(self, input_data: dict = None,
                 zero_shot: bool = False,
                 few_shot: bool = False,
                 knowledge: bool = False,
                 cot: bool = False):
        self.messages = []
        # 1. Zero-shot
        if zero_shot:
            self.zeroshot(input_data)
        # 2. Zero-shot with knowledge
        elif zero_shot and knowledge:
            self.zeroshot_with_knowledge(input_data)
        # 3. Zero-shot CoT
        elif zero_shot and cot: 
            self.zeroshot_cot(input_data)
        # 4. Few-shot
        elif few_shot:
            self.fewshot(input_data)
        # 2. Few-shot with knowledge
        elif few_shot and knowledge:
            self.fewshot_with_knowledge(input_data) 
        # 2. Few-shot CoT
        elif few_shot and cot: 
            self.fewshot_cot(input_data)

    def zeroshot(self, input_data: dict):
        system_string = input_data["system_instruction"]
        self.messages.append({"role": "system", "content": system_string})
        question = input_data["question"]
        choices = input_data["choices"]
        user_string = f'Question:\n{question}\n\nChoices:\n{choices}'
        assistant_string = f'Answer: '
        self.messages.append({"role": "user", "content": user_string})
        self.messages.append({"role": "assistant", "content": assistant_string})

    def zeroshot_with_knowledge(self, input_data: dict):
        system_string = input_data["system_instruction"]
        self.messages.append({"role": "system", "content": system_string})
        question = input_data["question"]
        choices = input_data["choices"]
        knowledge = input_data["knowledge"]
        user_string = f'Question:\n{question}\n\nChoices:\n{choices}\n\nKnowledge:\n{knowledge}'
        assistant_string = f'Answer: '
        self.messages.append({"role": "user", "content": user_string})
        self.messages.append({"role": "assistant", "content": assistant_string})

    def zeroshot_cot(self, input_data: dict):
        system_string = input_data["system_instruction"]
        self.messages.append({"role": "system", "content": system_string})
        question = input_data["question"]
        choices = input_data["choices"]
        user_string = f'Question:\n{question}\n\nChoices:\n{choices}'
        assistant_string = f"Let's think step by step: "
        self.messages.append({"role": "user", "content": user_string})
        self.messages.append({"role": "assistant", "content": assistant_string})

    def fewshot(self, input_data: dict):
        system_string = input_data["system_instruction"]
        self.messages.append({"role": "system", "content": system_string})

        fewshot_examples = input_data["fewshot_examples"]
        for i, example in enumerate(fewshot_examples):
            ex_question = example["question"]
            ex_choices = example["choices"]
            ex_answer = example["answerKey"]
            ex_user_string = f'Example {i}:\nQuestion:\n{ex_question}\n\nChoices:\n{ex_choices}'
            ex_assistant_string = f'Answer: {ex_answer}'
            self.messages.append({"role": "user", "content": ex_user_string})
            self.messages.append({"role": "assistant", "content": ex_assistant_string})

        question = input_data["question"]
        choices = input_data["choices"]
        user_string = f'Question:\n{question}\n\nChoices:\n{choices}'
        assistant_string = f"Answer: "
        self.messages.append({"role": "user", "content": user_string})
        self.messages.append({"role": "assistant", "content": assistant_string})

    def fewshot_with_knowledge(self, input_data: dict):
        system_string = input_data["system_instruction"]
        self.messages.append({"role": "system", "content": system_string})

        fewshot_examples = input_data["fewshot_examples"]
        for i, example in enumerate(fewshot_examples):
            ex_question = example["question"]
            ex_choices = example["choices"]
            ex_knowledge = example["knowledge"]
            ex_answer = example["answerKey"]
            ex_user_string = f'Example {i}:\nQuestion:\n{ex_question}\n\nChoices:\n{ex_choices}\n\nKnowledge:\n{ex_knowledge}'
            ex_assistant_string = f'Answer: {ex_answer}'
            self.messages.append({"role": "user", "content": ex_user_string})
            self.messages.append({"role": "assistant", "content": ex_assistant_string})

        question = input_data["question"]
        choices = input_data["choices"]
        knowledge = input_data["knowledge"]
        user_string = f'Question:\n{question}\n\nChoices:\n{choices}\n\nKnowledge:\n{knowledge}'
        assistant_string = f'Answer: '
        self.messages.append({"role": "user", "content": user_string})
        self.messages.append({"role": "assistant", "content": assistant_string})

    def fewshot_cot(self, input_data: dict):
        system_string = input_data["system_instruction"]
        self.messages.append({"role": "system", "content": system_string})

        fewshot_examples = input_data["fewshot_examples"]
        for i, example in enumerate(fewshot_examples):
            ex_question = example["question"]
            ex_choices = example["choices"]
            ex_reasoning = example["reasoning"]
            ex_answer = example["answerKey"]
            ex_user_string = f'Example {i}:\nQuestion:\n{ex_question}\n\nChoices:\n{ex_choices}'
            ex_assistant_string = f'Reasoning:\n{ex_reasoning}\n\nAnswer: {ex_answer}'
            self.messages.append({"role": "user", "content": ex_user_string})
            self.messages.append({"role": "assistant", "content": ex_assistant_string})

        question = input_data["question"]
        choices = input_data["choices"]
        user_string = f'Question:\n{question}\n\nChoices:\n{choices}'
        assistant_string = f"Reasoning: "
        self.messages.append({"role": "user", "content": user_string})
        self.messages.append({"role": "assistant", "content": assistant_string})