import json
from typing import List, Union
from abc import ABC, abstractmethod

class Prompt(ABC):
    
    @abstractmethod
    def zeroshot(
        self, 
        input_data: dict, 
    ) -> List[dict]:
        
        raise NotImplementedError

class LlamaPrompt(Prompt):
    def __init__(self, input_data: dict = None,
                 zero_shot: bool = False,
                 few_shot: bool = False,
                 knowledge: bool = False,
                 cot: bool = False):
        self.messages = []
        self.name = ""
        self.cot = None

        # 1. Zero-shot with knowledge
        if zero_shot and knowledge:
            self.zeroshot_with_knowledge(input_data)
        # 2. Zero-shot CoT
        elif zero_shot and cot: 
            self.zeroshot_cot(input_data)
        # 3. Zero-shot
        elif zero_shot:
            self.zeroshot(input_data)
        # 4. Few-shot with knowledge
        elif few_shot and knowledge:
            self.fewshot_with_knowledge(input_data) 
        # 5. Few-shot CoT
        elif few_shot and cot: 
            self.fewshot_cot(input_data)
        # 6. Few-shot
        elif few_shot:
            self.fewshot(input_data)

    def zeroshot(self, input_data: dict):
        self.name = input_data.get("prompt_name", "zeroshot")
        system_string = input_data["system_instruction"]
        self.messages.append({"role": "system", "content": system_string})
        question = input_data["question"]
        choices = input_data["choices"]
        user_string = f'Question:\n{question}\n\nChoices:\n{choices}'
        assistant_string = f'Answer: '
        self.messages.append({"role": "user", "content": user_string})
        #self.messages.append({"role": "assistant", "content": assistant_string})
        

    def zeroshot_with_knowledge(self, input_data: dict):
        self.name = input_data.get("prompt_name", "zeroshot_with_knowledge")
        system_string = input_data["system_instruction"]
        self.messages.append({"role": "system", "content": system_string})
        question = input_data["question"]
        choices = input_data["choices"]
        knowledge = "\n".join(input_data["knowledge"])
        user_string = f'Question:\n{question}\n\nChoices:\n{choices}\n\nKnowledge:\n{knowledge}'
        assistant_string = f'Answer: '
        self.messages.append({"role": "user", "content": user_string})
        #self.messages.append({"role": "assistant", "content": assistant_string})
        
    def zeroshot_cot(self, input_data: dict):
        self.name = input_data.get("prompt_name", "zeroshot_cot")
        self.cot = True
        system_string = input_data["system_instruction"]
        self.messages.append({"role": "system", "content": system_string})
        question = input_data["question"]
        choices = input_data["choices"]
        user_string = f'Question:\n{question}\n\nChoices:\n{choices}'
        assistant_string = f"Let's think step by step: "
        self.messages.append({"role": "user", "content": user_string})
        #self.messages.append({"role": "assistant", "content": assistant_string})
        

    def fewshot(self, input_data: dict):
        self.name = input_data.get("prompt_name", "fewshot")
        system_string = input_data["system_instruction"]
        self.messages.append({"role": "system", "content": system_string})

        fewshot_examples = input_data["fewshot_examples"]
        for i, example in enumerate(fewshot_examples, 1):
            ex_question = example["question_stem"]
            choices_dict = json.loads(example['choices'])
            ex_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(choices_dict['label'], choices_dict['text'])])
            ex_answer = example["answerKey"]
            ex_user_string = f'Example {i}:\n\nQuestion:\n{ex_question}\n\nChoices:\n{ex_choices}'
            ex_assistant_string = f'Answer: {ex_answer}'
            self.messages.append({"role": "user", "content": ex_user_string})
            self.messages.append({"role": "assistant", "content": ex_assistant_string})

        question = input_data["question"]
        choices = input_data["choices"]
        user_string = f'Question:\n{question}\n\nChoices:\n{choices}'
        assistant_string = f"Answer: "
        self.messages.append({"role": "user", "content": user_string})
        #self.messages.append({"role": "assistant", "content": assistant_string})
        

    def fewshot_with_knowledge(self, input_data: dict):
        self.name = input_data.get("prompt_name", "fewshot_with_knowledge")
        system_string = input_data["system_instruction"]
        self.messages.append({"role": "system", "content": system_string})

        fewshot_examples = input_data["fewshot_examples"]
        fewshot_knowledge = input_data["fewshot_knowledge"]
        
        for i, (example, ex_knowledge) in enumerate(zip(fewshot_examples, fewshot_knowledge), 1):
            ex_knowledge = "\n".join(ex_knowledge)
            ex_question = example["question_stem"]
            choices_dict = json.loads(example['choices'])
            ex_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(choices_dict['label'], choices_dict['text'])])
            ex_answer = example["answerKey"]
            ex_user_string = f'Example {i}:\n\nQuestion:\n{ex_question}\n\nChoices:\n{ex_choices}\n\nKnowledge:\n{ex_knowledge}'
            ex_assistant_string = f'Answer: {ex_answer}'
            self.messages.append({"role": "user", "content": ex_user_string})
            self.messages.append({"role": "assistant", "content": ex_assistant_string})

        question = input_data["question"]
        choices = input_data["choices"]
        knowledge = "\n".join(input_data["knowledge"])
        user_string = f'Question:\n{question}\n\nChoices:\n{choices}\n\nKnowledge:\n{knowledge}'
        assistant_string = f'Answer: '
        self.messages.append({"role": "user", "content": user_string})
        #self.messages.append({"role": "assistant", "content": assistant_string})
        
    def fewshot_cot(self, input_data: dict):
        self.name = input_data.get("prompt_name", "fewshot_cot")
        self.cot = True
        system_string = input_data["system_instruction"]
        self.messages.append({"role": "system", "content": system_string})

        fewshot_examples = input_data["fewshot_examples"]
        for i, example in enumerate(fewshot_examples, 1):
            ex_question = example["question_stem"]
            choices_dict = json.loads(example['choices'])
            ex_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(choices_dict['label'], choices_dict['text'])])
            ex_reasoning = example["cot"]
            ex_answer = example["answerKey"]
            ex_user_string = f'Example {i}:\n\nQuestion:\n{ex_question}\n\nChoices:\n{ex_choices}'
            ex_assistant_string = f'Reasoning:\n{ex_reasoning}\n\nAnswer: {ex_answer}'
            self.messages.append({"role": "user", "content": ex_user_string})
            self.messages.append({"role": "assistant", "content": ex_assistant_string})

        question = input_data["question"]
        choices = input_data["choices"]
        user_string = f'Question:\n{question}\n\nChoices:\n{choices}'
        assistant_string = f"Reasoning: "
        self.messages.append({"role": "user", "content": user_string})
        #self.messages.append({"role": "assistant", "content": assistant_string})
        