import json
from typing import List, Union
from abc import ABC, abstractmethod

class Prompt(ABC):
    
    @abstractmethod
    def _format_input(
        self, 
        input_data: dict, 
    ) -> List[dict]:
        
        raise NotImplementedError

class LlamaPrompt(Prompt):
    def __init__(self,
                 name: str = "prompt",
                 system_instruction: str = "",
                 sample: dict = None,
                 fewshot_examples: dict = None,
                 top_k: int = None,
                 cot: bool = False,
                 ):
        self.name = name
        self.system_instruction = system_instruction
        self.sample = sample
        self.fewshot_examples = fewshot_examples
        self.top_k = top_k
        self.cot = cot

        self.messages = []

        self._format_input()

    def _format_input(self):
        # 0. System instruction
        self.messages.append({"role": "system", "content": self.system_instruction})

        # 1. Fewshots
        if self.fewshot_examples:
            for i, example in enumerate(self.fewshot_examples, 1):
                ex_question = example["question"]
                ex_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(example['choices']['label'], example['choices']['text'])])
                ex_answer = example["answerKey"]
                ex_user_string=""
                ex_assistant_string=""

                if self.cot:
                    ex_reasoning = example["cot"]
                    ex_user_string = f'Example {i}:\n\nQuestion:\n{ex_question}\n\nChoices:\n{ex_choices}'
                    ex_assistant_string = f'Reasoning:\n{ex_reasoning}\n\nAnswer: {ex_answer}'

                elif self.top_k:
                    ex_kb_statements = "\n".join(example["kb_statements"][:self.top_k])
                    ex_user_string = f'Example {i}:\n\nQuestion:\n{ex_question}\n\nChoices:\n{ex_choices}\n\nKnowledge:\n{ex_kb_statements}'
                    ex_assistant_string = f'Answer: {ex_answer}'

                else:
                    ex_user_string = f'Example {i}:\n\nQuestion:\n{ex_question}\n\nChoices:\n{ex_choices}'
                    ex_assistant_string = f'Answer: {ex_answer}'
                    
                self.messages.append({"role": "user", "content": ex_user_string})
                self.messages.append({"role": "assistant", "content": ex_assistant_string})

        # 2. Final shot
        question = self.sample["question"]
        choices = "\n".join([f"{label}. {choice}" for label, choice in zip(self.sample['choices']['label'], self.sample['choices']['text'])])

        if self.cot:
            user_string = f'Question:\n{question}\n\nChoices:\n{choices}'
        elif self.top_k:
            kb_statements = "\n".join(self.sample["kb_statements"][:self.top_k])
            user_string = f'Question:\n{question}\n\nChoices:\n{choices}\n\nKnowledge:\n{kb_statements}'
        else:
            user_string = f'Question:\n{question}\n\nChoices:\n{choices}'

        self.messages.append({"role": "user", "content": user_string})   