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

        self.messages = self._format_input()
        self.text = self._messages_to_text()

    def _format_input(self):
        # 0. System instruction
        messages = []
        messages.append({"role": "system", "content": self.system_instruction})

        # 1. Fewshots
        if self.fewshot_examples:
            for i, example in enumerate(self.fewshot_examples, 1):
                ex_question = example["question"]
                ex_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(example['choices']['label'], example['choices']['text'])])
                ex_answer = example["ground_truth"]
                ex_user_string=""
                ex_assistant_string=""

                if self.cot and self.top_k:
                    ex_ckb_statements = "\n".join(example["ckb_statements"][:self.top_k])
                    ex_reasoning = example["cot"]
                    ex_user_string = f'Example {i}:\n\nQuestion:\n{ex_question}\n\nChoices:\n{ex_choices}\n\nKnowledge:\n{ex_ckb_statements}'
                    ex_assistant_string = f'Reasoning:\n{ex_reasoning}\n\nAnswer: {ex_answer}'

                elif self.cot:
                    ex_reasoning = example["cot"]
                    ex_user_string = f'Example {i}:\n\nQuestion:\n{ex_question}\n\nChoices:\n{ex_choices}'
                    ex_assistant_string = f'Reasoning:\n{ex_reasoning}\n\nAnswer: {ex_answer}'

                elif self.top_k:
                    ex_ckb_statements = "\n".join(example["ckb_statements"][:self.top_k])
                    ex_user_string = f'Example {i}:\n\nQuestion:\n{ex_question}\n\nChoices:\n{ex_choices}\n\nKnowledge:\n{ex_ckb_statements}'
                    ex_assistant_string = f'Answer: {ex_answer}'

                else:
                    ex_user_string = f'Example {i}:\n\nQuestion:\n{ex_question}\n\nChoices:\n{ex_choices}'
                    ex_assistant_string = f'Answer: {ex_answer}'
                    
                messages.append({"role": "user", "content": ex_user_string})
                messages.append({"role": "assistant", "content": ex_assistant_string})

        # 2. Final shot
        question = self.sample["question"]
        choices = "\n".join([f"{label}. {choice}" for label, choice in zip(self.sample['choices']['label'], self.sample['choices']['text'])])

        user_string = f'Question:\n{question}\n\nChoices:\n{choices}'

        if self.top_k:
            ckb_statements = "\n".join(self.sample["ckb_statements"][:self.top_k])
            user_string += f"\n\nKnowledge:\n{ckb_statements}"

        messages.append({"role": "user", "content": user_string})  

        return messages 
    
    def _messages_to_text(self):
        # Prompt plain text only, without system instruction
        return "\n\n".join([m["content"] for m in self.messages if m["role"] != "system"])
    
    def append_messages(self, messages: List[str]):
        messages = [messages] if not isinstance(messages, list) else messages
        for m in messages:
            self.messages.append(m) # Append new messages
        self.text = self._messages_to_text() # Update prompt text

        
class KnowledgePrompt(Prompt):
    def __init__(self,
                 name: str = "knowledge_prompt",
                 system_instruction: str = "",
                 sample: dict = None,
                 ckb_statements: List[str] = None,
                 ):
        self.name = name
        self.system_instruction = system_instruction
        self.sample = sample
        self.ckb_statements = ckb_statements

        self.messages = self._format_input()
        self.text = self._messages_to_text()

    def _format_input(self):
        # 0. System instruction
        messages = []
        messages.append({"role": "system", "content": self.system_instruction})

        # 1. Final shot
        question = self.sample["question"]
        choices = "\n".join([f"{label}. {choice}" for label, choice in zip(self.sample['choices']['label'], self.sample['choices']['text'])])

        user_string = f'Question:\n{question}\n\nChoices:\n{choices}\n\nKnowledge:\n{self.ckb_statements}'

        messages.append({"role": "user", "content": user_string})  

        return messages 
    
    def _messages_to_text(self):
        # Prompt plain text only, without system instruction
        return "\n\n".join([m["content"] for m in self.messages if m["role"] != "system"])
    
    def append_messages(self, messages: List[str]):
        messages = [messages] if not isinstance(messages, list) else messages
        for m in messages:
            self.messages.append(m) # Append new messages
        self.text = self._messages_to_text() # Update prompt text

