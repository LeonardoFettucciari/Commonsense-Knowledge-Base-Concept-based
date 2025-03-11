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