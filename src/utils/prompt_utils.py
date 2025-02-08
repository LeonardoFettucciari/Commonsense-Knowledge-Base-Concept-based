from src.prompts.prompts import (template, template_with_knowledge,
                             template_fewshot, template_fewshot_with_knowledge)
from src.prompts.prompts import (SYSTEM_ZERO_SHOT, SYSTEM_ZERO_SHOT_WITH_KNOWLEDGE,
                             SYSTEM_FEW_SHOT, SYSTEM_FEW_SHOT_WITH_KNOWLEDGE)
from settings.constants import TOP_K

def prepare_prompt(sample,
                   examples=[],
                   knowledge=[],
                   examples_knowledge=[],
                   zero_shot=False,
                   few_shot=False,
                   with_knowledge=False,
                   top_k=TOP_K,
                   ):
  # Sample data
  question = sample['question']
  choices = "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])])

  
  # 1/4 Zero-shot with knowledge
  if(zero_shot and with_knowledge):
    knowledge = "\n".join(list(knowledge[:top_k]))
    final_shot = template_with_knowledge.format(
      question=question,
      choices=choices,
      knowledge=knowledge
      )
    
    return (
    [SYSTEM_ZERO_SHOT_WITH_KNOWLEDGE]
    + [final_shot]
    + ["Answer: "]
    )

  # 2/4 Zero-shot
  elif(zero_shot):
    final_shot = template.format(
      question=question,
      choices=choices,
      )
    
    return (
    [SYSTEM_ZERO_SHOT]
    + [final_shot]
    + ["Answer: "]
    )

  # 3/4 Few-shot with knowledge
  elif(few_shot and with_knowledge):
    shots = []
    for i, (example, example_knowledge) in enumerate(zip(examples, examples_knowledge), 1):
      example_question = example['question']
      example_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(example['choices']['label'], example['choices']['text'])])
      example_answer = example['answerKey']
      example_knowledge = "\n".join(list(example_knowledge[:top_k]))

      shot = template_fewshot_with_knowledge.format(
          count=i,
          question=example_question,
          choices=example_choices,
          knowledge=example_knowledge,
          )
      
      shots += (
        [shot]
        + [f"Answer: {example_answer}"]
      )

    knowledge = "\n".join(list(knowledge[:top_k]))
    final_shot = template_with_knowledge.format(
      question=question,
      choices=choices,
      knowledge=knowledge
      )
    
    return (
    [SYSTEM_FEW_SHOT_WITH_KNOWLEDGE]
    + shots
    + [final_shot]
    + ["Answer: "]
    )
  
  # 4/4 Few-shot
  elif(few_shot):
    shots = []
    for i, example in enumerate(examples, 1):
      example_question = example['question']
      example_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(example['choices']['label'], example['choices']['text'])])
      example_answer = example['answerKey']

      shot = template_fewshot.format(
          count=i,
          question=example_question,
          choices=example_choices,
          )
      
      shots += (
        [shot]
        + [f"Answer: {example_answer}"]
      )

    final_shot = template.format(
      question=question,
      choices=choices,
      )
    
    return (
    [SYSTEM_FEW_SHOT]
    + shots
    + [final_shot]
    + ["Answer: "]
    )