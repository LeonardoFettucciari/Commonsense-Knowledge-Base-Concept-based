# 1
template = '''
Question:
{question}

Choices:
{choices}
'''

# 2
template_with_knowledge = '''
Question:
{question}

Choices:
{choices}

Knowledge:
{knowledge}
'''

# 3
template_fewshot = '''
Example {count}:

Question:
{question}

Choices:
{choices}
'''

# 4
template_fewshot_with_knowledge = '''
Example {count}:

Question:
{question}

Choices:
{choices}

Knowledge:
{knowledge}
'''


# System prompts
SYSTEM_ZERO_SHOT = '''
You are an expert in commonsense question answering. \
Given a question, along with its choices, \
provide only the label of the correct answer. \
'''

SYSTEM_ZERO_SHOT_WITH_KNOWLEDGE = '''
You are an expert in commonsense question answering. \
Given a question, along with its choices and a list of useful commonsense statements, \
provide only the label of the correct answer. \
'''

SYSTEM_FEW_SHOT = '''
You are an expert in commonsense question answering. \
Given a question, along with its choices, \
provide only the label of the correct answer. \
'''

SYSTEM_FEW_SHOT_WITH_KNOWLEDGE = '''
You are an expert in commonsense question answering. \
Given a question, along with its choices and a list of useful commonsense statements, \
provide only the label of the correct answer. \
'''
