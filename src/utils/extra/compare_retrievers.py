from datasets import load_dataset

from src.retriever.retriever import Retriever
from src.utils.io_utils import load_ckb
from src.utils.retriever_utils import retrieve_top_k_statements
from src.utils.data_utils import concatenate_question_choices
from src.datasets.dataset_loader import load_hf_dataset, load_local_dataset, preprocess_dataset



ckb_path = "data/ckb/cleaned/full_ckb.jsonl"
retrieval_strategy = "retriever"


# Load dataset
eval_dataset = load_dataset("allenai/openbookqa")['train']
eval_dataset = preprocess_dataset(eval_dataset, "obqa")

sample = eval_dataset.filter(lambda example: example['id'] == "8-72")
formatted_question = concatenate_question_choices(sample[0])

# Load knowledge base
ckb = load_ckb(ckb_path, retrieval_strategy)

# Initialize retriever trained
retriever_trained = Retriever(
    "models/retriever_mnr/final",
    retrieval_strategy,
    ckb,
    passage_prompt="passage: ",
    query_prompt="query: ",
)
statements_trained = retriever_trained.retrieve(formatted_question, 5)
statements_trained = "\n".join(statements_trained)

# Initialize retriever
retriever = Retriever(
    "intfloat/e5-base-v2",
    retrieval_strategy,
    ckb,
    passage_prompt="passage: ",
    query_prompt="query: ",
)
statements = retriever.retrieve(formatted_question, 5)
statements = "\n".join(statements)

print(f"QUERY:\n{formatted_question}")
print(f"CLASSIC RETRIEVER:\n{statements}")
print(f"TRAINED RETRIEVER:\n{statements_trained}")

