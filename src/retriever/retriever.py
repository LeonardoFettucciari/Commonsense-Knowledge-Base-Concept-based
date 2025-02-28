import torch
import random
from datasets import load_dataset
from transformers import set_seed
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining, semantic_search

from settings.constants import SEED


# Set seed for repeatable runs
torch.manual_seed(SEED)
random.seed(SEED)
set_seed(SEED)


passages_embeddings = None

def retriever(queries, passages, top_k, model_name="intfloat/e5-base-v2"):
    global passages_embeddings # Store KB embeddings so not to compute them everytime
    model = SentenceTransformer(model_name)

    # Get questions and prepare them for embedding layer
    queries_input = [f"query: {q}" for q in queries]

    # Get KB statements and prepare them for embedding layer
    passages_input = [f"passage: {s}" for s in passages]


    # Generate embeddings for questions
    queries_embeddings = model.encode(queries_input, normalize_embeddings=True)

    # Generate embeddings for KB (if not already done)
    if passages_embeddings is None:
        passages_embeddings = model.encode(passages_input, normalize_embeddings=True)

    # Compute cosine similarity and extract top-k statements from KB for each question
    all_top_k_scores = semantic_search(queries_embeddings, passages_embeddings, top_k=top_k)
    all_top_k_statements = [[passages[hit['corpus_id']] for hit in top_k_scores] for top_k_scores in all_top_k_scores]

    return all_top_k_statements


