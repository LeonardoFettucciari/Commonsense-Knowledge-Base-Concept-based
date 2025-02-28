import torch
import random
from transformers import set_seed
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining, semantic_search

from settings.constants import SEED

# Set seed for repeatable runs
torch.manual_seed(SEED)
random.seed(SEED)
set_seed(SEED)

class Retriever:
    def __init__(self, model_name="intfloat/e5-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.passages_embeddings = None
        self.passages = None

    def initialize(self, passages):
        passages_input = [f"passage: {s}" for s in passages]
        self.passages_embeddings = self.model.encode(passages_input, normalize_embeddings=True)
        self.passages = passages

    def retrieve(self, queries, top_k):
        if self.passages_embeddings is None:
            raise ValueError("Passage embeddings have not been initialized. Call initialize first.")
        
        # Get questions and prepare them for embedding layer
        queries_input = [f"query: {q}" for q in queries]

        # Generate embeddings for questions
        queries_embeddings = self.model.encode(queries_input, normalize_embeddings=True)

        # Compute cosine similarity and extract top-k statements from KB for each question
        
        all_top_k_statements = []
        for qe in queries_embeddings:
            top_k_statements = []
            multiplier = 1
            while len(set(top_k_statements)) < top_k:
                top_k_scores = semantic_search(qe, self.passages_embeddings, top_k=top_k*multiplier)[0]
                top_k_statements = [self.passages[hit['corpus_id']] for hit in top_k_scores]
                multiplier *= 2
            all_top_k_statements.append(list(set(top_k_statements))[:top_k])

        return all_top_k_statements


