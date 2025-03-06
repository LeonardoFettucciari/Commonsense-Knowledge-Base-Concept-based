import torch
import random
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining, semantic_search


class Retriever:
    def __init__(self, passages, retriever_config):
        self.rc = retriever_config
        self.model_name = self.rc["model_name"]
        self.model = SentenceTransformer(self.model_name)
        self.passages = passages
        self.passages_embeddings = self._encode_passages()

    def _encode_passages(self):
        if(self.model_name == "intfloat/e5-base-v2"):
            passages_input = [f"passage: {s}" for s in self.passages]
            return self.model.encode(passages_input, normalize_embeddings=True)
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
        
    def _encode_query(self, query):
        if(self.model_name == "intfloat/e5-base-v2"):
            query_input = [f"query: {query}"]
            return self.model.encode(query_input, normalize_embeddings=True)
        else:
            raise ValueError(f"Model {self.model_name} not supported.")

    def retrieve(self, query, top_k):
        qe = self._encode_query(query)
              
        # Compute cosine similarity and extract top-k statements from KB for each question
        hits_per_iteration = top_k  
        retrieved_statements = set()
        while len(retrieved_statements) < top_k:
            hits = semantic_search(qe, self.passages_embeddings, top_k=hits_per_iteration)[0]
            for hit in hits:
                retrieved_statements.add(self.passages[hit['corpus_id']])
                if(len(retrieved_statements) == top_k):
                    break
            hits_per_iteration *= 2 # Increase the number of hits to retrieve if top-k statements are not unique
        
        return list(retrieved_statements)
    
    def retrieve_all(self, queries, top_k):
        return [self.retrieve(query, top_k) for query in queries]



