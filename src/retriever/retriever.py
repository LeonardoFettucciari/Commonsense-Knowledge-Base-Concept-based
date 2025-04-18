import os
import torch
import numpy as np
import logging
import hashlib
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

class Retriever:

    def __init__(self, retrieval_strategy, ckb, config, cache_dir="cache"):
        """
        :param passages: List of passages (strings)
        :param config: Dictionary with retriever settings (e.g. {"model_name": ...})
        :param cache_dir: Directory to store/read cached embeddings
        """
        self.retrieval_strategy = retrieval_strategy
        self.ckb = ckb
        self.config = config
        self.cache_dir = cache_dir
        
        self.model_name = self.config["model_name"]
        logging.info(f"Initializing Retriever with model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
                
        if self.retrieval_strategy == "retriever":
            self.save_embeddings = True
            # CKB here is a list of all ckb statements
            self.passages = ckb

        elif self.retrieval_strategy == "cner+retriever":
            self.save_embeddings = False
            # We won't set passages right now, because we do that per-sample
            # CKB here is a dict synset:statements
            self.passages = []
            
        else:
            self.passages = []
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        # Compute a hash of the passages + model name for uniqueness
        self.passages_hash = self._compute_hash(self.passages, self.model_name)
        # Embeddings file path
        self.embeddings_file = os.path.join(self.cache_dir, f"embeddings_{self.model_name.replace('/', '_')}_{self.passages_hash}.npy")
        self.passages_embeddings = self._load_or_compute_embeddings() if self.save_embeddings else self.compute_embeddings()


    @staticmethod
    def _compute_hash(passages, model_name):
        """
        Compute a hash based on all passages and the model name.
        This ensures a unique hash whenever the passages or model differ.
        """
        md5 = hashlib.md5()
        md5.update(model_name.encode('utf-8'))
        for p in passages:
            md5.update(p.encode('utf-8'))
        return md5.hexdigest()

    def _load_or_compute_embeddings(self):
        """
        If a cached file of embeddings exists, load it; otherwise encode passages and save.
        """
        if os.path.exists(self.embeddings_file):
            logging.info(f"Loading cached embeddings from {self.embeddings_file}")
            embeddings = np.load(self.embeddings_file, allow_pickle=False)
        else:
            logging.info("No cached embeddings found. Encoding passages...")
            embeddings = self._encode_passages()
            logging.info(f"Saving embeddings to {self.embeddings_file}")
            np.save(self.embeddings_file, embeddings)
        return embeddings
    
    def compute_embeddings(self):
        """
        Encode passages.
        """
        logging.info("Encoding passages...")
        embeddings = self._encode_passages()
        logging.info(f"Passages encoded.")
        return embeddings

    def _encode_passages(self, batch_size=512):

        if self.model_name == "intfloat/e5-base-v2":
            # Compute the number of batches
            num_batches = max(1, len(self.passages) // batch_size + (len(self.passages) % batch_size > 0))

            passages_input = [f"passage: {s}" for s in self.passages]
            
            # Show the progress bar only if there are more than 10 batches
            show_pbar = num_batches > 10

            encoded_passages = self.model.encode(
                passages_input,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=show_pbar,
                convert_to_numpy=True,
            )
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
        
        return encoded_passages

        
    def _encode_query(self, queries, batch_size=512):
        encoded_queries = []
        if self.model_name == "intfloat/e5-base-v2":
            # Compute the number of batches
            num_batches = max(1, len(queries) // batch_size + (len(queries) % batch_size > 0))

            queries_input = [f"query: {q}" for q in queries]
            
            # Show the progress bar only if there are more than 10 batches
            show_pbar = num_batches > 10

            encoded_queries = self.model.encode(
                queries_input,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=show_pbar,
            )
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
        
        return encoded_queries


    def retrieve(self, queries, top_k, batch_size=512):
        queries = [queries] if not isinstance(queries, list) else queries

        question_embeddings = self._encode_query(queries)
        top_k_statements_per_question = semantic_search(question_embeddings, self.passages_embeddings, top_k=top_k, query_chunk_size=batch_size)
 
        all_questions_statements = [
            [self.passages[statement['corpus_id']] for statement in question_statements]
            for question_statements in top_k_statements_per_question
        ]
        return all_questions_statements

    def set_passages(self, passages):
        """
        Sets (or updates) the passages and re-encodes them if necessary.
        """
        self.passages = [passages] if not isinstance(passages, list) else passages
        # If you want to cache embeddings for these passages:
        if self.save_embeddings:
            self.passages_embeddings = self._load_or_compute_embeddings()
        else:
            self.passages_embeddings = self._encode_passages()