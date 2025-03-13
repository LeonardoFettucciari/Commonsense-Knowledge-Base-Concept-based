import os
import torch
import logging
import hashlib
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

class Retriever:
    def __init__(self, passages, retriever_config, cache_dir="cache", save_embeddings=True):
        """
        :param passages: List of passages (strings)
        :param retriever_config: Dictionary with retriever settings (e.g. {"model_name": ...})
        :param cache_dir: Directory to store/read cached embeddings
        """
        self.retriever_config = retriever_config
        self.model_name = self.retriever_config["model_name"]
        self.cache_dir = cache_dir
        
        logging.info(f"Initializing Retriever with model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        # Compute a hash of the passages + model name for uniqueness
        self.passages_hash = self._compute_hash(passages, self.model_name)
        # Embeddings file path
        self.embeddings_file = os.path.join(self.cache_dir, f"embeddings_{self.model_name.replace('/', '_')}_{self.passages_hash}.pt")
        
        self.passages = passages
        self.passages_embeddings = self._load_or_compute_embeddings() if save_embeddings else self.compute_embeddings()

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
            embeddings = torch.load(self.embeddings_file)
        else:
            logging.info("No cached embeddings found. Encoding passages...")
            embeddings = self._encode_passages()
            logging.info(f"Saving embeddings to {self.embeddings_file}")
            torch.save(embeddings, self.embeddings_file)
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
        logging.info("Encoding passages...")
        if self.model_name == "intfloat/e5-base-v2":
            passages_input = [f"passage: {s}" for s in self.passages]
            embeddings = self.model.encode(passages_input, normalize_embeddings=True, batch_size=batch_size)
            logging.info(f"Encoded {len(self.passages)} passages.")
            return embeddings
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
        
    def _encode_query(self, queries, batch_size=512):
        logging.debug(f"Encoding queries.")
        if self.model_name == "intfloat/e5-base-v2":
            queries_input = [f"query: {q}" for q in queries]
            return self.model.encode(queries_input, normalize_embeddings=True, batch_size=batch_size)
        else:
            raise ValueError(f"Model {self.model_name} not supported.")

    def retrieve(self, queries, top_k, batch_size=512):
        question_embeddings = self._encode_query(queries)
        top_k_statements_batches = semantic_search(question_embeddings, self.passages_embeddings, top_k=top_k, query_chunk_size=batch_size)
        all_batches_statements = [
            [self.passages[statement['corpus_id']] for statement in batch]
            for batch in top_k_statements_batches
        ]
        return all_batches_statements

    def add_ckb_statements_to_samples(self, samples, ckb_statements_list):
        if len(samples) != len(ckb_statements_list):
            logging.warning(f"Mismatch: {len(samples)} samples but {len(ckb_statements_list)} ckb_statements.")
        for s, statements in zip(samples, ckb_statements_list):
            s["ckb_statements"] = statements or []