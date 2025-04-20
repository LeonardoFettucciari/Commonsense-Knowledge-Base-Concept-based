import os
import torch
import numpy as np
import logging
import hashlib
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from huggingface_hub import ModelCard
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, StaticEmbedding
UNIQUE_STRING_FOR_HASHING = "Tavshg;re749hgqg7e5g@hgaklsf09786GRE"

class Retriever:

    def __init__(
            self,
            model_name_or_path: str,
            retrieval_strategy: str,
            ckb: dict | list,
            passage_prompt: str | None = None,
            query_prompt: str | None = None,
            cache_dir: str = "cache/embeddings",
    ):
        """
        Retriever class for retrieving statements from a CKB given a query i.e. question+choices.

        :param model_name_or_path: Hugging face model name or local model path.
        :param retrieval_strategy: Retriever or cner+retriever.
        :param ckb: Commonsense Knowledge-Base object.
        :param passage_prompt: Prefix for each passage i.e. passage: <passage>.
        :param query_prompt: Prefix for each query i.e. query: <query>.
        :param cache_dir: Directory for cache.sss
        """
        self.model_name_or_path = model_name_or_path
        self.retrieval_strategy = retrieval_strategy
        self.ckb = ckb
        self.passage_prompt = passage_prompt
        self.query_prompt = query_prompt
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Model
        self.model = SentenceTransformer(self.model_name_or_path)
        
        # Retrieval strategy
        self.save_embeddings = False
        self.passages = []
        if self.retrieval_strategy == "retriever":
            # CKB here is a list of all ckb statements
            self.save_embeddings = True
            self.passages = ckb

        # Passage embeddings + cache
        self.passages_hash = None
        self.passages_embeddings_cache_path = None
        self.passages_embeddings = self._load_or_encode_passages()

    def retrieve(self, queries, top_k, batch_size=512):
        """
        Retrieves top_k statements for each query in queries. Returns a list[str] or str according to queries type. 

        :param queries: list of queries or single query to retriever statements for.
        :param top_k: number of statements to retriever for each query.
        :param batch_size: size of batches.
        """
        is_single_string = isinstance(queries, str)
        queries = [queries] if is_single_string else queries

        question_embeddings = self._encode_query(queries)
        top_k_statements_per_question = semantic_search(
            question_embeddings,
            self.passages_embeddings,
            top_k=top_k,
            query_chunk_size=batch_size
        )
 
        all_questions_statements = [
            [self.passages[statement['corpus_id']] for statement in question_statements]
            for question_statements in top_k_statements_per_question
        ]
        return all_questions_statements[0] if is_single_string else all_questions_statements

    def set_passages(self, passages):
        """
        Sets (or updates) the passages and re-encodes them if necessary.

        :param passages: new passages to set.
        """
        self.passages = [passages] if not isinstance(passages, list) else passages
        self.passages_embeddings = self._load_or_encode_passages()
    
    @staticmethod
    def _compute_hash(passages, unique_string):
        """
        Compute a hash based on all passages and the model name.
        This ensures a unique hash whenever the passages or model differ.
        """
        md5 = hashlib.md5()
        md5.update(unique_string.encode('utf-8'))
        for p in passages:
            md5.update(p.encode('utf-8'))
        return md5.hexdigest()

    def _load_or_encode_passages(self):
        """
        If a cached file of embeddings exists, load it; otherwise encode passages and save.
        """
        self._update_cache_path()
        if os.path.exists(self.passages_embeddings_cache_path):
            logging.info(f"Loading cached embeddings from {self.passages_embeddings_cache_path}")
            return np.load(self.passages_embeddings_cache_path, allow_pickle=False)
        
        logging.info("No cached embeddings found. Encoding passages...")
        passages_embeddings = self._encode_passages()

        if self.save_embeddings:
            logging.info(f"Saving embeddings to {self.passages_embeddings_cache_path}")
            np.save(self.passages_embeddings_cache_path, passages_embeddings)

        return passages_embeddings

    def _encode_passages(self, batch_size=512):
        """
        Encode passages as 'passage_prompt + passage'.
        
        :param batch_size: size of batches.
        """
        encoded_passages = self.model.encode(
            self.passages,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=self._show_progress_bar(self.passages, batch_size),
            convert_to_numpy=True,
            prompt=self.passage_prompt,
        )
        return encoded_passages

    def _encode_query(self, queries, batch_size=512):
        """
        Encode queries as 'query_prompt + query'.

        :param batch_size: size of batches.
        """
        encoded_queries = self.model.encode(
            queries,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=self._show_progress_bar(queries, batch_size),
            convert_to_numpy=True,
            prompt=self.query_prompt,
        )
        return encoded_queries

    @staticmethod
    def _show_progress_bar(inputs: list[str], batch_size: int):
        """
        Show a loading bar only if more than 10 batches.

        :param inputs: input to group in batches of batch_size.
        :param batch_size: size of batches.
        """
        # Compute the number of batches
        num_batches = max(1, len(inputs) // batch_size + (len(inputs) % batch_size > 0))        
        # Show the progress bar only if there are more than 10 batches
        return num_batches > 10
    
    def _update_cache_path(self):
        """
        Update passages embeddings cache path.
        """
        self.passages_hash = self._compute_hash(
            self.passages,
            str(self.model_name_or_path),
        )
        self.passages_embeddings_cache_path = os.path.join(
            self.cache_dir,
            f"{_get_sanitized_model_id(self.model)}_{self.passages_hash}.npy"
        )


def _get_model_id(model: SentenceTransformer) -> str:
    """
    Returns the name_or_path (for HF models) or base_model (for local/static embeddings)
    of a SentenceTransformer, regardless of how it was loaded.
    """
    # grab the very first sub‐module
    first = model._first_module()

    if isinstance(first, Transformer):
        # HF‐style: AutoModel.config.name_or_path is exactly what you passed in
        return first.auto_model.config.name_or_path
    elif isinstance(first, StaticEmbedding):
        # static local embedding: base_model holds the identifier
        return first.base_model
    else:
        # fallback: scan all modules for something with an auto_model
        for m in model._modules.values():
            if hasattr(m, "auto_model"):
                return m.auto_model.config.name_or_path
        raise RuntimeError("Could not determine model identifier")

def _get_sanitized_model_id(model: SentenceTransformer) -> str:
    """
    Like get_model_id, but with '/' and '\' replaced by '_'
    so it’s safe as a filename or key.
    """
    raw = _get_model_id(model)
    # normalize both forward and backward slashes
    return raw.replace("/", "_").replace("\\", "_")
