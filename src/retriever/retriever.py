# src/retriever/retriever.py

import hashlib
import logging
from pathlib import Path
from typing import List, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, StaticEmbedding

from src.utils.data_utils import synsets_from_samples


class Retriever:
    """
    Dense retriever with optional Maximal Marginal Relevance (MMR) post-filtering.

    Strategies:
      - "retriever"      – index the full CKB once and cache the FAISS file.
      - "cner+retriever" – passages provided later (set_passages); no caching.
    """

    def __init__(
        self,
        model_name_or_path: str,
        retrieval_strategy: str,
        ckb: Union[dict, List[str]],
        passage_prompt: str | None = None,
        query_prompt: str | None = None,
        cache_dir: str = "cache/index",
    ):
        assert retrieval_strategy in {"retriever", "cner+retriever"}
        self.ckb = ckb
        self.model = SentenceTransformer(model_name_or_path)
        self.retrieval_strategy = retrieval_strategy
        self.passage_prompt = passage_prompt
        self.query_prompt = query_prompt

        # Initialize passages and index only for the "retriever" strategy
        self.passages: list[str] = ckb if retrieval_strategy == "retriever" else []
        self.save_index = retrieval_strategy == "retriever"
        self.cache_dir = Path(cache_dir)
        if self.save_index:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = (
            self.cache_dir / f"{self._cache_key()}.faiss" if self.save_index else None
        )
        self.index = self._load_or_build_index() if self.passages else None

    def retrieve_top_k(
        self,
        query: str,
        top_k: int,
        *,
        diversify: bool = False,
        re_rank: str = "mmr",
        lambda_: float = 0.7,
        pool_size: int | None = None,
        diversity_threshold: float = 0.9,
        batch_size: int = 512,
    ) -> List[str]:
        """
        Retrieve top_k passages for a pre-formatted query string.

        If using "cner+retriever", will:
          - extract synsets from `query`
          - collect matching CKB statements
          - call set_passages(...) on that subset
        Then delegates to self.retrieve(...)
        """
        if self.retrieval_strategy == "cner+retriever":
            # 1) extract synsets
            sample_synsets = synsets_from_samples(query)

            # 2) gather all unique statements for those synsets
            ckb_statements = list({
                stmt
                for syn in sample_synsets
                for stmt in self.ckb.get(syn.name(), [])
            })

            # 3) swap in those passages
            self.set_passages(ckb_statements)

        # 4) perform the dense retrieval
        return self.retrieve(
            query,
            top_k,
            diversify=diversify,
            re_rank=re_rank,
            lambda_=lambda_,
            pool_size=pool_size,
            diversity_threshold=diversity_threshold,
            batch_size=batch_size,
        )

    def retrieve(
        self,
        queries: Union[str, List[str]],
        top_k: int,
        *,
        diversify: bool = False,
        re_rank: str = "mmr",
        lambda_: float = 0.7,
        pool_size: int | None = None,
        diversity_threshold: float = 0.9,
        batch_size: int = 512,
    ) -> Union[List[str], List[List[str]]]:
        if self.index is None:
            raise RuntimeError(
                "No FAISS index available. Did you forget to call set_passages?"
            )

        single = isinstance(queries, str)
        queries_list = [queries] if single else queries

        # encode
        q_emb = self._encode(
            queries_list, prompt=self.query_prompt, batch_size=batch_size
        )

        # how many to pull
        cand = pool_size or top_k if diversify else top_k
        cand = min(cand, len(self.passages))

        # FAISS search
        scores, idx = self.index.search(q_emb.astype(np.float32), cand)

        hits: list[list[str]] = []
        for q_i, ids in enumerate(idx):
            cand_texts = [self.passages[i] for i in ids if i != -1]

            if not diversify:
                hits.append(cand_texts[:top_k])
                continue

            # get vectors back
            cand_vecs = np.stack([self.index.reconstruct(int(i)) for i in ids if i != -1])

            if re_rank == "mmr":
                hits.append(_mmr(
                    query_vec=q_emb[q_i],
                    cand_vecs=cand_vecs,
                    cand_texts=cand_texts,
                    k=top_k,
                    lambda_=lambda_,
                ))
            elif re_rank == "filter":
                kept_texts: list[str] = []
                kept_vecs: list[np.ndarray] = []
                for vec, text in zip(cand_vecs, cand_texts):
                    if not kept_vecs:
                        kept_texts.append(text)
                        kept_vecs.append(vec)
                    else:
                        sims = np.dot(np.stack(kept_vecs), vec)
                        if np.all(sims < diversity_threshold):
                            kept_texts.append(text)
                            kept_vecs.append(vec)
                    if len(kept_texts) == top_k:
                        break
                hits.append(kept_texts)
            else:
                raise ValueError(f"Unknown re_rank mode: {re_rank}")

        return hits[0] if single else hits

    def set_passages(self, passages: Union[str, List[str]]):
        self.passages = [passages] if isinstance(passages, str) else passages
        if self.save_index:
            self.index_path = self.cache_dir / f"{self._cache_key()}.faiss"
        self.index = self._load_or_build_index()

    def _encode(self, texts: List[str], *, prompt: str | None, batch_size: int):
        return self.model.encode(
            texts,
            prompt=prompt,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=self._show_progress_bar(texts, batch_size),
        )

    def _load_or_build_index(self):
        if self.save_index and self.index_path.exists():
            logging.info(f"Loading FAISS index from {self.index_path}")
            return faiss.read_index(str(self.index_path))

        logging.info("Building FAISS index …")
        embs = (
            self._encode(self.passages, prompt=self.passage_prompt, batch_size=1024)
            .astype(np.float32)
        )
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        if self.save_index:
            faiss.write_index(index, str(self.index_path))
        return index

    def _cache_key(self) -> str:
        md5 = hashlib.md5()
        md5.update(_get_model_id(self.model).encode())
        for p in self.passages:
            md5.update(p.encode())
        return md5.hexdigest()

    @staticmethod
    def _show_progress_bar(inputs: List[str], batch_size: int) -> bool:
        nb = max(1, len(inputs) // batch_size + (len(inputs) % batch_size > 0))
        return nb > 10


def _get_model_id(model: SentenceTransformer) -> str:
    first = model._first_module()
    if isinstance(first, Transformer):
        return first.auto_model.config.name_or_path
    if isinstance(first, StaticEmbedding):
        return first.base_model
    for m in model._modules.values():
        if hasattr(m, "auto_model"):
            return m.auto_model.config.name_or_path
    raise RuntimeError("Cannot determine model identifier")


def _mmr(
    query_vec: np.ndarray,
    cand_vecs: np.ndarray,
    cand_texts: List[str],
    k: int,
    lambda_: float = 0.7,
) -> List[str]:
    # Ensure lambda_ is a valid trade-off parameter between 0 and 1
    assert 0.0 <= lambda_ <= 1.0

    # List to store selected texts and their corresponding vectors
    selected_texts: List[str] = []
    selected_vecs: List[np.ndarray] = []

    # Compute similarity of each candidate to the query (dot product)
    sim_to_q = cand_vecs @ query_vec

    # Initialize the list of free (not yet selected) indices
    free_idx = np.arange(cand_vecs.shape[0])

    # Iteratively select texts until we reach k or run out of candidates
    while len(selected_texts) < min(k, cand_vecs.shape[0]):
        if selected_vecs:
            # If we have already selected texts, compute diversity penalty
            # For each candidate, find the maximum similarity to any selected vector
            div_penalty = np.max(cand_vecs[free_idx] @ np.stack(selected_vecs, axis=1), axis=1)
        else:
            # No diversity penalty if no texts are selected yet
            div_penalty = 0.0

        # Compute the MMR score: balance between relevance and diversity
        mmr_score = (1 - lambda_) * sim_to_q[free_idx] - lambda_ * div_penalty

        # Select the candidate with the highest MMR score
        best = int(free_idx[np.argmax(mmr_score)])

        # Add the selected candidate to the list of selected vectors and texts
        selected_vecs.append(cand_vecs[best])
        selected_texts.append(cand_texts[best])

        # Remove the selected index from the list of free indices
        free_idx = free_idx[free_idx != best]

    # Return the selected texts
    return selected_texts