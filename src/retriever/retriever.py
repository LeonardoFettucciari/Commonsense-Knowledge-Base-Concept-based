import hashlib
import logging
from pathlib import Path
from typing import List, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, StaticEmbedding

from src.utils.model_utils import get_ner_pipeline
from src.utils.data_utils import synsets_from_batch


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
        self.model_name_or_path = model_name_or_path
        assert retrieval_strategy in {"retriever", "cner+retriever"}
        self.ckb = ckb
        self.model = SentenceTransformer(model_name_or_path)
        self.retrieval_strategy = retrieval_strategy
        self.passage_prompt = passage_prompt
        self.query_prompt = query_prompt

        # For "retriever", we'll maintain a FAISS index on disk;
        # for small-scale "cner+retriever", we'll do in-memory brute force.
        self.save_index = retrieval_strategy == "retriever"
        self.cache_dir = Path(cache_dir)

        # Only build index for full-retriever immediately
        if self.save_index:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.passages = ckb
            self.index_path = (self.cache_dir / f"{self._cache_key()}.faiss")
            self.index = self._load_or_build_index()
            self._inmem_embs: np.ndarray | None = None
            
        else:
            self.passages = []
            self.index_path = None
            self.index = None
            self._inmem_embs: np.ndarray | None = None

        self.ner_pipeline = get_ner_pipeline("Babelscape/cner-base")

    from typing import Union, List



    def retrieve_top_k(
        self,
        queries: Union[str, List[str]],
        top_k: int,
        *,
        re_rank: str = None,
        lambda_: float = 0.7,
        pool_size: int | None = None,
        diversity_threshold: float = 0.9,
        batch_size: int = 512,
    ) -> Union[List[str], List[List[str]]]:
        """
        Retrieve top_k passages for either a single query (str) or a batch of queries (List[str]).
        If using "cner+retriever" strategy, extracts synsets per query in batch by calling
        the NER pipeline once on the full list, subsets the CKB, rebuilds passages, then delegates
        to self.retrieve().
        """
        single = isinstance(queries, str)
        queries_list: List[str] = [queries] if single else queries  # type: ignore

        # Pre-allocate batch_synsets
        batch_synsets: List[List] = []
        if self.retrieval_strategy == "cner+retriever" and queries_list:
            # Batch-extract synsets once via raw Python list batching
            from src.utils.data_utils import synsets_from_batch
            batch_synsets = synsets_from_batch(
                samples=queries_list,
                ner_pipeline=self.ner_pipeline,
                batch_size=32,
            )

        all_results: List[Union[str, List[str]]] = []

        for idx, query in enumerate(queries_list):
            # For cner+retriever, use precomputed synsets and filter the CKB
            if self.retrieval_strategy == "cner+retriever":
                sample_synsets = batch_synsets[idx] if batch_synsets else []
                ckb_statements = list({
                    stmt
                    for syn in sample_synsets
                    for stmt in self.ckb.get(syn.name(), [])
                })
                self.set_passages(ckb_statements)

            # Perform the (dense) retrieval for this single query
            result = self.retrieve(
                query,
                top_k,
                re_rank=re_rank,
                lambda_=lambda_,
                pool_size=pool_size,
                diversity_threshold=diversity_threshold,
                batch_size=batch_size,
            )
            all_results.append(result)

        # if single query, return flat list; else return list-of-lists
        return all_results[0] if single else all_results




    def retrieve(
        self,
        queries: Union[str, List[str]],
        top_k: int,
        *,
        re_rank: str = None,  # 'mmr', or 'filter' or None
        lambda_: float = 0.7,
        pool_size: int | None = None,
        diversity_threshold: float = 0.9,
        batch_size: int = 512,
    ) -> Union[List[str], List[List[str]]]:
        single = isinstance(queries, str)
        queries_list = [queries] if single else queries

        if not self.passages:
            return [] if single else [[] for _ in queries_list]

        q_emb = self._encode(
            queries_list,
            prompt=self.query_prompt,
            batch_size=batch_size
        ).astype(np.float32)

        # In-memory retrieval for 'CNER+retriever' strategy
        if self._inmem_embs is not None:
            sims = self._inmem_embs @ q_emb.T
            results = []
            for qi in range(sims.shape[1]):
                top_ix = np.argsort(-sims[:, qi])[:top_k]
                cand_texts = [self.passages[i] for i in top_ix]
                cand_vecs = self._inmem_embs[top_ix]
                results.append(_deduplicate(
                    re_rank=re_rank,
                    query_vec=q_emb[qi],
                    cand_vecs=cand_vecs,
                    cand_texts=cand_texts,
                    top_k=top_k,
                    lambda_=lambda_,
                    diversity_threshold=diversity_threshold,
                ))
            return results[0] if single else results

        # FAISS-based retrieval for 'retriever' strategy
        if self.index is None:
            raise RuntimeError("No FAISS index available. Did you forget to call set_passages?")

        cand = pool_size or top_k if re_rank else top_k
        cand = min(cand, len(self.passages))
        scores, idx = self.index.search(q_emb, cand)

        hits = []
        for q_i, ids in enumerate(idx):
            valid_ids = [i for i in ids if i != -1]
            cand_texts = [self.passages[i] for i in valid_ids]
            cand_vecs = np.stack([self.index.reconstruct(int(i)) for i in valid_ids])


            hits.append(_deduplicate(
                re_rank=re_rank,
                query_vec=q_emb[q_i],
                cand_vecs=cand_vecs,
                cand_texts=cand_texts,
                top_k=top_k,
                lambda_=lambda_,
                diversity_threshold=diversity_threshold,
            ))

        return hits[0] if single else hits


    def set_passages(self, passages: Union[str, List[str]]):
        """
        Replace current passages and rebuild either FAISS index or in-memory embeddings.
        """
        self.passages = [passages] if isinstance(passages, str) else passages
        if self.save_index:
            # full-retriever: build or reload disk-based FAISS index
            self.index_path = self.cache_dir / f"{self._cache_key()}.faiss"
            self.index = self._load_or_build_index()
            self._inmem_embs = None
        else:
            # small-scale: encode all passages in-memory
            self._inmem_embs = self._encode(
                self.passages,
                prompt=self.passage_prompt,
                batch_size=512
            ).astype(np.float32)
            self.index = None

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
        md5.update(self.model_name_or_path.encode())
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
    lambda_: float = 0.8,
) -> List[str]:
    
    # 0 means diversity, 1 means relevance
    assert 0.0 <= lambda_ <= 1.0

    selected_texts = []
    selected_vecs = []

    sim_to_query = cand_vecs @ query_vec
    free_idx = np.arange(len(cand_texts))

    while len(selected_texts) < min(k, len(cand_texts)):
        if selected_vecs:
            div_penalty = np.max(cand_vecs[free_idx] @ np.stack(selected_vecs, axis=1), axis=1)
        else:
            div_penalty = np.zeros(len(free_idx))

        mmr_scores = lambda_ * sim_to_query[free_idx] - (1 - lambda_) * div_penalty
        best_idx = free_idx[np.argmax(mmr_scores)]

        selected_texts.append(cand_texts[best_idx])
        selected_vecs.append(cand_vecs[best_idx])
        free_idx = free_idx[free_idx != best_idx]

    return selected_texts

def _deduplicate(
        re_rank: str,
        query_vec: np.ndarray,
        cand_vecs: np.ndarray,
        cand_texts: List[str],
        top_k: int,
        lambda_: float,
        diversity_threshold: float,
    ) -> List[str]:
        
        if not re_rank:
            return cand_texts[:top_k]

        elif re_rank == "mmr":
            return _mmr(query_vec, cand_vecs, cand_texts, top_k, lambda_)

        elif re_rank == "filter":
            kept_texts = []
            kept_vecs = []
            for vec, text in zip(cand_vecs, cand_texts):
                if not kept_vecs:
                    kept_vecs.append(vec)
                    kept_texts.append(text)
                else:
                    sims = np.dot(np.stack(kept_vecs), vec)
                    if np.max(sims) < diversity_threshold:
                        kept_vecs.append(vec)
                        kept_texts.append(text)
                if len(kept_texts) == top_k:
                    break
            return kept_texts

        else:
            raise ValueError(f"Unknown re_rank method: {re_rank}")
