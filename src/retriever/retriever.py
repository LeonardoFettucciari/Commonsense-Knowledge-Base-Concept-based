import hashlib
import logging
from pathlib import Path
from typing import List, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, StaticEmbedding

from src.utils.model_utils import get_ner_pipeline
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
        diversify: bool = False,
        re_rank: str = "mmr",
        lambda_: float = 0.7,
        pool_size: int | None = None,
        diversity_threshold: float = 0.9,
        batch_size: int = 512,
    ) -> Union[List[str], List[List[str]]]:
        """
        Retrieve top_k passages for either a single query (str) or a batch of queries (List[str]).
        If using "cner+retriever" strategy, extracts synsets per query, subsets the CKB,
        rebuilds passages, then delegates to self.retrieve().
        """
        single = isinstance(queries, str)
        queries_list: List[str] = [queries] if single else queries  # type: ignore

        all_results: List[Union[str, List[str]]] = []

        for query in queries_list:
            if self.retrieval_strategy == "cner+retriever":
                # 1) extract synsets for this query
                sample_synsets = synsets_from_samples(query, self.ner_pipeline)

                # 2) gather all unique statements for those synsets (default to empty list)
                ckb_statements = list({
                    stmt
                    for syn in sample_synsets
                    for stmt in self.ckb.get(syn.name(), [])
                })

                # 3) reset passages to only those CKB statements
                self.set_passages(ckb_statements)

            # 4) perform the (dense) retrieval for this single query
            result = self.retrieve(
                query,
                top_k,
                diversify=diversify,
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
        diversify: bool = False,
        re_rank: str = "mmr",
        lambda_: float = 0.7,
        pool_size: int | None = None,
        diversity_threshold: float = 0.9,
        batch_size: int = 512,
    ) -> Union[List[str], List[List[str]]]:
        single = isinstance(queries, str)
        queries_list = [queries] if single else queries

        # if we've got no passages at all, just return empties
        if not self.passages:
            return [] if single else [[] for _ in queries_list]
    
        # encode query embeddings
        q_emb = self._encode(
            queries_list,
            prompt=self.query_prompt,
            batch_size=batch_size
        ).astype(np.float32)

        # ---- In-memory brute-force path ----
        if self._inmem_embs is not None:
            # (n_passages, dim) @ (dim, n_queries) -> (n_passages, n_queries)
            sims = self._inmem_embs @ q_emb.T
            results: list[list[str]] = []
            for qi in range(sims.shape[1]):
                top_ix = np.argsort(-sims[:, qi])[:top_k]
                results.append([self.passages[i] for i in top_ix])
            return results[0] if single else results

        # ---- FAISS path ----
        if self.index is None:
            raise RuntimeError(
                "No FAISS index available. Did you forget to call set_passages?"
            )

        # how many to pull for MMR pool
        cand = pool_size or top_k if diversify else top_k
        cand = min(cand, len(self.passages))

        scores, idx = self.index.search(q_emb, cand)
        hits: list[list[str]] = []
        for q_i, ids in enumerate(idx):
            cand_texts = [self.passages[i] for i in ids if i != -1]
            if not diversify:
                hits.append(cand_texts[:top_k])
                continue

            # get vectors back for MMR/filter
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

    selected_texts: List[str] = []
    selected_vecs: List[np.ndarray] = []

    sim_to_q = cand_vecs @ query_vec
    free_idx = np.arange(cand_vecs.shape[0])

    while len(selected_texts) < min(k, cand_vecs.shape[0]):
        if selected_vecs:
            div_penalty = np.max(cand_vecs[free_idx] @ np.stack(selected_vecs, axis=1), axis=1)
        else:
            div_penalty = 0.0

        mmr_score = (1 - lambda_) * sim_to_q[free_idx] - lambda_ * div_penalty
        best = int(free_idx[np.argmax(mmr_score)])

        selected_vecs.append(cand_vecs[best])
        selected_texts.append(cand_texts[best])
        free_idx = free_idx[free_idx != best]

    return selected_texts
