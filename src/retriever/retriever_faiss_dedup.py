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
    Dense retriever with FAISS-based semantic deduplication.

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

        self.ner_pipeline = get_ner_pipeline("Babelscape/cner-base")

    def retrieve_top_k(
        self,
        query: str,
        top_k: int,
    ) -> List[str]:
        """
        Retrieve top_k passages with FAISS-based deduplication.
        """
        if self.retrieval_strategy == "cner+retriever":
            # extract synsets and gather matching statements
            sample_synsets = synsets_from_samples(query, self.ner_pipeline)
            ckb_statements = list({
                stmt
                for syn in sample_synsets
                for stmt in self.ckb.get(syn.name(), [])
            })
            self.set_passages(ckb_statements)

        return self.retrieve(query, top_k)

    def retrieve(
        self,
        queries: Union[str, List[str]],
        top_k: int,
        *,
        dedupe_threshold: float = 0.85,
        dedupe_nn_k: int = 5,
        batch_size: int = 512,
    ) -> Union[List[str], List[List[str]]]:
        if self.index is None:
            raise RuntimeError(
                "No FAISS index available. Did you forget to call set_passages?"
            )

        single = isinstance(queries, str)
        queries_list = [queries] if single else queries

        # encode queries
        q_emb = self._encode(
            queries_list, prompt=self.query_prompt, batch_size=batch_size
        )

        # retrieve raw candidates
        scores, idx = self.index.search(q_emb.astype(np.float32), top_k)

        hits: list[list[str]] = []
        for score_row, ids_row in zip(scores, idx):
            valid = [i for i in ids_row if i != -1]
            cand_texts = [self.passages[i] for i in valid]
            cand_vecs = np.stack([self.index.reconstruct(int(i)) for i in valid])
            cand_scores = score_row[: len(valid)]

            kept, dropped = self._faiss_dedupe(
                cand_texts, cand_vecs, cand_scores, top_k,
                threshold=dedupe_threshold, nn_k=dedupe_nn_k,
            )

            hits.append({
                "kept": kept,
                "dropped": dropped
            })

        return hits[0] if single else hits

    def _faiss_dedupe(
        self,
        cand_texts: List[str],
        cand_vecs: np.ndarray,
        cand_scores: np.ndarray,
        top_k: int,
        threshold: float,
        nn_k: int,
    ) -> tuple[list[str], list[str]]:
        """
        Returns (kept_texts, filtered_out_texts).
        """
        dim = cand_vecs.shape[1]
        local_index = faiss.IndexFlatIP(dim)
        local_index.add(cand_vecs)

        D, I = local_index.search(cand_vecs, nn_k)

        to_skip = set()
        for i, (neighbors, sims) in enumerate(zip(I, D)):
            if i in to_skip:
                continue
            for j, sim in zip(neighbors, sims):
                if i != j and sim >= threshold:
                    # drop the lower‐scored one
                    if cand_scores[i] >= cand_scores[j]:
                        to_skip.add(j)
                    else:
                        to_skip.add(i)

        kept = []
        dropped = []
        for idx, text in enumerate(cand_texts):
            if idx in to_skip:
                dropped.append(text)
            else:
                kept.append(text)
            if len(kept) >= top_k:
                break

        return kept, dropped


    def set_passages(self, passages: Union[str, List[str]]):
        self.passages = [passages] if isinstance(passages, str) else passages
        if self.save_index:
            self.index_path = self.cache_dir / f"{self._cache_key()}.faiss"
        self.index = self._load_or_build_index()

    def _encode(
        self,
        texts: List[str],
        *,
        prompt: str | None,
        batch_size: int,
    ):
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