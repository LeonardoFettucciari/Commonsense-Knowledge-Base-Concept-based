import hashlib
import logging
from pathlib import Path
from typing import List, Union
import faiss
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.model_utils import get_ner_pipeline
from src.utils.data_utils import synsets_from_batch


class Retriever:
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
        self.save_index = retrieval_strategy == "retriever"
        self.cache_dir = Path(cache_dir)

        if self.save_index:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.passages = ckb
            self.index_path = self.cache_dir / f"{self._cache_key()}.faiss"
            self.index = self._load_or_build_index()
            self._inmem_embs: np.ndarray | None = None
        else:
            self.passages = []
            self.index_path = None
            self.index = None
            self._inmem_embs: np.ndarray | None = None

        self.ner_pipeline = get_ner_pipeline("Babelscape/cner-base")

    def retrieve_top_k(
        self,
        queries: Union[str, List[str]],
        top_k: int,
        *,
        re_rank: str = None,
        pool_size: int | None = None,
        diversity_threshold: float = 0.9,
        batch_size: int = 512,
    ) -> Union[List[str], List[List[str]]]:
        single = isinstance(queries, str)
        queries_list = [queries] if single else queries

        # Extracy synsets if using CNER strategy
        batch_synsets: List[List] = []
        if self.retrieval_strategy == "cner+retriever" and queries_list:
            batch_synsets = synsets_from_batch(
                samples=queries_list,
                ner_pipeline=self.ner_pipeline,
                batch_size=32,
            )

        all_results: List[Union[str, List[str]]] = []

        for idx, query in enumerate(queries_list):
            if self.retrieval_strategy == "cner+retriever":
                sample_synsets = batch_synsets[idx] if batch_synsets else []
                ckb_statements = list({
                    stmt
                    for syn in sample_synsets
                    for stmt in self.ckb.get(syn.name(), [])
                })
                self.set_passages(ckb_statements)

            result = self.retrieve(
                query,
                top_k,
                re_rank=re_rank,
                pool_size=pool_size,
                diversity_threshold=diversity_threshold,
                batch_size=batch_size,
            )
            all_results.append(result)

        return all_results[0] if single else all_results

    def retrieve(
        self,
        queries: Union[str, List[str]],
        top_k: int,
        *,
        re_rank: str = None,
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

        hits = []
        # FAISS search
        if self.index is not None:
            for q_vec in q_emb:
                hits.append(self._dedup_until_k(
                    lambda c: self.index.search(q_vec[None, :], c)[1][0],
                    top_k=top_k,
                    pool_size=pool_size,
                    diversity_threshold=diversity_threshold,
                    re_rank=re_rank,
                ))
        # In-memory search
        else:
            sims = self._inmem_embs @ q_emb.T
            for qi in range(sims.shape[1]):
                def top_ids(c):
                    part = np.argpartition(-sims[:, qi], c - 1)[:c]
                    return part[np.argsort(-sims[part, qi])]
                hits.append(self._dedup_until_k(
                    top_ids,
                    top_k=top_k,
                    pool_size=pool_size,
                    diversity_threshold=diversity_threshold,
                    re_rank=re_rank,
                ))
        return hits[0] if single else hits

    def _dedup_until_k(
        self,
        score_fn: callable,
        *,
        top_k: int,
        pool_size: int | None,
        diversity_threshold: float,
        re_rank: str | None,
    ) -> List[str]:
        
        cand = min(pool_size, len(self.passages))
        while True:
            ids = score_fn(cand)
            texts = [self.passages[i] for i in ids]
            vecs = np.stack([
                self.index.reconstruct(int(i))
                for i in ids
            ])

            picked = self._deduplicate(
                re_rank=re_rank,
                cand_vecs=vecs,
                cand_texts=texts,
                top_k=top_k,
                diversity_threshold=diversity_threshold,
            )

            if len(picked) == top_k or cand >= len(self.passages):
                return picked

            cand = min(cand * 2, len(self.passages))

    # Deduplication
    def _deduplicate(
        re_rank: str,
        cand_vecs: np.ndarray,
        cand_texts: List[str],
        top_k: int,
        diversity_threshold: float,
    ) -> List[str]:
        if not re_rank:
            return cand_texts[:top_k]

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

    def set_passages(self, passages: Union[str, List[str]]):
        self.passages = [passages] if isinstance(passages, str) else passages
        if self.save_index:
            self.index_path = self.cache_dir / f"{self._cache_key()}.faiss"
            self.index = self._load_or_build_index()
            self._inmem_embs = None
        else:
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

    # FAISS index
    def _load_or_build_index(self) -> faiss.Index:
        # Load existing index if available
        if self.save_index and self.index_path.exists():
            logging.info("Loading FAISS index from %s", self.index_path)
            return faiss.read_index(str(self.index_path))

        # Build a new index
        logging.info("Building FAISS index (streaming)…")
        dim = int(
            self._encode(self.passages[:1],
                        prompt=self.passage_prompt,
                        batch_size=1)[0].shape[0]
        )
        index = faiss.IndexFlatIP(dim)
        logging.info("  • using CPU-based IndexFlatIP")

        # Enocde passages and add them to the index
        BATCH = 512
        for b in tqdm(range(0, len(self.passages), BATCH),
                    desc="Adding embeddings",
                    unit="vec"):
            chunk = self.passages[b:b+BATCH]
            embs  = self._encode(chunk,
                                prompt=self.passage_prompt,
                                batch_size=BATCH).astype(np.float32)
            index.add(embs)

        # Save index  
        index_to_save = index
        if self.save_index:
            faiss.write_index(index_to_save, str(self.index_path))
            logging.info("Saved FAISS index to %s", self.index_path)
        return index_to_save

    # FAISS Utils
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

