"""Retrieval: vector search, optional BM25 hybrid fusion, optional reranking."""
import re

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from . import config


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self, collection=config.COLLECTION, embed_model=config.EMBED_MODEL,
                 hybrid=config.USE_HYBRID, rerank=config.USE_RERANKER,
                 db_path=config.DB_PATH):
        self.embedder = SentenceTransformer(embed_model)
        self.name = collection
        self.client = chromadb.PersistentClient(path=db_path)
        self.hybrid, self.rerank = hybrid, rerank
        self._reranker = None
        self._load()

    def _load(self):
        """(Re)load the collection, its chunks, and the BM25 index."""
        self.col = self.client.get_collection(self.name)
        data = self.col.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.store = {i: {"id": i, "text": d, "meta": m}
                      for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])}
        self.count = len(self.ids)
        self.bm25 = None
        if self.hybrid and self.ids:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([_tokenize(self.store[i]["text"]) for i in self.ids])

    def _refresh_if_stale(self):
        """If the index was rebuilt since we loaded it, reload automatically."""
        try:
            current = self.client.get_collection(self.name)
        except Exception:
            raise RuntimeError("Search index not found. Click Build / rebuild index in the sidebar, or run: python -m src.ingest")
        if current.id != self.col.id:
            self._load()

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(config.RERANK_MODEL)
        return self._reranker

    def search(self, question, k=config.TOP_K):
        """Return the top-k chunks as [{id, text, meta}]."""
        self._refresh_if_stale()
        n = max(k * 5, 20) if (self.hybrid or self.rerank) else k
        n = min(n, self.count)

        emb = self.embedder.encode(question).tolist()
        res = self.col.query(query_embeddings=[emb], n_results=n)
        vec_ids = res["ids"][0]

        if self.hybrid:
            scores = self.bm25.get_scores(_tokenize(question))
            order = np.argsort(scores)[::-1][:n]
            bm_ids = [self.ids[i] for i in order if scores[i] > 0]
            fused = {}
            for ranked in (vec_ids, bm_ids):  # reciprocal rank fusion
                for rank, cid in enumerate(ranked):
                    fused[cid] = fused.get(cid, 0) + 1 / (60 + rank + 1)
            candidates = sorted(fused, key=fused.get, reverse=True)
        else:
            candidates = vec_ids

        if self.rerank:
            candidates = candidates[:20]
            ce = self._get_reranker()
            scores = ce.predict([(question, self.store[c]["text"]) for c in candidates])
            candidates = [c for _, c in sorted(zip(scores, candidates),
                                               key=lambda x: x[0], reverse=True)]

        return [self.store[c] for c in candidates[:k]]
