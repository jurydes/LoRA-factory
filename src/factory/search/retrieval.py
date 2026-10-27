"""Гибридный поиск по каталогу: BM25 + dense → RRF → cross-encoder."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from factory.core.config import SearchConfig
from factory.search.catalog import Catalog, embed_query, record_text

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank), rank с единицы."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


def bm25_ranking(catalog: Catalog, query: str) -> list[int]:
    from rank_bm25 import BM25Okapi

    corpus = [tokenize(record_text(r)) or ["_"] for r in catalog.records]
    scores = BM25Okapi(corpus).get_scores(tokenize(query))
    order = np.argsort(-scores)
    return [int(i) for i in order if scores[i] > 0]


def dense_ranking(catalog: Catalog, query: str, model_name: str) -> list[int]:
    if catalog.embeddings is None or len(catalog.embeddings) == 0:
        return []
    q = embed_query(query, model_name)
    sims = catalog.embeddings @ q
    return [int(i) for i in np.argsort(-sims)]


@lru_cache(maxsize=1)
def get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def hybrid_search(catalog: Catalog, query: str, cfg: SearchConfig, rerank: bool = True) -> list[tuple[int, dict]]:
    """Возвращает [(индекс записи, stage_scores)] для top_llm финалистов."""
    if not len(catalog):
        return []
    bm = bm25_ranking(catalog, query)
    dense = dense_ranking(catalog, query, cfg.embedding_model)
    fused = rrf([bm, dense], k=cfg.rrf_k)
    bm_rank = {idx: r for r, idx in enumerate(bm, start=1)}
    dense_rank = {idx: r for r, idx in enumerate(dense, start=1)}
    top = sorted(fused, key=lambda i: -fused[i])[: cfg.top_rerank]

    stages = {i: {"bm25_rank": bm_rank.get(i), "dense_rank": dense_rank.get(i), "rrf": fused[i]} for i in top}
    if rerank and top:
        docs = [record_text(catalog.records[i]) for i in top]
        ranked = get_reranker(cfg.reranker_model).rank(query, docs, top_k=len(docs))
        order = []
        for item in ranked:
            idx = top[item["corpus_id"]]
            stages[idx]["rerank"] = float(item["score"])
            order.append(idx)
        top = order
    return [(i, stages[i]) for i in top]
