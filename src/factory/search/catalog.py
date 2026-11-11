"""Кэшированный каталог датасетов Hugging Face + офлайн-эмбеддинги описаний.

Записи совместимы с форматом курсовой (dataset_work.search_datasets: id, description, tags, score)
и дополнены метаданными для детерминированных фильтров.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from factory.core.config import AppConfig, get_config
from factory.core.storage import read_json, write_json

logger = logging.getLogger(__name__)


@dataclass
class Catalog:
    records: list[dict]
    embeddings: np.ndarray | None

    def __len__(self) -> int:
        return len(self.records)


def _catalog_paths(cfg: AppConfig, language: str) -> tuple[Path, Path]:
    return cfg.cache_dir / f"catalog_{language}.json", cfg.cache_dir / f"embeddings_{language}.npy"


def record_text(rec: dict) -> str:
    tags = " ".join(t for t in rec.get("tags", []) if not t.startswith(("region:", "size_categories:")))
    return f"{rec['id'].replace('/', ' ').replace('_', ' ').replace('-', ' ')}. {rec.get('description', '')} {tags}"


def _license_from_tags(tags: list[str]) -> str | None:
    for t in tags:
        if t.startswith("license:"):
            return t.split(":", 1)[1]
    return None


def fetch_catalog(language: str, limit: int) -> list[dict]:
    """Скачивает список датасетов с HF Hub с фильтрами курсовой: язык и наличие описания."""
    from huggingface_hub import HfApi

    seen: dict[str, dict] = {}
    for ds in HfApi().list_datasets(language=language, limit=limit, full=True):
        if not ds.description or ds.id in seen or getattr(ds, "disabled", False):
            continue
        tags = list(ds.tags or [])
        seen[ds.id] = {
            "id": ds.id,
            "description": (ds.description or "")[:1000],
            "tags": tags,
            "score": 0,
            "license": _license_from_tags(tags),
            "task_categories": [t.split(":", 1)[1] for t in tags if t.startswith("task_categories:")],
            "size_category": next((t.split(":", 1)[1] for t in tags if t.startswith("size_categories:")), None),
            "downloads": getattr(ds, "downloads", 0) or 0,
            "gated": bool(getattr(ds, "gated", False)),
        }
    return list(seen.values())


@lru_cache(maxsize=2)
def get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _prefix(model_name: str, kind: str) -> str:
    # модели семейства e5 обучены с префиксами query:/passage:
    return f"{kind}: " if "e5" in model_name.lower() else ""


def embed_passages(texts: list[str], model_name: str) -> np.ndarray:
    model = get_embedder(model_name)
    pre = _prefix(model_name, "passage")
    return model.encode([pre + t for t in texts], normalize_embeddings=True, batch_size=64, show_progress_bar=False)


def embed_query(text: str, model_name: str) -> np.ndarray:
    model = get_embedder(model_name)
    return model.encode([_prefix(model_name, "query") + text], normalize_embeddings=True)[0]


def build_catalog(language: str = "ru", cfg: AppConfig | None = None, records: list[dict] | None = None) -> Catalog:
    """Строит кэш каталога и эмбеддинги описаний (офлайн-шаг)."""
    cfg = cfg or get_config()
    if cfg.offline and records is None:
        raise RuntimeError("В закрытом контуре нельзя строить каталог с Hugging Face")
    records = records if records is not None else fetch_catalog(language, cfg.search.catalog_limit)
    emb = embed_passages([record_text(r) for r in records], cfg.search.embedding_model) if records else np.zeros((0, 1))
    meta_path, emb_path = _catalog_paths(cfg, language)
    write_json(meta_path, {"embedding_model": cfg.search.embedding_model, "records": records})
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    logger.info("Каталог %s: %d датасетов, эмбеддинги %s", language, len(records), emb.shape)
    return Catalog(records, emb)


def load_catalog(language: str = "ru", cfg: AppConfig | None = None, build_if_missing: bool = True) -> Catalog:
    cfg = cfg or get_config()
    meta_path, emb_path = _catalog_paths(cfg, language)
    meta = read_json(meta_path)
    if meta is None:
        if cfg.offline:
            return Catalog([], None)
        if not build_if_missing:
            return Catalog([], None)
        return build_catalog(language, cfg)
    emb = np.load(emb_path) if emb_path.exists() else None
    if emb is not None and (meta.get("embedding_model") != cfg.search.embedding_model or len(emb) != len(meta["records"])):
        if cfg.offline:
            return Catalog(meta["records"], None)
        logger.warning("Эмбеддинги каталога устарели (сменилась модель или записи) — пересчитываю")
        return build_catalog(language, cfg, records=meta["records"])
    return Catalog(meta["records"], emb)
