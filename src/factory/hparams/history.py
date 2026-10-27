"""История запусков (runs/*/manifest.json) и заметки по гиперпараметрам (docs/hparams/) для A3 и эвристик."""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path

from factory.core.contracts import Manifest
from factory.core.registry import Registry
from factory.core.storage import read_model

logger = logging.getLogger(__name__)


def iter_manifests(runs_dir: Path):
    for path in sorted(runs_dir.glob("*/manifest.json")):
        try:
            m = read_model(path, Manifest)
        except Exception as e:
            logger.warning("Повреждённый манифест %s: %s", path, e)
            continue
        if m:
            yield m


def similarity(m: Manifest, task_type: str, language: str, params_b: float, n_train: int, n_classes: int,
               registry: Registry) -> float:
    s = 0.0
    s += 3.0 if m.spec.task_type == task_type else 0.0
    s += 1.0 if m.spec.language == language else 0.0
    try:
        other_params = registry.model(m.spec.base_model_id).params_b
        s += 1.5 * math.exp(-abs(math.log(other_params / params_b)))
    except KeyError:
        pass
    other_n = m.n_train or sum(m.dataset.split_sizes.values()) or 1
    s += 1.5 * math.exp(-abs(math.log(max(other_n, 1) / max(n_train, 1))))
    other_classes = len(m.spec.labels or [])
    if n_classes and other_classes:
        s += 1.0 * math.exp(-abs(math.log(other_classes / n_classes)))
    return round(s, 3)


def find_similar_runs(runs_dir: Path, registry: Registry, task_type: str, language: str, params_b: float,
                      n_train: int, n_classes: int, top: int = 5) -> list[dict]:
    """Сводка по завершённым запускам, похожим по типу задачи, языку, размеру модели, данных и числу классов.
    Метрика — финальная на val (test не участвует в подборе)."""
    rows = []
    for m in iter_manifests(runs_dir):
        if m.status != "finished" or not m.final_val_metrics:
            continue
        c = m.config
        rows.append({
            "run_id": m.run_id,
            "similarity": similarity(m, task_type, language, params_b, n_train, n_classes, registry),
            "task_type": m.spec.task_type, "language": m.spec.language, "base_model": m.spec.base_model_id,
            "n_classes": len(m.spec.labels or []),
            "backend": c.backend, "lr": c.optim.lr, "epochs_done": m.rounds[-1].epochs_done if m.rounds else None,
            "lora": c.lora.model_dump(include={"r", "alpha", "variant"}) if c.lora else None,
            "ptuning": c.ptuning.model_dump(include={"num_virtual_tokens", "encoder_hidden_size"}) if c.ptuning else None,
            "val_metrics": m.final_val_metrics,
            "rounds": len(m.rounds),
        })
    return sorted(rows, key=lambda r: -r["similarity"])[:top]


_HEADING = re.compile(r"^#{1,4}\s", re.MULTILINE)


def doc_chunks(docs_dir: Path) -> list[dict]:
    """Фрагменты заметок: разбиение по заголовкам markdown; doc_chunk_id = <файл>#<номер>."""
    chunks = []
    for path in sorted(docs_dir.glob("**/*.md")):
        text = path.read_text(encoding="utf-8")
        starts = [m.start() for m in _HEADING.finditer(text)] or [0]
        if starts[0] != 0:
            starts.insert(0, 0)
        for i, (a, b) in enumerate(zip(starts, starts[1:] + [len(text)])):
            body = text[a:b].strip()
            if body:
                chunks.append({"doc_chunk_id": f"{path.relative_to(docs_dir)}#{i}", "text": body})
    return chunks


def lookup_docs(docs_dir: Path, query: str, top: int = 4) -> list[dict]:
    from rank_bm25 import BM25Okapi
    from factory.search.retrieval import tokenize

    chunks = doc_chunks(docs_dir)
    if not chunks:
        return []
    bm = BM25Okapi([tokenize(c["text"]) or ["_"] for c in chunks])
    scores = bm.get_scores(tokenize(query))
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:top]
    return [{"doc_chunk_id": chunks[i]["doc_chunk_id"], "text": chunks[i]["text"][:1500]} for i in order if scores[i] > 0]
