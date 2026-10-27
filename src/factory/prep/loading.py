"""Загрузка датасетов (HF Hub или файл пользователя) в pandas. Приведение к формату задачи — prep/schema.py."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from factory.core.contracts import DatasetCandidate
from factory.prep.schema import lang_code, mentions_language, tokens

logger = logging.getLogger(__name__)

USER_FORMATS = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".feather", ".arrow", ".xlsx"}
MAX_ROWS = 300_000


def read_user_file(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Читает датасет пользователя в любом совместимом с pandas формате."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in USER_FORMATS:
        raise ValueError(f"Формат {suffix} не поддерживается; допустимы {', '.join(sorted(USER_FORMATS))}")
    if suffix == ".csv":
        df = pd.read_csv(path, nrows=nrows)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t", nrows=nrows)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True, nrows=nrows)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in (".feather", ".arrow"):  # формат Arrow IPC — родной и для polars
        df = pd.read_feather(path)
    else:
        df = pd.read_excel(path, nrows=nrows)
    return df.head(nrows) if nrows else df


def _seq_names(feat) -> list[str] | None:
    """Имена ClassLabel внутри последовательности (теги NER: Sequence/List(ClassLabel))."""
    inner = getattr(feat, "feature", None)
    names = getattr(inner, "names", None)
    return list(names) if names else None


def _decode_class_labels(df: pd.DataFrame, features) -> pd.DataFrame:
    """Переводит ClassLabel-индексы (скалярные и в последовательностях) в названия меток.
    -1 — «метки нет» (неразмеченный test-сплит GLUE и подобных): становится пропуском, а не классом «-1»."""
    if features is None:
        return df
    for col, feat in features.items():
        if col not in df.columns:
            continue
        names = getattr(feat, "names", None)
        if names:
            df[col] = df[col].map(lambda v: (names[v] if 0 <= v < len(names) else None)
                                  if isinstance(v, (int, np.integer)) else v)
        elif seq := _seq_names(feat):
            df[col] = df[col].map(lambda vs: [seq[int(v)] if 0 <= int(v) < len(seq) else str(v) for v in vs]
                                  if _is_seq(vs) else vs)
    return df


def _is_seq(v) -> bool:
    return isinstance(v, (list, tuple, np.ndarray))


def _lang_rank(name: str, language: str) -> tuple:
    """Чем меньше, тем лучше: один язык лучше пары; в паре — с английским (en-ru чаще ar-ru); короче имя."""
    # язык пары — и незнакомый код из 2–3 букв (hu-ru)
    langs = {lang_code(t) or t for t in tokens(name) if lang_code(t) or (len(t) in (2, 3) and t.isalpha())}
    return len(langs), not ({"en", language} <= langs or langs == {language}), len(tokens(name)), name


def choose_splits(split_names: list[str], language: str | None) -> list[str] | None:
    """Сплиты по языкам (ru; train_ru/val_ru/test_ru; ru_mnli, ru_anli…) — берём сплиты нужного языка;
    иначе все (разбиваем сами)."""
    if not language:
        return None
    if language in split_names:
        return [language]
    picked = [s for s in split_names if mentions_language(s, language)]
    return picked or None


def pick_config(names: list[str], language: str | None) -> str | None:
    """Конфигурация на языке задачи (ru, rus_Cyrl, russian, xquad.ru, en-ru), иначе default, иначе первая."""
    if not names:
        return None
    if language:
        matched = [n for n in names if mentions_language(n, language)]
        if matched:
            return min(matched, key=lambda n: _lang_rank(n, language))
    if len(names) == 1:
        return None if names[0] == "default" else names[0]
    return "default" if "default" in names else names[0]


def resolve_config(hf_id: str, revision: str | None, language: str | None) -> str | None:
    from datasets import get_dataset_config_names
    try:
        names = get_dataset_config_names(hf_id, revision=revision)
    except Exception:
        return None
    return pick_config(names, language)


FULL_SHUFFLE_ROWS = 20_000  # сплит не больше — перемешивается целиком


def hf_sample(hf_id: str, n: int, revision: str | None = None, config_name: str | None = None,
              language: str | None = None, split_sizes: dict[str, int] | None = None) -> tuple[pd.DataFrame, dict]:
    """Сэмпл строк без полного скачивания (streaming) + метаданные: метки ClassLabel, выбранные конфигурация и сплиты.

    Строки перемешиваются через буфер: у отсортированных датасетов первые строки — один класс. Небольшой сплит
    (размер известен из split_sizes) перемешивается целиком — иначе первые тысячи строк одного класса дают
    «колонку меток с одним значением»."""
    from datasets import load_dataset

    if config_name is None:  # у многоязычных датасетов конфигурация по умолчанию — часто другой язык
        config_name = resolve_config(hf_id, revision, language)
    dsd = load_dataset(hf_id, config_name, streaming=True, revision=revision)
    names = list(dsd.keys())
    splits = choose_splits(names, language)
    pool = splits or names
    first = next((s for s in pool if "train" in tokens(s)), pool[0])
    ds = dsd[first]
    size = (split_sizes or {}).get(first, 0)
    buffer = size if 0 < size <= FULL_SHUFFLE_ROWS else max(1000, n * 10)
    rows = list(ds.shuffle(seed=0, buffer_size=max(buffer, n)).take(n))
    df = _decode_class_labels(pd.DataFrame(rows), ds.features)
    label_names = {c: list(f.names) for c, f in (ds.features or {}).items() if getattr(f, "names", None)}
    return df, {"label_names": label_names, "config_name": config_name, "splits": splits}


def hf_full(hf_id: str, revision: str | None = None, config_name: str | None = None, max_rows: int = MAX_ROWS,
            splits: list[str] | None = None) -> pd.DataFrame:
    """Загружает сплиты HF-датасета (все или выбранные) и объединяет: разбиение делаем сами с фиксированным seed."""
    from datasets import load_dataset

    dsd = load_dataset(hf_id, config_name, revision=revision)
    frames = []
    for split_name, ds in dsd.items():
        if splits and split_name not in splits:
            continue
        if len(ds) > max_rows:  # выборка до перевода в pandas: огромный сплит не должен целиком лечь в память
            ds = ds.shuffle(seed=0).select(range(max_rows))
        df = _decode_class_labels(ds.to_pandas(), ds.features)
        df["_orig_split"] = split_name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if len(df) > max_rows:
        logger.warning("Датасет %s обрезан до %d строк из %d", hf_id, max_rows, len(df))
        df = df.sample(n=max_rows, random_state=0).reset_index(drop=True)
    return df


def load_candidate_frame(candidate: DatasetCandidate, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    if candidate.source == "user":
        return read_user_file(candidate.local_path).head(max_rows)
    return hf_full(candidate.hf_id, candidate.revision, candidate.config_name, max_rows, candidate.splits)
