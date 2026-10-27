"""Операции рецепта подготовки данных. Одна реализация для инструментов A2 (на выборке) и для
применения рецепта ко всему train (обычным кодом). Все операции детерминированы при фиксированном seed."""
from __future__ import annotations

import logging
import operator as op_mod
import re
from pathlib import Path

import numpy as np
import pandas as pd

from factory.core.contracts import (
    AugmentOp, CleanTextOp, DedupeOp, DropLabelNoiseOp, FilterOp, MapLabelsOp, MapSchemaOp, RebalanceOp, Recipe, RecipeOp,
)
from factory.core.storage import sha256_file
from factory.prep import lang

logger = logging.getLogger(__name__)

_OPS = {"<": op_mod.lt, "<=": op_mod.le, ">": op_mod.gt, ">=": op_mod.ge, "==": op_mod.eq, "!=": op_mod.ne}


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(s).lower())).strip()


def field_values(df: pd.DataFrame, field: str) -> pd.Series:
    if field == "input_len":
        return df["input"].str.len()
    if field == "target_len":
        return df["target"].str.len()
    if field == "language":
        return df["input"].map(lang.detect)
    if field == "label":
        return df["target"]
    if field == "is_dupe":
        return df["input"].map(normalize_text).duplicated(keep="first")
    raise ValueError(f"Поле фильтра {field} не из закрытого набора")


def filter_mask(df: pd.DataFrame, f: FilterOp) -> pd.Series:
    """True — строка остаётся."""
    values = field_values(df, f.field)
    if f.operator in ("in", "not_in"):
        allowed = set(f.value if isinstance(f.value, (list, tuple, set)) else [f.value])
        mask = values.isin(allowed)
        return mask if f.operator == "in" else ~mask
    value = f.value
    if f.field in ("input_len", "target_len"):
        value = float(value)
    if f.field == "is_dupe":
        value = bool(value) if not isinstance(value, str) else value.lower() == "true"
    return _OPS[f.operator](values, value)


def apply_filter(df: pd.DataFrame, f: FilterOp) -> pd.DataFrame:
    return df[filter_mask(df, f)].reset_index(drop=True)


def _minhash(text: str, num_perm: int = 128):
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm, seed=1)
    tokens = normalize_text(text).split()
    shingles = {" ".join(tokens[i:i + 3]) for i in range(max(1, len(tokens) - 2))} or {""}
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def near_duplicate_mask(df: pd.DataFrame, threshold: float) -> pd.Series:
    """True для строк, у которых есть более ранний near-дубль (MinHash LSH по 3-граммам слов входа)."""
    from datasketch import MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    dup = np.zeros(len(df), dtype=bool)
    for i, text in enumerate(df["input"].tolist()):
        m = _minhash(text)
        if lsh.query(m):
            dup[i] = True
        else:
            lsh.insert(str(i), m)
    return pd.Series(dup, index=df.index)


def apply_dedupe(df: pd.DataFrame, d: DedupeOp) -> pd.DataFrame:
    if d.mode == "exact":
        keys = df["input"].map(normalize_text) + "\x1f" + df["target"].map(normalize_text)
        return df[~keys.duplicated(keep="first")].reset_index(drop=True)
    return df[~near_duplicate_mask(df, d.threshold)].reset_index(drop=True)


def label_issue_indices(df: pd.DataFrame, seed: int = 0) -> list[int]:
    """Индексы строк с вероятным шумом меток, по убыванию подозрительности.

    cleanlab.find_label_issues по out-of-sample вероятностям TF-IDF + LogReg (5-fold).
    """
    from cleanlab.filter import find_label_issues
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    labels_str = df["target"].astype(str)
    counts = labels_str.value_counts()
    keep = labels_str.isin(counts[counts >= 5].index)
    sub = df[keep]
    if sub["target"].nunique() < 2 or len(sub) < 20:
        return []
    classes = sorted(sub["target"].unique())
    y = sub["target"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=50_000, sublinear_tf=True).fit_transform(sub["input"])
    n_splits = int(min(5, counts[counts >= 5].min()))
    probs = cross_val_predict(
        LogisticRegression(max_iter=1000, C=4.0), X, y,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), method="predict_proba")
    ranked = find_label_issues(y, probs, return_indices_ranked_by="self_confidence")
    return [int(sub.index[i]) for i in ranked]


def apply_drop_label_noise(df: pd.DataFrame, d: DropLabelNoiseOp, seed: int = 0) -> pd.DataFrame:
    issues = label_issue_indices(df.reset_index(drop=True), seed)
    limit = int(len(df) * d.max_fraction)
    drop = set(issues[:limit])
    return df.reset_index(drop=True).drop(index=list(drop)).reset_index(drop=True)


def apply_rebalance(df: pd.DataFrame, r: RebalanceOp, seed: int = 0) -> pd.DataFrame:
    """undersample — урезать крупные классы до max_ratio×min; oversample — повторить мелкие до max/max_ratio."""
    real = df[~df["synthetic"]] if "synthetic" in df else df
    counts = real["target"].value_counts()
    if len(counts) < 2:
        return df
    rng = np.random.default_rng(seed)
    parts = []
    if r.method == "undersample":
        cap = int(counts.min() * r.max_ratio)
        for label, group in df.groupby("target", sort=True):
            if len(group) > cap:
                group = group.iloc[np.sort(rng.choice(len(group), size=cap, replace=False))]
            parts.append(group)
    else:
        floor = int(np.ceil(counts.max() / r.max_ratio))
        for label, group in df.groupby("target", sort=True):
            parts.append(group)
            if len(group) < floor:
                extra = group.iloc[rng.choice(len(group), size=floor - len(group), replace=True)]
                parts.append(extra)
    return pd.concat(parts).sort_index(kind="stable").reset_index(drop=True)


def load_synthetic(path: Path) -> pd.DataFrame:
    if path.stat().st_size == 0:
        return pd.DataFrame({"_id": [], "input": [], "target": [], "synthetic": []}).astype({"synthetic": bool})
    df = pd.read_json(path, lines=True, dtype={"_id": str, "input": str, "target": str})
    df["synthetic"] = True
    return df[["_id", "input", "target", "synthetic"]]


def apply_augment(df: pd.DataFrame, a: AugmentOp, base_dir: Path) -> pd.DataFrame:
    path = base_dir / a.path
    if sha256_file(path) != a.sha256:
        raise ValueError(f"Файл синтетики {path} изменился после фиксации рецепта")
    return pd.concat([df, load_synthetic(path)], ignore_index=True)


_HTML_TAG = re.compile(r"<[^>]{1,200}>")
_URL = re.compile(r"https?://\S+|www\.\S+")


def clean_text(text: str, c: CleanTextOp) -> str:
    import html
    if c.strip_html:
        text = html.unescape(_HTML_TAG.sub(" ", text))
    if c.remove_urls:
        text = _URL.sub(" ", text)
    if c.max_char_repeat:
        n = c.max_char_repeat
        text = re.sub(rf"(.)\1{{{n},}}", lambda m: m.group(1) * n, text)
    if c.collapse_whitespace:
        text = re.sub(r"[ \t\u00a0]+", " ", re.sub(r"\s*\n\s*", "\n", text)).strip()
    return text


def apply_clean_text(df: pd.DataFrame, c: CleanTextOp) -> pd.DataFrame:
    out = df.copy()
    out["input"] = out["input"].map(lambda t: clean_text(t, c))
    out["target"] = out["target"].map(lambda t: clean_text(t, c))
    return out


def apply_map_labels(df: pd.DataFrame, m: MapLabelsOp) -> pd.DataFrame:
    out = df.copy()
    out["target"] = out["target"].map(lambda v: m.mapping.get(v, v))
    return out


def apply_op(df: pd.DataFrame, op: RecipeOp, seed: int, base_dir: Path | None = None) -> pd.DataFrame:
    if isinstance(op, MapSchemaOp):
        return df  # map_schema применяется при загрузке, до разбиения
    if isinstance(op, CleanTextOp):
        return apply_clean_text(df, op)
    if isinstance(op, MapLabelsOp):
        return apply_map_labels(df, op)
    if isinstance(op, DedupeOp):
        return apply_dedupe(df, op)
    if isinstance(op, FilterOp):
        return apply_filter(df, op)
    if isinstance(op, DropLabelNoiseOp):
        return apply_drop_label_noise(df, op, seed)
    if isinstance(op, RebalanceOp):
        return apply_rebalance(df, op, seed)
    if isinstance(op, AugmentOp):
        if base_dir is None:
            raise ValueError("Для augment нужна папка с файлом синтетики")
        return apply_augment(df, op, base_dir)
    raise TypeError(f"Неизвестная операция {op}")


def apply_label_schema(df: pd.DataFrame, recipe: Recipe) -> pd.DataFrame:
    """Операции рецепта, меняющие сам набор меток (map_labels, фильтр по метке), — для val и test: иначе модель
    учится отвечать новыми метками, а оценивается по старым. Чистка (дубли, мусор, шум, баланс) остаётся только в train."""
    for op in recipe.ops:
        if isinstance(op, MapLabelsOp) or (isinstance(op, FilterOp) and op.field == "label"):
            df = apply_op(df, op, recipe.split_seed)
    return df


def apply_recipe(train: pd.DataFrame, recipe: Recipe, base_dir: Path | None = None) -> pd.DataFrame:
    """Применяет операции рецепта к train по порядку. val/test рецепт не трогает (кроме apply_label_schema)."""
    df = train
    for op in recipe.ops:
        before = len(df)
        df = apply_op(df, op, recipe.split_seed, base_dir)
        logger.info("Рецепт: %s → %d → %d строк", op.op, before, len(df))
    return df
