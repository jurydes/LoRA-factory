"""Профиль данных: баланс меток, длины, дубликаты, пустые, язык."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factory.core.contracts import DataProfile, LengthStats, TaskFamily
from factory.prep import lang
from factory.prep.ops import normalize_text

CHARS_PER_TOKEN = 3.5
LANG_SAMPLE = 500


def length_stats(values: pd.Series) -> LengthStats:
    if values.empty:
        return LengthStats()
    q = np.quantile(values.to_numpy(dtype=float), [0.5, 0.9, 0.95, 0.99])
    return LengthStats(mean=round(float(values.mean()), 1), p50=float(q[0]), p90=float(q[1]),
                       p95=float(q[2]), p99=float(q[3]), max=float(values.max()))


def profile_frame(df: pd.DataFrame, family: TaskFamily, seed: int = 0) -> DataProfile:
    n = len(df)
    in_len, tgt_len = df["input"].str.len(), df["target"].str.len()
    keys = df["input"].map(normalize_text) + "\x1f" + df["target"].map(normalize_text)
    label_counts, imbalance = {}, None
    if family == "classification" and n:
        vc = df["target"].value_counts()
        label_counts = {str(k): int(v) for k, v in vc.items()}
        imbalance = round(float(vc.max() / max(vc.min(), 1)), 2)
    sample = df["input"].sample(n=min(LANG_SAMPLE, n), random_state=seed) if n else df["input"]
    return DataProfile(
        n_rows=n,
        n_empty=int(((df["input"].str.strip() == "") | (df["target"].str.strip() == "")).sum()),
        n_exact_dupes=int(keys.duplicated().sum()),
        n_synthetic=int(df["synthetic"].sum()) if "synthetic" in df else 0,
        label_counts=label_counts,
        imbalance_ratio=imbalance,
        input_chars=length_stats(in_len),
        target_chars=length_stats(tgt_len),
        input_tokens_est=length_stats(in_len / CHARS_PER_TOKEN),
        target_tokens_est=length_stats(tgt_len / CHARS_PER_TOKEN),
        language_share={k: round(v, 3) for k, v in list(lang.language_share(sample.tolist()).items())[:5]},
    )


def profile_summary(p: DataProfile) -> dict:
    """Компактная сводка для LLM (без сырых данных)."""
    d = p.model_dump(exclude={"input_tokens_est", "target_tokens_est"})
    d["input_chars"] = {k: d["input_chars"][k] for k in ("p50", "p95", "max")}
    d["target_chars"] = {k: d["target_chars"][k] for k in ("p50", "p95", "max")}
    if len(p.label_counts) > 30:
        d["label_counts"] = dict(list(p.label_counts.items())[:30]) | {"…": f"ещё {len(p.label_counts) - 30} меток"}
    return d
