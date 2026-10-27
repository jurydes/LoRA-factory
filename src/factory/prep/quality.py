"""Цели качества данных: что должно быть верно для train перед обучением.

Отчёт используют правила (нужен ли агент) и агент A2 (что ещё не сделано): рецепт нельзя зафиксировать,
пока цели не выполнены или для невыполненной не дано обоснование."""
from __future__ import annotations

import re

import pandas as pd

from factory.core.config import PrepRules
from factory.core.contracts import TaskFamily
from factory.prep import lang
from factory.prep import ops as ops_mod

_HTML = re.compile(r"<[a-zA-Z/][^>]{0,200}>|&[a-z]{2,8};")
_URL = re.compile(r"https?://|www\.")
LANG_SAMPLE = 400


def junk_mask(texts: pd.Series, answers: bool = False) -> pd.Series:
    """Мусор: HTML-разметка, URL, почти нет букв (≥ 50% не-буквенных символов) или текст короче 2 символов.

    answers — это ответы модели: короткий ответ из цифр («8800», «1995 г.», «5») — норма для QA и генерации,
    поэтому доля букв проверяется только у ответов длиннее 20 символов."""
    def bad(t: str) -> bool:
        t = str(t)
        s = t.strip()
        if _HTML.search(t) or _URL.search(t):
            return True
        if answers and len(s) < 20:
            return not s
        if len(s) < 2:
            return True
        letters = sum(ch.isalpha() for ch in t)
        return letters / max(len(t), 1) < 0.5
    return texts.map(bad)


def length_outlier_mask(df: pd.DataFrame, rules: PrepRules) -> pd.Series:
    """Выбросы по длине: вход длиннее p99.5 × 3 или ответ длиннее лимита токенов (~3.5 символа на токен)."""
    in_len, tgt_len = df["input"].str.len(), df["target"].str.len()
    cap = in_len.quantile(0.995) * 3 if len(df) else 0
    return (in_len > cap) | (tgt_len > rules.max_target_tokens_p99 * 3.5)


def quality_report(df: pd.DataFrame, family: TaskFamily, rules: PrepRules, language: str,
                   labels: list[str] | None = None, seed: int = 0, label_noise: bool = True) -> dict[str, dict]:
    """{цель: {value, threshold, ok, note}}. Значения — доли строк, если не сказано иное."""
    n = max(len(df), 1)
    goals: dict[str, dict] = {}

    def goal(name, value, threshold, ok, note=""):
        goals[name] = {"value": round(float(value), 4), "threshold": threshold, "ok": bool(ok), "note": note}

    empty = ((df["input"].str.strip() == "") | (df["target"].str.strip() == "")).sum()
    goal("no_empty_rows", empty, 0, empty == 0, "пустой вход или ответ")
    keys = df["input"].map(ops_mod.normalize_text) + "\x1f" + df["target"].map(ops_mod.normalize_text)
    dupes = keys.duplicated().sum()
    goal("no_exact_duplicates", dupes, 0, dupes == 0, "одинаковые пары вход–ответ")
    near = ops_mod.near_duplicate_mask(df, 0.9).sum() / n if len(df) <= 20_000 else 0.0
    goal("near_duplicates", near, rules.max_near_dupe_share, near <= rules.max_near_dupe_share, "почти-дубли входа (MinHash ≥ 0.9)")
    junk = (junk_mask(df["input"]) | junk_mask(df["target"], answers=True) & (family != "classification")).sum() / n
    goal("junk_text", junk, rules.max_junk_share, junk <= rules.max_junk_share, "HTML, URL, строки почти без букв")
    outliers = length_outlier_mask(df, rules).sum() / n
    goal("length_outliers", outliers, rules.max_length_outlier_share, outliers <= rules.max_length_outlier_share,
         "аномально длинный вход или ответ")
    sample = df["input"].sample(n=min(LANG_SAMPLE, len(df)), random_state=seed) if len(df) else df["input"]
    share = lang.language_share(sample.tolist()).get(language, 0.0) if len(df) else 1.0
    goal("language", share, rules.min_lang_share, share >= rules.min_lang_share, f"доля языка {language}")
    if family == "classification" and len(df):
        vc = df["target"].value_counts()
        ratio = vc.max() / max(vc.min(), 1)
        goal("class_balance", ratio, rules.max_imbalance, ratio <= rules.max_imbalance, "max/min примеров на класс")
        if labels:
            unknown = (~df["target"].isin(labels)).sum() / n
            goal("labels_in_spec", unknown, 0, unknown == 0, f"метки вне постановки {sorted(set(vc.index) - set(labels))[:10]}")
        if label_noise:
            noisy = len(ops_mod.label_issue_indices(df.reset_index(drop=True), seed)) / n
            goal("label_noise", noisy, rules.max_label_noise_share, noisy <= rules.max_label_noise_share,
                 "подозрительные метки (cleanlab, оценка)")
    return goals


def failing(goals: dict[str, dict]) -> list[str]:
    return [k for k, g in goals.items() if not g["ok"]]
