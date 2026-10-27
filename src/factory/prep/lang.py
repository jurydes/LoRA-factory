"""Определение языка: fasttext lid.176 из курсовой (dataset_work.detect_language), запасной вариант — по алфавиту."""
from __future__ import annotations

import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

_CYR = re.compile(r"[а-яё]", re.IGNORECASE)
_LAT = re.compile(r"[a-z]", re.IGNORECASE)


@lru_cache(maxsize=1)
def _fasttext_detector():
    try:
        from dataset_work import detect_language
        return detect_language
    except Exception as e:  # нет lid.176.bin или fasttext
        logger.warning("fasttext-детектор языка недоступен (%s), использую эвристику по алфавиту", e)
        return None


def _script_guess(text: str) -> str:
    cyr, lat = len(_CYR.findall(text)), len(_LAT.findall(text))
    if cyr == lat == 0:
        return "unk"
    return "ru" if cyr >= lat else "en"


def detect(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return "unk"
    detector = _fasttext_detector()
    if detector is None:
        return _script_guess(text)
    try:
        return detector(text)[0]
    except Exception:
        return _script_guess(text)


def language_share(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {}
    counts: dict[str, int] = {}
    for t in texts:
        lang = detect(t)
        counts[lang] = counts.get(lang, 0) + 1
    return {k: v / len(texts) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}
