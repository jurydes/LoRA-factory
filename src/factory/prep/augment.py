"""Синтез данных для train: генерация LLM по образцам исходного датасета (параллельные батчи-субагенты),
детерминированная аугментация (синонимизация по словарю, перестановки/удаления слов, порядок предложений) или гибрид.
Синтетические строки помечаются synthetic=true и попадают только в train."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from factory.core import llm
from factory.core.contracts import AugmentOp, TaskFamily
from factory.core.storage import sha256_file
from factory.prep import lang
from factory.prep.ops import normalize_text
from factory.prep.schema import row_id

logger = logging.getLogger(__name__)

EXAMPLES_PER_PROMPT = 6
ROWS_PER_CALL = 10


class SyntheticExample(BaseModel):
    input: str
    target: str = ""


class SyntheticBatch(BaseModel):
    examples: list[SyntheticExample] = Field(description="Новые примеры в стиле образцов")


GEN_INSTRUCTIONS = """Ты генерируешь новые обучающие примеры, похожие на образцы из датасета:
тот же стиль, домен, длина, язык и уровень шума. Не копируй образцы и не перефразируй их дословно.
Каждый пример должен быть правдоподобен как реальная запись этого датасета."""


def _prompt(task_description: str, family: TaskFamily, samples: pd.DataFrame, n: int, label: str | None) -> str:
    shown = samples[["input", "target"]].astype(str).map(lambda s: s[:500]).to_dict(orient="records")
    if family == "classification":
        head = (f"Задача: {task_description}\nСгенерируй {n} новых входных текстов с меткой «{label}». "
                f"Поле target оставь пустым — метку подставит код.")
        shown = [{"input": r["input"]} for r in shown]
    else:
        head = f"Задача: {task_description}\nСгенерируй {n} новых пар input → target."
    return f"{head}\nОбразцы:\n{json.dumps(shown, ensure_ascii=False, indent=1)}"


async def synthesize_llm(train: pd.DataFrame, family: TaskFamily, n_rows: int, task_description: str,
                         language: str, workers: int = 4, seed: int = 0) -> pd.DataFrame:
    """Параллельные батчи-«субагенты»: каждый видит свою выборку образцов и генерирует ROWS_PER_CALL строк."""
    rng = random.Random(seed)
    jobs: list[tuple[pd.DataFrame, str | None]] = []
    if family == "classification":
        counts = train["target"].value_counts()
        # больше синтетики — более редким меткам
        need = (counts.max() - counts).clip(lower=0)
        weights = (need / need.sum()) if need.sum() > 0 else pd.Series(1 / len(counts), index=counts.index)
        for label, w in weights.items():
            pool = train[train["target"] == label]
            for _ in range(int(round(n_rows * w / ROWS_PER_CALL))):
                jobs.append((pool.sample(n=min(EXAMPLES_PER_PROMPT, len(pool)), random_state=rng.randint(0, 1 << 30)), label))
    else:
        for _ in range(max(1, n_rows // ROWS_PER_CALL)):
            jobs.append((train.sample(n=min(EXAMPLES_PER_PROMPT, len(train)), random_state=rng.randint(0, 1 << 30)), None))

    sem = asyncio.Semaphore(workers)

    async def run(samples, label):
        async with sem:
            try:
                batch: SyntheticBatch = await llm.complete(
                    "augment", _prompt(task_description, family, samples, ROWS_PER_CALL, label),
                    output_type=SyntheticBatch, instructions=GEN_INSTRUCTIONS)
            except Exception as e:
                logger.warning("Батч синтеза не удался: %s", e)
                return []
            return [(ex.input, label if label is not None else ex.target) for ex in batch.examples]

    rows = [r for chunk in await asyncio.gather(*(run(s, l) for s, l in jobs)) for r in chunk]
    return validate_synthetic(pd.DataFrame(rows, columns=["input", "target"]), train, language)


_SENT = re.compile(r"(?<=[.!?…])\s+")
_WORD = re.compile(r"^[^\W\d_]{4,}$")


class SynonymMap(BaseModel):
    synonyms: dict[str, list[str]] = Field(description="слово → 1–3 синонима в той же форме (падеж, число, род)")


async def build_synonyms(texts: list[str], language: str, n_words: int = 150) -> dict[str, list[str]]:
    """Словарь синонимов для частотных слов датасета — один раз через LLM, дальше подстановка детерминирована.
    Проверка: синоним — одно слово из букв, не совпадает с исходным."""
    counts: dict[str, int] = {}
    for t in texts:
        for w in str(t).lower().split():
            w = w.strip(".,!?;:«»\"'()")
            if _WORD.match(w):
                counts[w] = counts.get(w, 0) + 1
    top = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n_words]]
    if not top:
        return {}
    try:
        res: SynonymMap = await llm.complete(
            "augment", f"Язык: {language}. Дай синонимы к словам (сохраняя грамматическую форму): {', '.join(top)}",
            output_type=SynonymMap,
            instructions="Для каждого слова — до 3 синонимов в той же форме; служебные и неоднозначные слова пропусти.")
    except Exception as e:
        logger.warning("Словарь синонимов не построен (%s) — только перестановки и удаления", e)
        return {}
    return {w.lower(): [s for s in syns if _WORD.match(s) and s.lower() != w.lower()][:3]
            for w, syns in res.synonyms.items() if w.lower() in counts}


def _perturb(text: str, rng: random.Random, synonyms: dict[str, list[str]] | None = None) -> str:
    """Синонимизация (если есть словарь), перестановка предложений, удаление или перестановка слов."""
    words = text.split()
    if synonyms and rng.random() < 0.6:
        out, changed = [], False
        for w in words:
            core = w.strip(".,!?;:«»\"'()")
            alts = synonyms.get(core.lower())
            if alts and rng.random() < 0.3:
                alt = rng.choice(alts)
                alt = alt.capitalize() if core[:1].isupper() else alt
                out.append(w.replace(core, alt))
                changed = True
            else:
                out.append(w)
        if changed:
            return " ".join(out)
    sentences = _SENT.split(text)
    if len(sentences) > 2 and rng.random() < 0.5:
        rng.shuffle(sentences)
        return " ".join(sentences)
    if len(words) < 4:
        return text
    if rng.random() < 0.5:  # случайное удаление ~10% слов
        kept = [w for w in words if rng.random() > 0.1]
        return " ".join(kept or words)
    i, j = rng.sample(range(len(words)), 2)  # перестановка двух слов
    words[i], words[j] = words[j], words[i]
    return " ".join(words)


def synthesize_deterministic(train: pd.DataFrame, family: TaskFamily, n_rows: int, seed: int = 0,
                             synonyms: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Только для классификации: возмущение входа не меняет метку; для генерации эталон бы разошёлся со входом."""
    if family != "classification":
        logger.warning("Детерминированная аугментация для генеративных задач не применяется")
        return pd.DataFrame(columns=["input", "target"])
    rng = random.Random(seed)
    counts = train["target"].value_counts()
    need = (counts.max() - counts).clip(lower=0)
    weights = need / need.sum() if need.sum() > 0 else pd.Series(1 / len(counts), index=counts.index)
    rows = []
    for label, w in weights.items():
        pool = train[train["target"] == label]["input"].tolist()
        for _ in range(int(round(n_rows * w))):
            rows.append((_perturb(rng.choice(pool), rng, synonyms), label))
    return validate_synthetic(pd.DataFrame(rows, columns=["input", "target"]), train, None)


def validate_synthetic(synth: pd.DataFrame, train: pd.DataFrame, language: str | None) -> pd.DataFrame:
    """Отсев: пустые, копии train и друг друга, чужой язык, длины за пределами наблюдаемых."""
    if synth.empty:
        return synth.assign(_id=[], synthetic=[])
    synth = synth.astype(str)
    synth = synth[(synth["input"].str.strip() != "") & (synth["target"].str.strip() != "")]
    seen = set(train["input"].map(normalize_text))
    norm = synth["input"].map(normalize_text)
    synth = synth[~norm.isin(seen) & ~norm.duplicated()]
    lo, hi = train["input"].str.len().quantile([0.01, 0.99])
    synth = synth[synth["input"].str.len().between(max(1, lo * 0.5), hi * 1.5)]
    if language:
        synth = synth[synth["input"].map(lang.detect) == language]
    synth = synth.reset_index(drop=True)
    synth.insert(0, "_id", [row_id(i, t) for i, t in zip(synth["input"], synth["target"])])
    synth["synthetic"] = True
    return synth


async def synthesize(method: Literal["llm", "deterministic", "hybrid"], train: pd.DataFrame, family: TaskFamily,
                     n_rows: int, task_description: str, language: str, out_dir: Path, workers: int = 4,
                     seed: int = 0) -> AugmentOp:
    if method == "llm":
        df = await synthesize_llm(train, family, n_rows, task_description, language, workers, seed)
    else:
        synonyms = await build_synonyms(train["input"].tolist(), language) if family == "classification" else {}
        if method == "deterministic":
            df = synthesize_deterministic(train, family, n_rows, seed, synonyms)
        else:
            half = n_rows // 2
            df = pd.concat([
                await synthesize_llm(train, family, half, task_description, language, workers, seed),
                synthesize_deterministic(train, family, n_rows - half, seed, synonyms),
            ], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"synthetic_{method}_{seed}_{len(df)}.jsonl"
    df[["_id", "input", "target"]].to_json(out_dir / name, orient="records", lines=True, force_ascii=False)
    return AugmentOp(method=method, path=name, sha256=sha256_file(out_dir / name), n_rows=len(df))
