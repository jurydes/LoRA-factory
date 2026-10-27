"""Метрики по типу задачи (заданы реестром, не LLM) и простые baseline'ы.

Формат ответа задаёт способ оценки (registry/tasks.yaml → format):
  label    — классификация: macro/weighted-F1, accuracy, per-class;
  text     — генерация: ROUGE-1/2/L, chrF, BLEU, BERTScore, ExactMatch;
  entities — NER: entity-level micro-P/R/F1 (тип + нормализованный текст), per-type;
  span     — экстрактивный QA: ExactMatch и token-F1 по SQuAD, лучший из нескольких эталонов.
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter

from factory.core.contracts import PredictionExample

UNPARSED = "__unparsed__"
NO_ENTITIES = "нет"


def normalize(s: str) -> str:
    return re.sub(r"[\s\W_]+", " ", str(s).lower()).strip()


# ---------------------------------------------------------------- классификация

def parse_label(prediction: str, labels: list[str]) -> str | None:
    """Ответ модели → метка из набора. Точное совпадение после нормализации, затем однозначный префикс/вхождение."""
    p = normalize(prediction)
    if not p:
        return None
    by_norm = {normalize(l): l for l in labels}
    if p in by_norm:
        return by_norm[p]
    starts = [l for n, l in by_norm.items() if p.startswith(n)]
    if len(starts) == 1:
        return starts[0]
    contains = [l for n, l in by_norm.items() if re.search(rf"\b{re.escape(n)}\b", p)]
    return contains[0] if len(contains) == 1 else None


def classification_metrics(predictions: list[str], references: list[str], labels: list[str]
                           ) -> tuple[dict[str, float], dict[str, dict[str, float]], float, list[str]]:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    parsed = [parse_label(p, labels) for p in predictions]
    unparsed = sum(p is None for p in parsed) / max(len(parsed), 1)
    y_pred = [p if p is not None else UNPARSED for p in parsed]
    label_set = sorted(set(labels) | set(references))
    metrics = {
        "accuracy": float(accuracy_score(references, y_pred)),
        "macro_f1": float(f1_score(references, y_pred, labels=label_set, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(references, y_pred, labels=label_set, average="weighted", zero_division=0)),
    }
    prec, rec, f1, sup = precision_recall_fscore_support(references, y_pred, labels=label_set, zero_division=0)
    per_class = {l: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                 for l, p, r, f, s in zip(label_set, prec, rec, f1, sup)}
    return metrics, per_class, unparsed, y_pred


# ---------------------------------------------------------------- генерация текста

class _WhitespaceTokenizer:
    """rouge-score по умолчанию выкидывает не-латиницу; для русского токенизируем по словам."""

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


def generation_metrics(predictions: list[str], references: list[str], names: list[str]) -> dict[str, float]:
    import sacrebleu

    safe = [p if p.strip() else "нет" for p in predictions]  # как в inference.compute_metrics курсовой
    n = max(len(references), 1)
    out: dict[str, float] = {}
    if "chrf" in names:
        out["chrf"] = sacrebleu.corpus_chrf(safe, [references]).score / 100
    if "bleu" in names:  # sacrebleu отдаёт 0–100; как chrF — к шкале 0–1 остальных метрик
        out["bleu"] = sacrebleu.corpus_bleu(safe, [references]).score / 100
    if "exact_match" in names:
        out["exact_match"] = sum(normalize(p) == normalize(r) for p, r in zip(predictions, references)) / n
    rouge = [k for k in ("rouge1", "rouge2", "rougeL") if k in names]
    if rouge:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(rouge, tokenizer=_WhitespaceTokenizer())
        scores = [scorer.score(r, p) for p, r in zip(safe, references)]
        for k in rouge:
            out[k] = sum(s[k].fmeasure for s in scores) / n
    if "bert_score_f1" in names:
        out["bert_score_f1"] = _bert_score_f1(safe, references)
    return out


def _bert_score_f1(predictions: list[str], references: list[str]) -> float:
    """BERTScore F1 на GPU, если там свободно ≥ 2 ГБ (на CPU 2000 длинных пар считаются ~7 минут), иначе на CPU."""
    import torch
    from bert_score import score as bert_score_fn

    device = "cpu"
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        free += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        device = "cuda" if free >= 2 * 1024 ** 3 else "cpu"
    try:
        _, _, f1 = bert_score_fn(predictions, references, model_type="bert-base-multilingual-cased",
                                 batch_size=32, device=device, verbose=False)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        _, _, f1 = bert_score_fn(predictions, references, model_type="bert-base-multilingual-cased",
                                 batch_size=32, device="cpu", verbose=False)
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()
    return float(f1.mean().item())


# ---------------------------------------------------------------- NER

def parse_entities(text: str) -> tuple[list[tuple[str, str]], bool]:
    """«ТИП: текст; ТИП: текст» или «нет» → ([(ТИП, текст)], разобрано ли). Мусорные фрагменты пропускаются."""
    text = str(text).strip()
    if not text or normalize(text) in (NO_ENTITIES, "none", "нет сущностей"):
        return [], bool(text)
    out, ok = [], True
    for part in re.split(r";|\n", text):
        if ":" not in part:
            ok = ok and not part.strip()
            continue
        typ, ent = part.split(":", 1)
        typ, ent = typ.strip().upper(), ent.strip()
        if typ and ent:
            out.append((typ, ent))
    return out, ok and bool(out)


def _ent_key(e: tuple[str, str]) -> tuple[str, str]:
    return e[0], normalize(e[1])


def ner_metrics(predictions: list[str], references: list[str]) -> tuple[dict[str, float], dict, float]:
    """Entity-level micro-P/R/F1: сущность верна при совпадении типа и нормализованного текста (мультимножества)."""
    tp = fp = fn = 0
    per_type: dict[str, Counter] = {}
    unparsed = 0
    for p, r in zip(predictions, references):
        pred, ok = parse_entities(p)
        unparsed += not ok
        gold = Counter(map(_ent_key, parse_entities(r)[0]))
        got = Counter(map(_ent_key, pred))
        hit = gold & got
        tp, fp, fn = tp + sum(hit.values()), fp + sum((got - hit).values()), fn + sum((gold - hit).values())
        for (typ, _), c in gold.items():
            per_type.setdefault(typ, Counter())["support"] += c
        for key, c in hit.items():
            per_type.setdefault(key[0], Counter())["tp"] += c
        for key, c in (got - hit).items():
            per_type.setdefault(key[0], Counter())["fp"] += c
        for key, c in (gold - hit).items():
            per_type.setdefault(key[0], Counter())["fn"] += c

    def prf(t, f_p, f_n):
        p = t / (t + f_p) if t + f_p else 0.0
        r = t / (t + f_n) if t + f_n else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    p, r, f = prf(tp, fp, fn)
    per_class = {}
    for typ, c in sorted(per_type.items()):
        tp_, fp_, fn_ = c["tp"], c["fp"], c["fn"]
        pp, rr, ff = prf(tp_, fp_, fn_)
        per_class[typ] = {"precision": pp, "recall": rr, "f1": ff, "support": int(c["support"])}
    return ({"entity_f1": f, "entity_precision": p, "entity_recall": r}, per_class,
            unparsed / max(len(predictions), 1))


# ---------------------------------------------------------------- QA (SQuAD)

def squad_normalize(s: str) -> str:
    s = str(s).lower().replace("ё", "е")
    s = "".join(ch for ch in s if ch not in set(string.punctuation) | {"«", "»", "—", "–", "…"})
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = squad_normalize(pred).split(), squad_normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def qa_metrics(predictions: list[str], references: list[str], alternatives: list[list[str]] | None
               ) -> dict[str, float]:
    em = f1 = 0.0
    for i, (p, r) in enumerate(zip(predictions, references)):
        golds = [g for g in ((alternatives[i] if alternatives else None) or [r]) if g] or [r]
        em += max(float(squad_normalize(p) == squad_normalize(g)) for g in golds)
        f1 += max(token_f1(p, g) for g in golds)
    n = max(len(predictions), 1)
    return {"exact_match": em / n, "token_f1": f1 / n}


def parse_alternatives(values) -> list[list[str]] | None:
    """Колонка alt (json-списки эталонов QA) → списки; None, если колонки нет."""
    if values is None:
        return None
    out = []
    for v in values:
        try:
            out.append(json.loads(v) if isinstance(v, str) else [])
        except ValueError:
            out.append([])
    return out


# ---------------------------------------------------------------- общий вход

EXPENSIVE_METRICS = {"bert_score_f1"}


def compute(fmt: str, names: list[str], predictions: list[str], references: list[str],
            labels: list[str] | None, cheap_only: bool = False, alternatives: list[list[str]] | None = None) -> dict:
    """Возвращает {"metrics", "per_class", "unparsed_fraction", "normalized_predictions"}."""
    if cheap_only:
        names = [n for n in names if n not in EXPENSIVE_METRICS]
    per_class: dict = {}
    norm = predictions
    if fmt == "label":
        metrics, per_class, unparsed, norm = classification_metrics(predictions, references, labels or [])
    elif fmt == "entities":
        metrics, per_class, unparsed = ner_metrics(predictions, references)
    elif fmt == "span":
        metrics = qa_metrics(predictions, references, alternatives)
        unparsed = sum(not p.strip() for p in predictions) / max(len(predictions), 1)
    else:
        metrics = generation_metrics(predictions, references, names)
        unparsed = sum(not p.strip() for p in predictions) / max(len(predictions), 1)
    return {"metrics": {k: v for k, v in metrics.items() if k in names}, "per_class": per_class,
            "unparsed_fraction": unparsed, "normalized_predictions": norm}


def example_correct(fmt: str, prediction: str, normalized: str, reference: str, alternatives: list[str] | None) -> bool:
    """Верен ли отдельный пример — для «верных» и «ошибочных» примеров в итоговом отчёте."""
    if fmt == "label":
        return normalized == reference
    if fmt == "entities":
        return Counter(map(_ent_key, parse_entities(prediction)[0])) == Counter(map(_ent_key, parse_entities(reference)[0]))
    if fmt == "span":
        return any(squad_normalize(prediction) == squad_normalize(g) for g in (alternatives or [reference]))
    return normalize(prediction) == normalize(reference)


def prediction_shares(normalized: list[str]) -> dict[str, float]:
    c = Counter(normalized)
    return {k: v / max(len(normalized), 1) for k, v in c.most_common()}


# ---------------------------------------------------------------- baseline'ы без модели

def _lead(text: str, max_chars: int) -> str:
    lead = ""
    for s in re.split(r"(?<=[.!?…])\s+", text):
        if lead and len(lead) + len(s) > max_chars:
            break
        lead = f"{lead} {s}".strip()
    return lead


def _gazetteer(train_targets: list[str]) -> list[tuple[str, str]]:
    """Словарь сущностей из train: текст → самый частый тип; длинные первыми (жадное сопоставление)."""
    seen: dict[str, Counter] = {}
    for t in train_targets:
        for typ, ent in parse_entities(t)[0]:
            seen.setdefault(ent, Counter())[typ] += 1
    return sorted(((ent, c.most_common(1)[0][0]) for ent, c in seen.items() if len(ent) > 1), key=lambda x: -len(x[0]))


def gazetteer_predict(text: str, gazetteer: list[tuple[str, str]]) -> str:
    found, taken = [], []
    for ent, typ in gazetteer:
        for m in re.finditer(rf"(?<!\w){re.escape(ent)}(?!\w)", text):
            if not any(a < m.end() and m.start() < b for a, b in taken):
                taken.append((m.start(), m.end()))
                found.append((m.start(), typ, ent))
    return "; ".join(f"{typ}: {ent}" for _, typ, ent in sorted(found)) or NO_ENTITIES


def best_sentence(qa_input: str) -> str:
    """Предложение контекста с наибольшим пересечением слов с вопросом."""
    m = re.match(r"Контекст:\s*(.*)\nВопрос:\s*(.*)$", qa_input, re.S)
    if not m:
        return qa_input[:200]
    context, question = m.group(1), m.group(2)

    def stems(text: str) -> set[str]:  # грубая основа: первые 5 букв — сглаживает падежи («Москве» ~ «Москва»)
        return {w[:5] for w in squad_normalize(text).split() if len(w) > 2}
    q = stems(question)
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", context) if s.strip()] or [context]
    return max(sentences, key=lambda s: len(q & stems(s)))


def simple_baseline_predictions(name: str, inputs: list[str], train_targets: list[str], target_lengths: list[int]) -> list[str] | None:
    """Baseline'ы без модели. base_prompt (исходная модель с промптом) считает рабочий процесс."""
    if name == "majority":
        top = Counter(train_targets).most_common(1)[0][0]
        return [top] * len(inputs)
    if name == "copy_input":
        return list(inputs)
    if name == "lead":
        avg = int(sum(target_lengths) / max(len(target_lengths), 1)) or 200
        return [_lead(t, avg) for t in inputs]
    if name == "gazetteer":
        gaz = _gazetteer(train_targets)
        return [gazetteer_predict(t, gaz) for t in inputs]
    if name == "best_sentence":
        return [best_sentence(t) for t in inputs]
    return None


def examples(inputs: list[str], refs: list[str], preds: list[str], correct: list[bool | None], k: int
             ) -> tuple[list[PredictionExample], list[PredictionExample]]:
    ok, bad = [], []
    for i, r, p, c in zip(inputs, refs, preds, correct):
        ex = PredictionExample(input=i[:500], target=r[:500], prediction=p[:500], correct=c)
        (ok if c else bad).append(ex)
    return ok[:k], bad[:k]
