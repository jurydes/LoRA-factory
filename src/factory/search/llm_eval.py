"""LLM-оценка финалистов по критериям (вызов без состояния). Итоговый скор считает код.

LLM видит все колонки датасета и примеры уже в формате задачи («вход → ответ») — и может исправить сопоставление
колонок. Правка проходит ту же проверку совместимости, что и автоматическое сопоставление."""
from __future__ import annotations

import json
import logging

import pandas as pd
from pydantic import BaseModel, Field

from factory.core import llm
from factory.core.contracts import CriterionScore, DatasetCandidate, TaskSpec
from factory.core.registry import TaskInfo
from factory.prep.schema import flatten_dicts, profile_columns

logger = logging.getLogger(__name__)

CRITERIA = {
    "relevance": "релевантность задаче",
    "label_fit": "соответствие меток/формата ответа постановке",
    "size": "достаточность размера",
    "text_quality": "качество текста",
    "domain_fit": "соответствие домену",
    "label_reliability": "надёжность разметки",
}

KIND_NAMES = {
    "text": "текст", "category": "категория", "constant": "одно значение", "url": "ссылки", "number": "число",
    "date": "дата", "dict": "словарь", "str_list": "список строк", "int_list": "список чисел",
    "float_list": "вектор чисел", "messages": "диалог", "spans": "сущности с позициями", "dict_list": "список словарей",
    "empty_list": "пустые списки", "empty": "пусто", "list": "список", "mixed": "разные типы", "other": "другое",
}


class MappingFix(BaseModel):
    input_columns: list[str] = Field(description="Колонки входа в порядке подачи модели (для QA — колонка контекста)")
    target_column: str = Field(description="Колонка ответа модели (для классификации — колонка меток)")
    question_column: str | None = Field(default=None, description="Только для QA: колонка вопроса")


class DatasetAssessment(BaseModel):
    relevance: CriterionScore
    label_fit: CriterionScore
    size: CriterionScore
    text_quality: CriterionScore
    domain_fit: CriterionScore
    label_reliability: CriterionScore
    why: str = Field(description="1–2 предложения для пользователя: почему датасет подходит (или чем плох)")
    label_mapping: dict[str, str] = Field(default_factory=dict, description=(
        "Только для классификации: метка датасета → метка постановки, если по примерам смысл однозначен "
        "(например, '0' → 'нейтрально'). Пусто, если не уверен."))
    mapping_ok: bool = Field(default=True, description=(
        "Верно ли сопоставление колонок: во входе всё нужное для ответа и нет самого ответа, ответ — то, "
        "что модель должна выдавать по задаче"))
    mapping_fix: MappingFix | None = Field(default=None, description=(
        "Только если mapping_ok = false или сопоставление не определено: правильные колонки из списка колонок"))


INSTRUCTIONS = """Ты оцениваешь, подходит ли датасет для дообучения модели под задачу пользователя.
Оцени каждый критерий целым числом 0–5 и кратко обоснуй (одно предложение):
""" + "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items()) + """
0 — совсем не подходит, 5 — идеально. Опирайся только на описание, колонки и примеры; не выдумывай факты.
Общие датасеты для претрейна без разметки под задачу получают 0–1 по relevance и label_fit.
Проверь сопоставление колонок по примерам «вход → ответ»: модель будет учиться по входу выдавать ответ. Если во
входе не хватает нужного (вопроса, контекста, инструкции), в нём есть сам ответ, или ответ взят не из той колонки —
mapping_ok = false и в mapping_fix укажи правильные колонки (только имена из списка колонок). Оценки ставь с учётом
правильного сопоставления."""


def _value(v, limit: int = 120) -> str:
    s = json.dumps(v, ensure_ascii=False, default=str) if not isinstance(v, str) else v
    return s[:limit] + ("…" if len(s) > limit else "")


def describe_columns(sample: pd.DataFrame) -> list[str]:
    """Колонки с видом содержимого и примером значения — чтобы LLM могла проверить сопоставление."""
    out = []
    for c in profile_columns(flatten_dicts(sample)):
        kind = KIND_NAMES.get(c.kind, c.kind)
        if c.kind == "category":
            kind += f", {len(set(map(str, c.values)))} значений"
        out.append(f"- {c.name} ({kind}{', заполнено ' + format(c.fill, '.0%') if c.fill < 0.95 else ''}): "
                   f"{_value(c.values[0]) if c.values else '—'}")
    return out


def describe_mapping(cand: DatasetCandidate) -> str:
    c = cand.compat
    if c is None or not c.input_column or not c.target_column:
        missing = "; ".join(i for i in (c.issues if c else []) if i.startswith("не найдена")) or "колонки не найдены"
        return f"не определено автоматически ({missing}) — предложи mapping_fix"
    if c.schema_kind == "chat":
        return f"диалог из колонки {c.input_column}: вход — реплики пользователя до первого ответа, ответ — ответ ассистента"
    inputs = [c.input_column, *c.extra_input_columns] + ([c.question_column] if c.question_column else [])
    text = f"вход — {' + '.join(inputs)}; ответ — {c.target_column}"
    if c.options_columns:
        text += f" (ключ правильного варианта из {', '.join(c.options_columns)} заменён текстом варианта)"
    if c.language_filter:
        text += f"; только строки на языке {', '.join(sorted(set(c.language_filter.values())))} (по {', '.join(c.language_filter)})"
    return text


def build_prompt(spec: TaskSpec, cand: DatasetCandidate, sample: pd.DataFrame, n_examples: int) -> str:
    compat = cand.compat
    if compat and compat.preview:
        examples = [{"вход": e["input"], "ответ": e["target"]} for e in compat.preview[:n_examples]]
    else:
        examples = sample.head(min(n_examples, 3)).astype(str).map(lambda s: s[:300]).to_dict(orient="records")
    return "\n".join([
        f"Задача: {spec.description}",
        f"Тип задачи: {spec.task_type}; язык: {spec.language}; метки: {spec.labels or '—'}",
        f"Датасет: {cand.hf_id}",
        f"Описание: {cand.description[:1500] or '—'}",
        f"Размеры сплитов: {cand.split_sizes or 'неизвестно'}; примеров на нужном языке: ~{cand.lang_examples}",
        "Колонки (вид, пример значения):",
        *describe_columns(sample),
        f"Сопоставление колонок: {describe_mapping(cand)}",
        f"Значения меток в датасете: {compat.label_values[:30] if compat and compat.label_values else '—'}",
        f"Метки постановки: {spec.labels or '—'}",
        "Примеры в формате задачи:" if compat and compat.preview else "Строки датасета:",
        json.dumps(examples, ensure_ascii=False, indent=1),
    ])


def weighted_score(assessment: DatasetAssessment, weights: dict[str, float]) -> float:
    """Взвешенная сумма оценок 0–5 → 0–100. Веса нормируются на сумму."""
    total_w = sum(weights.get(k, 0.0) for k in CRITERIA) or 1.0
    s = sum(weights.get(k, 0.0) * getattr(assessment, k).score for k in CRITERIA) / total_w
    return round(s / 5 * 100, 1)


def apply_mapping_fix(spec: TaskSpec, task: TaskInfo, cand: DatasetCandidate, sample: pd.DataFrame,
                      fix: MappingFix):
    """Правка сопоставления от LLM → новый отчёт совместимости или None, если правка не проходит проверку:
    колонки должны существовать, ответ — не из колонок входа, пробное приведение — давать примеры."""
    from factory.search.compat import check_compat, schema_of

    compat = cand.compat
    if compat is None or compat.schema_kind not in ("plain", "qa"):
        return None
    columns = set(map(str, flatten_dicts(sample).columns))
    names = [*fix.input_columns, fix.target_column] + ([fix.question_column] if fix.question_column else [])
    if not fix.input_columns or any(n not in columns for n in names) or fix.target_column in fix.input_columns:
        logger.info("Правка сопоставления %s отклонена: колонки %s", cand.hf_id, names)
        return None
    schema = schema_of(compat)
    schema.update(input_column=fix.input_columns[0], target_column=fix.target_column, notes=[])
    if compat.schema_kind == "qa":
        schema["question_column"] = fix.question_column or compat.question_column
    else:
        schema["extra_input_columns"] = fix.input_columns[1:]
    if fix.target_column != compat.target_column:  # варианты ответа относились к прежнему ключу
        schema["options_columns"] = []
    if cand.source == "user":  # лицензию своего файла пользователь знает сам (как в evaluate_user_dataset)
        spec = spec.model_copy(update={"constraints": spec.constraints.model_copy(update={"commercial_license": False})})
    report = check_compat(spec, task, sample, license_id=cand.license, schema=schema)
    if not report.ok:
        logger.info("Правка сопоставления %s не прошла проверку: %s", cand.hf_id, report.issues)
        return None
    return report.model_copy(update={"issues": ["сопоставление колонок исправлено по LLM-оценке"] + report.issues})


async def assess(spec: TaskSpec, task: TaskInfo, cand: DatasetCandidate, sample: pd.DataFrame,
                 weights: dict[str, float], n_examples: int = 5) -> DatasetCandidate:
    async def ask(c: DatasetCandidate) -> DatasetAssessment:
        return await llm.complete("dataset_eval", build_prompt(spec, c, sample, n_examples),
                                  output_type=DatasetAssessment, instructions=INSTRUCTIONS)
    result = await ask(cand)
    compat = cand.compat
    if result.mapping_fix and compat and (not result.mapping_ok or compat.needs_mapping):
        fixed = apply_mapping_fix(spec, task, cand, sample, result.mapping_fix)
        if fixed:  # оценки первого ответа — по неверным примерам: переоцениваем по исправленным
            compat = fixed
            try:
                result = await ask(cand.model_copy(update={"compat": fixed}))
            except Exception as e:
                logger.warning("Переоценка %s после правки сопоставления не удалась: %s", cand.hf_id, e)
    if compat and spec.labels and result.label_mapping:
        from factory.search.compat import checked_label_map
        llm_map = checked_label_map(result.label_mapping, compat.label_values, spec.labels)
        # автоматическое совпадение написания приоритетнее предложения LLM
        compat = compat.model_copy(update={"label_map": llm_map | compat.label_map})
    return cand.model_copy(update={
        "compat": compat,
        "llm_scores": {k: getattr(result, k) for k in CRITERIA},
        "score": weighted_score(result, weights),
        "why": result.why,
    })
