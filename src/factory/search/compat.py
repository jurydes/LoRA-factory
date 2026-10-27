"""Детерминированная проверка совместимости датасета с задачей: колонки, метки, лицензия, язык."""
from __future__ import annotations

import logging

import pandas as pd

from factory.core.contracts import CompatReport, DatasetCandidate, MapSchemaOp, TaskSpec
from factory.core.registry import TaskInfo
from factory.prep import lang
from factory.prep.schema import (
    MAX_CLASSES,
    drop_unusable,
    flatten_dicts,
    infer_schema,
    normalize_label,
    phrases,
    profile_columns,
    to_task_frame,
)

logger = logging.getLogger(__name__)

COMMERCIAL_LICENSES = {
    "apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "cc-by-2.0", "cc-by-3.0",
    "cc-by-4.0", "cc-by-sa-3.0", "cc-by-sa-4.0", "odc-by", "odbl", "pddl", "unlicense", "openrail",
}
PREVIEW_ROWS = 5  # для LLM-оценки (cfg.search.llm_examples) и карточки датасета


def check_license(license_id: str | None, commercial: bool) -> tuple[bool, list[str]]:
    if not commercial:
        return True, [] if license_id else ["лицензия не указана"]
    if not license_id:
        return False, ["лицензия не указана, а нужна коммерческая"]
    if license_id.lower() not in COMMERCIAL_LICENSES:
        return False, [f"лицензия {license_id} не разрешает коммерческое использование (или неизвестна)"]
    return True, []


REQUIRED = {
    "ner_bio": [("input_column", "колонка токенов"), ("target_column", "колонка BIO-тегов")],
    "ner_spans": [("input_column", "колонка текста"), ("target_column", "колонка сущностей")],
    "qa": [("input_column", "колонка контекста"), ("question_column", "колонка вопроса"), ("target_column", "колонка ответов")],
    "chat": [("input_column", "колонка диалога")],
    "plain": [("input_column", "текстовая колонка входа"), ("target_column", "колонка цели (метка/ответ)")],
}


def check_compat(
    spec: TaskSpec,
    task: TaskInfo,
    sample: pd.DataFrame,
    label_names: dict[str, list[str]] | None = None,
    license_id: str | None = None,
    has_loading_script: bool = False,
    schema: dict | None = None,
) -> CompatReport:
    """schema — готовое сопоставление (правка LLM-оценки); без него колонки определяются по данным."""
    issues: list[str] = []
    if has_loading_script:
        return CompatReport(ok=False, license=license_id, issues=["датасет со скриптом загрузки не поддерживается"])
    if sample.empty:
        return CompatReport(ok=False, license=license_id, issues=["не удалось получить примеры"])

    license_ok, lic_issues = check_license(license_id, spec.constraints.commercial_license)
    issues += lic_issues
    schema = schema or infer_schema(sample, task.format, task.family, label_names, spec.labels, spec.language,
                                    spec.description, task.key)
    issues += schema.get("notes", [])
    missing = [name for key, name in REQUIRED[schema["kind"]] if not schema.get(key)]
    issues += [f"не найдена {name}" for name in missing]
    ok = license_ok and not missing
    fields = {k: schema.get(k) for k in ("input_column", "target_column", "question_column")}
    fields |= {k: schema.get(k) or default for k, default in
               (("extra_input_columns", []), ("options_columns", []), ("language_filter", {}))}
    needs_mapping = license_ok and bool(missing) and schema["kind"] in ("plain", "qa") and bool(
        free_columns(sample, fields, task.format == "label" and not fields["target_column"]))
    if schema["kind"] == "chat":
        fields["target_column"] = fields["input_column"]

    label_values: list[str] = []
    overlap = None
    label_map: dict[str, str] = {}
    frame = None
    if ok:
        try:  # пробное приведение к формату задачи ловит несовместимые данные до выбора датасета
            frame = to_task_frame(sample, MapSchemaOp(kind=schema["kind"], **fields), task.format)
        except Exception as e:
            ok = False
            issues.append(f"не удалось привести к формату задачи: {type(e).__name__}: {e}")
    if ok:
        usable, dropped = drop_unusable(frame)
        if fields["language_filter"] and frame.empty:
            ok = False
            issues.append(f"в выборке нет строк на языке {spec.language} "
                          f"(колонка {', '.join(fields['language_filter'])})")
        elif len(usable) < 0.5 * len(frame) or usable.empty:
            ok = False
            issues.append(f"у {dropped} из {len(frame)} примеров нет входа или ответа")
        frame = usable
    if ok and task.format == "label":
        names = (label_names or {}).get(fields["target_column"])
        present = frame["target"].unique().tolist()
        label_values = [str(v) for v in (names or present)]
        if len(present) < 2:
            ok = False
            issues.append("в колонке меток меньше двух значений")
        elif len(label_values) > MAX_CLASSES:
            ok = False
            issues.append(f"слишком много различных меток ({len(label_values)})")
    elif ok and task.format == "entities":
        label_values = sorted({t for v in frame["target"] for t, _ in parse_entities_loose(v)})
        if not label_values:
            ok = False
            issues.append("в выборке нет ни одной размеченной сущности")
    if ok and spec.labels and label_values:
        have = {normalize_label(v) for v in label_values}
        overlap = sum(normalize_label(l) in have for l in spec.labels) / len(spec.labels)
        label_map = auto_label_map(label_values, spec.labels)
        if overlap < 1:
            issues.append(f"совпало {overlap:.0%} меток постановки; остальные потребуют сопоставления (label_map)")
    preview = []
    if frame is not None and ok:
        # сначала по примеру на каждый ответ (разные метки классов), затем остальные
        rows = pd.concat([frame.drop_duplicates("target"), frame]).drop_duplicates("_id").head(PREVIEW_ROWS)
        preview = [{"input": r.input[:400], "target": r.target[:200]} for r in rows.itertuples()]

    return CompatReport(
        ok=ok, schema_kind=schema["kind"], **fields, needs_mapping=needs_mapping, preview=preview,
        label_values=label_values[:MAX_CLASSES], label_overlap=overlap,
        label_map=label_map if task.format == "label" else {}, license=license_id, license_ok=license_ok, issues=issues)


def free_columns(sample: pd.DataFrame, fields: dict, category_ok: bool) -> list[str]:
    """Колонки, не занятые сопоставлением, которыми можно закрыть недостающую роль: текст (или категория,
    если не хватает колонки меток). Нет таких — LLM исправлять нечего."""
    used = {fields.get(k) for k in ("input_column", "target_column", "question_column")}
    used |= {*fields.get("extra_input_columns", []), *fields.get("options_columns", []), *fields.get("language_filter", {})}
    return [c.name for c in profile_columns(flatten_dicts(sample)) if c.name not in used
            and (c.is_text or phrases(c) or (category_ok and c.kind == "category" and not c.meta))]


def schema_op(compat: CompatReport, label_map: dict[str, str] | None = None) -> MapSchemaOp:
    return MapSchemaOp(kind=compat.schema_kind, input_column=compat.input_column, target_column=compat.target_column,
                       question_column=compat.question_column, extra_input_columns=compat.extra_input_columns,
                       options_columns=compat.options_columns, language_filter=compat.language_filter,
                       label_map=label_map or None)


def schema_of(compat: CompatReport) -> dict:
    """Сопоставление колонок из отчёта — в виде результата infer_schema (для повторной проверки с правкой)."""
    return {"kind": compat.schema_kind, "input_column": compat.input_column, "target_column": compat.target_column,
            "question_column": compat.question_column, "extra_input_columns": list(compat.extra_input_columns),
            "options_columns": list(compat.options_columns), "language_filter": dict(compat.language_filter)}


def parse_entities_loose(text: str) -> list[tuple[str, str]]:
    from factory.training.metrics import parse_entities
    return parse_entities(text)[0]


def auto_label_map(dataset_labels: list[str], spec_labels: list[str]) -> dict[str, str]:
    """Сопоставление по нормализованному написанию (регистр, пробелы, пунктуация)."""
    by_norm = {normalize_label(l): l for l in spec_labels}
    return {v: by_norm[normalize_label(v)] for v in dataset_labels if normalize_label(v) in by_norm}


def checked_label_map(proposed: dict[str, str], dataset_labels: list[str], spec_labels: list[str]) -> dict[str, str]:
    """Предложение LLM проходит проверку: ключи — реальные метки датасета, значения — метки постановки."""
    ds, sp = set(map(str, dataset_labels)), set(spec_labels)
    return {str(k): v for k, v in proposed.items() if str(k) in ds and v in sp}


def estimate_lang_examples(texts: list[str], language: str, total_rows: int) -> int | None:
    """Оценка числа примеров на нужном языке экстраполяцией с выборки (как count_russian в курсовой)."""
    if not texts:
        return None
    return int(lang.language_share(texts).get(language, 0.0) * total_rows)


def sha256_short(path) -> str:
    """«Ревизия» файла пользователя — хэш содержимого."""
    from factory.core.storage import sha256_file
    return sha256_file(path)[:16]


def hf_metadata(hf_id: str) -> dict:
    """Ревизия (sha), лицензия, размеры сплитов и наличие скрипта загрузки — без скачивания данных."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(hf_id, files_metadata=False)
    files = [s.rfilename for s in (info.siblings or [])]
    card = info.card_data.to_dict() if info.card_data else {}
    split_sizes: dict[str, int] = {}
    ds_info = card.get("dataset_info")
    for block in (ds_info if isinstance(ds_info, list) else [ds_info] if ds_info else []):
        for split in block.get("splits", []) or []:
            split_sizes[split["name"]] = split_sizes.get(split["name"], 0) + int(split.get("num_examples", 0))
    if not split_sizes:
        try:
            from datasets import load_dataset_builder
            builder = load_dataset_builder(hf_id, revision=info.sha)
            if builder.info.splits:
                split_sizes = {k: int(v.num_examples) for k, v in builder.info.splits.items()}
        except Exception as e:
            logger.debug("load_dataset_builder(%s): %s", hf_id, e)
    lic = card.get("license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    script_name = hf_id.split("/")[-1] + ".py"
    return {
        "revision": info.sha,
        "license": lic,
        "split_sizes": split_sizes,
        "has_loading_script": script_name in files,
        "gated": bool(getattr(info, "gated", False)),
    }


def candidate_from_sample(
    hf_id: str, spec: TaskSpec, task: TaskInfo, sample: pd.DataFrame, meta: dict, description: str = "",
    source: str = "hf", local_path: str | None = None, min_lang_examples: int = 0,
) -> DatasetCandidate:
    compat = check_compat(spec, task, sample, meta.get("label_names"), meta.get("license"), meta.get("has_loading_script", False))
    sizes = meta.get("split_sizes", {})
    if meta.get("splits"):
        sizes = {k: v for k, v in sizes.items() if k in meta["splits"]}
    total = sum(sizes.values())
    # размер неизвестен — оценки числа примеров на языке нет, и отказывать по ней нельзя
    lang_n = None
    if compat.ok and total:  # строки других языков и без ответа отпадут при загрузке — оценка по оставшимся
        frame, _ = drop_unusable(to_task_frame(sample, schema_op(compat), task.format))
        lang_n = estimate_lang_examples(frame["input"].tolist(), spec.language, round(total * len(frame) / len(sample)))
    if compat.ok and lang_n is not None and lang_n < min_lang_examples:
        compat = compat.model_copy(update={
            "ok": False, "issues": compat.issues + [f"мало примеров на языке {spec.language}: ~{lang_n}"]})
    return DatasetCandidate(
        hf_id=hf_id, source=source, local_path=local_path, revision=meta.get("revision"),
        config_name=meta.get("config_name"), splits=meta.get("splits"),
        description=description, license=meta.get("license"), split_sizes=sizes,  # только сплиты, которые возьмём
        columns=[str(c) for c in sample.columns], lang_examples=lang_n, compat=compat,
    )
