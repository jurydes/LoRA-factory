"""Реестры задач (registry/tasks.yaml) и базовых моделей (registry/models.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from factory.core.config import AppConfig, get_config
from factory.core.contracts import Constraints, TaskFamily, TaskSpec


class TaskInfo(BaseModel):
    key: str
    title: str
    family: TaskFamily
    format: Literal["label", "text", "entities", "span"] = "text"
    supported: bool = True
    metrics: list[str]
    primary_metric: str
    evaluation: str = ""
    baselines: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    id: str
    params_b: float
    hidden_size: int
    num_layers: int
    vocab_size: int = 152_000
    max_context: int
    languages: list[str] = Field(default_factory=list)
    license: str = ""
    commercial_ok: bool = False
    gated: bool = False  # нужен HF_TOKEN с принятой лицензией
    lora_modules: dict[Literal["attention", "all_linear"], list[str]]
    unsloth: bool = False
    ptuning: bool = True
    chat_template: bool = False
    quality: float = 0.0  # MMLU-Pro без рассуждений (карточки моделей): по нему выбирается рекомендованная модель; 0 — неизвестно
    quality_note: str = ""  # откуда число (для моделей с HF — что выписано из карточки)
    text_only: bool = False  # чекпоинт мультимодальный: обучается только текстовая часть
    active_params_b: float | None = None  # MoE: параметры, работающие на шаге (время обучения); память — по params_b
    kv_bytes_per_token: int | None = None  # KV-кэш генерации на токен (bf16, слои с кэшем); None — оценка по hidden_size
    prequantized: bool = False  # веса в репозитории уже в 4 битах (bitsandbytes): только QLoRA, скачивание вчетверо меньше
    base_model: str | None = None  # для квантованной копии — исходная модель
    source: Literal["registry", "hub", "local"] = "registry"


def catalog_path(data_dir: Path) -> Path:
    return data_dir / "cache" / "models" / "catalog.json"


class Registry:
    def __init__(self, tasks: dict[str, TaskInfo], models: list[ModelInfo], offline: bool = False):
        self.tasks = tasks
        self.models = {m.id: m for m in models}
        self.offline = offline

    @classmethod
    def load(cls, cfg: AppConfig | None = None) -> "Registry":
        """Реестр + модели, подобранные с HF Hub под эту машину (кэш search/models_hub)."""
        cfg = cfg or get_config()
        reg = cls.from_dir(cfg.registry_dir)
        reg.offline = cfg.offline
        if not cfg.offline:
            reg.add_models(load_catalog(cfg.data_dir))
        reg.add_models(load_local_models(cfg.registry_dir))
        return reg

    def add_models(self, models: list[ModelInfo]) -> None:
        """Модели каталога HF; записи реестра не перекрываются — они проверены вручную."""
        for m in models:
            if m.source == "local":
                self.models[m.id] = m
            else:
                self.models.setdefault(m.id, m)

    @classmethod
    def from_dir(cls, registry_dir: Path) -> "Registry":
        raw_tasks = yaml.safe_load((registry_dir / "tasks.yaml").read_text(encoding="utf-8"))
        raw_models = yaml.safe_load((registry_dir / "models.yaml").read_text(encoding="utf-8"))
        tasks = {k: TaskInfo(key=k, **v) for k, v in raw_tasks.items()}
        return cls(tasks, [ModelInfo(**m) for m in raw_models])

    def task(self, key: str) -> TaskInfo:
        if key not in self.tasks:
            raise KeyError(f"Неизвестный тип задачи: {key}. Доступны: {', '.join(self.tasks)}")
        return self.tasks[key]

    def supported_tasks(self) -> list[TaskInfo]:
        return [t for t in self.tasks.values() if t.supported]

    def model(self, model_id: str) -> ModelInfo:
        if model_id not in self.models:
            raise KeyError(f"Модели {model_id} нет в реестре")
        return self.models[model_id]

    def filter_models(self, constraints: Constraints, language: str | None = None) -> list[ModelInfo]:
        """Модели, подходящие под ограничения постановки. Из этого списка LLM может выбирать."""
        out = []
        for m in self.models.values():
            if self.offline and m.source != "local":
                continue
            if not self.offline and m.source == "local":
                continue
            if constraints.max_model_params_b and m.params_b > constraints.max_model_params_b:
                continue
            if constraints.commercial_license and not m.commercial_ok:
                continue
            if constraints.adapter_method == "ptuning" and not m.ptuning:
                continue
            if language and language not in m.languages and "multi" not in m.languages:
                continue
            out.append(m)
        return sorted(out, key=lambda m: m.params_b)

    def available_models(self, constraints: Constraints, language: str | None, hw=None, safety: float = 0.85) -> list[dict]:
        """Модели под ограничения постановки и железо машины, с оценкой пригодности и рекомендацией.

        Без GPU (или без сведений о железе) фильтруются только ограничения постановки."""
        from factory.hparams.resources import rank_models
        models = self.filter_models(constraints, language)
        if hw is None or not hw.has_gpu:
            return [{"id": m.id, "params_b": m.params_b, "quality": m.quality, "fits": True, "mode": None,
                     "recommended": False} for m in models]
        if constraints.adapter_method == "lora":  # явный выбор LoRA — без квантизации; QLoRA-only не подходит
            rows = [r for r in rank_models(models, hw, safety, constraints.gpu_hours, language) if r["mode"] == "lora"]
        else:
            rows = rank_models(models, hw, safety, constraints.gpu_hours, language,
                               prefer_lora=constraints.adapter_method != "qlora")
        return [r for r in rows if r["fits"]]

    def apply_task_metrics(self, spec: TaskSpec) -> TaskSpec:
        """Подставляет метрики из реестра поверх ответа LLM и проверяет обязательные поля."""
        task = self.task(spec.task_type)
        if not task.supported:
            raise ValueError(f"Тип задачи {task.title} пока не поддерживается")
        if task.family == "classification" and not spec.labels:
            raise ValueError("Для классификации нужен набор меток")
        # для NER labels — типы сущностей (необязательны: возьмутся из данных)
        if spec.labels:
            labels = list(dict.fromkeys(str(l).strip() for l in spec.labels if str(l).strip()))
            if task.family == "classification" and len(labels) < 2:
                raise ValueError("Для классификации нужно минимум две метки")
        else:
            labels = None
        return spec.model_copy(update={
            "primary_metric": task.primary_metric,
            "metrics": list(task.metrics),
            "labels": labels if task.family == "classification" or task.format == "entities" else spec.labels,
        })


def load_catalog(data_dir: Path) -> list[ModelInfo]:
    """Модели с HF Hub из кэша каталога; битый или старый формат — пусто (реестр работает и без каталога)."""
    from factory.core.storage import read_json
    raw = read_json(catalog_path(data_dir), {}) or {}
    out = []
    for m in raw.get("models", []):
        try:
            out.append(ModelInfo.model_validate(m))
        except Exception:
            continue
    return out


def load_local_models(registry_dir: Path) -> list[ModelInfo]:
    path = registry_dir / "local_models.yaml"
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    models = [ModelInfo.model_validate(m | {"source": "local"}) for m in raw]
    for model in models:
        if not Path(model.id).is_dir():
            raise FileNotFoundError(f"Каталог зарегистрированной локальной модели не найден: {model.id}")
    return models
