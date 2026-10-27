"""Конфигурация приложения: DATA_DIR/config.yaml поверх значений по умолчанию из кода."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


class RoleConfig(BaseModel):
    model: str | None = Field(default=None, description="Имя модели на сервере; None — default_model из dev.env")
    temperature: float = 0.0
    max_tokens: int = 2000


class LLMConfig(BaseModel):
    roles: dict[str, RoleConfig] = Field(default_factory=lambda: {
        "orchestrator": RoleConfig(),
        "intake": RoleConfig(),
        "dataprep": RoleConfig(max_tokens=6000),  # ReAct: длинные рассуждения и аргументы инструментов
        "hparam": RoleConfig(max_tokens=6000),
        "monitor": RoleConfig(max_tokens=4000),
        "query_expansion": RoleConfig(temperature=0.3, max_tokens=300),
        "dataset_eval": RoleConfig(max_tokens=1200),  # оценки + исправленное сопоставление колонок
        "augment": RoleConfig(temperature=0.8, max_tokens=3000),
        "model_scout": RoleConfig(max_tokens=600),  # факты из карточки модели HF (search/models_hub)
    })
    max_retries: int = 3
    timeout_s: float = 300.0  # сервер под нагрузкой отвечал до ~2 мин на запрос
    disable_thinking: bool = True


class SearchConfig(BaseModel):
    embedding_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rrf_k: int = 60
    top_rerank: int = 50
    top_llm: int = 10
    catalog_limit: int = 3000
    min_lang_examples: int = 100
    llm_examples: int = 5
    compat_timeout_s: float = 240.0  # на проверку совместимости всех кандидатов; не успевшие пропускаются
    auto_choose_min_score: float = 60.0  # Авто берёт лучший датасет сам, только если его оценка не ниже (из 100)
    # ⚠️ веса критериев LLM-оценки (сумма = 1)
    criteria_weights: dict[str, float] = Field(default_factory=lambda: {
        "relevance": 0.30, "label_fit": 0.20, "size": 0.10,
        "text_quality": 0.10, "domain_fit": 0.15, "label_reliability": 0.15,
    })


class PrepRules(BaseModel):
    max_empty_share: float = 0.01
    max_dupe_share: float = 0.02
    max_imbalance: float = 10.0
    min_lang_share: float = 0.8
    max_target_tokens_p99: int = 2048
    max_near_dupe_share: float = 0.02
    max_junk_share: float = 0.01
    max_length_outlier_share: float = 0.01
    max_label_noise_share: float = 0.05


class DataPrepConfig(BaseModel):
    proportions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    sample_size: int = 3000
    max_label_noise_fraction: float = 0.05
    confirm_removed_fraction: float = 0.2
    rules: PrepRules = Field(default_factory=PrepRules)
    request_limit: int = 60   # ReAct: агент итерирует, пока цели качества не выполнены
    tool_calls_limit: int = 90
    augment_max_rows: int = 2000
    augment_confidence: float = 0.8
    augment_workers: int = 4


class HParamRanges(BaseModel):
    lr: tuple[float, float] = (1e-6, 5e-3)
    epochs: tuple[int, int] = (1, 20)
    lora_r: tuple[int, int] = (2, 128)
    lora_alpha_ratio: tuple[float, float] = (0.5, 4.0)
    num_virtual_tokens: tuple[int, int] = (4, 128)
    encoder_hidden_size: tuple[int, int] = (32, 2048)


class PilotConfig(BaseModel):
    fraction: float = 0.02
    min_steps: int = 20
    max_steps: int = 60
    max_fix_attempts: int = 2


class HParamConfig(BaseModel):
    strategy: Literal["heuristic", "agent"] = "heuristic"
    ranges: HParamRanges = Field(default_factory=HParamRanges)
    pilot: PilotConfig = Field(default_factory=PilotConfig)
    vram_safety: float = 0.85
    max_seq_len_cap: int = 2048
    request_limit: int = 20
    tool_calls_limit: int = 20


class DetectorThresholds(BaseModel):
    """⚠️ пороги детекторов"""
    window: int = 5
    plateau_min_delta: float = 0.002
    plateau_evals: int = 3
    overfit_evals: int = 3
    grad_explosion_factor: float = 10.0
    grad_explosion_abs: float = 100.0
    throughput_drop: float = 0.5
    oom_risk_share: float = 0.95
    collapse_share: float = 0.9
    loss_spike_factor: float = 3.0


class TrainingConfig(BaseModel):
    detectors: DetectorThresholds = Field(default_factory=DetectorThresholds)
    heartbeat_timeout_s: float = 900.0
    poll_interval_s: float = 5.0
    monitor_cooldown_steps: int = 200
    monitor_check_every_evals: int = 3   # плановая проверка A4 каждые K оценок на val (0 — только по аномалиям)
    monitor_ack_timeout_s: float = 90.0  # сколько ждать подтверждения команды процессом
    default_extra_epochs: int = 2
    min_round_gain: float = 0.005
    num_eval_samples: int = 200
    # длина генерации при оценке — по данным (p95 длины ответа × 1.3); это лишь предохранитель от зацикливания:
    # обрезанный ответ против полного эталона занижает метрики и искажает сравнение с baseline'ами
    max_new_tokens: int = 1024
    # время оценки ограничивается числом примеров, а не длиной ответа: сгенерированных токенов на проход
    eval_token_budget: int = 40_000  # val во время обучения (оценок много): метки — все 200, длинные ответы — ~60
    final_eval_token_budget: int = 100_000  # test: метки — 2000 примеров, длинные ответы — ~200
    min_eval_samples: int = 50
    test_examples_shown: int = 5


class AppConfig(BaseModel):
    data_dir: Path = DEFAULT_DATA_DIR
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    dataprep: DataPrepConfig = Field(default_factory=DataPrepConfig)
    hparams: HParamConfig = Field(default_factory=HParamConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mlflow_tracking_dir: str = "mlruns"

    # пути внутри DATA_DIR (§5)
    @property
    def registry_dir(self) -> Path:
        return self.data_dir / "registry"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache" / "datasets"

    @property
    def hparam_docs_dir(self) -> Path:
        return self.data_dir / "docs" / "hparams"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def locks_dir(self) -> Path:
        return self.data_dir / "locks"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(data_dir: Path | str | None = None) -> AppConfig:
    """Читает DATA_DIR/config.yaml (если есть) и накладывает на значения по умолчанию.

    DATA_DIR берётся из аргумента, переменной FACTORY_DATA_DIR или ./data в корне проекта.
    HPARAM_STRATEGY из окружения перекрывает hparams.strategy.
    """
    data_dir = Path(data_dir or os.environ.get("FACTORY_DATA_DIR") or DEFAULT_DATA_DIR).resolve()
    raw = AppConfig(data_dir=data_dir).model_dump(mode="json")
    path = data_dir / "config.yaml"
    if path.exists():
        raw = _deep_merge(raw, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    raw["data_dir"] = str(data_dir)
    if os.environ.get("HPARAM_STRATEGY"):
        raw["hparams"]["strategy"] = os.environ["HPARAM_STRATEGY"]
    return AppConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_config()
