"""Контракты между модулями (§4 архитектуры). Единственное место, где описаны структуры данных сценария."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- постановка

AdapterMethod = Literal["auto", "lora", "qlora", "ptuning"]
TaskFamily = Literal["classification", "generation"]


class Constraints(_Model):
    gpu_hours: float | None = Field(default=None, gt=0, description="Бюджет обучения в GPU-часах")
    max_model_params_b: float | None = Field(default=None, gt=0, description="Максимальный размер модели, млрд параметров")
    max_latency_ms: int | None = Field(default=None, gt=0)
    target_metric_value: float | None = Field(default=None, description="Целевое значение основной метрики")
    commercial_license: bool = Field(default=False, description="Нужна лицензия, разрешающая коммерческое использование")
    adapter_method: AdapterMethod = "auto"


class TaskSpec(_Model):
    description: str = Field(description="Постановка задачи своими словами (1–3 предложения)")
    task_type: str = Field(description="Ключ из registry/tasks.yaml")
    language: str = Field(default="ru", description="ISO-код языка данных")
    labels: list[str] | None = Field(default=None, description="Набор меток (обязателен для классификации)")
    base_model_id: str
    primary_metric: str = ""
    metrics: list[str] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    confirmed_by_user: bool = False


# ---------------------------------------------------------------- датасеты

SchemaKind = Literal["plain", "ner_bio", "ner_spans", "qa", "chat"]


class CompatReport(_Model):
    ok: bool
    schema_kind: SchemaKind = "plain"
    input_column: str | None = None
    target_column: str | None = None
    question_column: str | None = None
    extra_input_columns: list[str] = Field(default_factory=list)
    options_columns: list[str] = Field(default_factory=list)
    language_filter: dict[str, str] = Field(default_factory=dict)
    needs_mapping: bool = Field(default=False, description="Мешает только то, что колонки не распознаны: "
                                                           "сопоставление может предложить LLM-оценка")
    preview: list[dict[str, str]] = Field(default_factory=list, description="Примеры в формате задачи (вход → ответ)")
    label_values: list[str] = Field(default_factory=list)
    label_overlap: float | None = Field(default=None, description="Доля меток постановки, найденных в датасете")
    label_map: dict[str, str] = Field(default_factory=dict, description="Метка датасета → метка постановки")
    license: str | None = None
    license_ok: bool = True
    issues: list[str] = Field(default_factory=list)


class CriterionScore(_Model):
    score: int = Field(ge=0, le=5)
    reason: str


class StageScores(_Model):
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf: float | None = None
    rerank: float | None = None


class DatasetCandidate(_Model):
    hf_id: str
    source: Literal["hf", "user"] = "hf"
    local_path: str | None = None
    revision: str | None = None
    config_name: str | None = None
    splits: list[str] | None = Field(default=None, description="Какие сплиты брать (None — все)")
    description: str = ""
    license: str | None = None
    split_sizes: dict[str, int] = Field(default_factory=dict)
    columns: list[str] = Field(default_factory=list)
    lang_examples: int | None = None
    stage_scores: StageScores = Field(default_factory=StageScores)
    compat: CompatReport | None = None
    llm_scores: dict[str, CriterionScore] = Field(default_factory=dict)
    score: float = Field(default=0.0, ge=0, le=100)
    why: str = ""


# ---------------------------------------------------------------- данные

class LengthStats(_Model):
    mean: float = 0
    p50: float = 0
    p90: float = 0
    p95: float = 0
    p99: float = 0
    max: float = 0


class DataProfile(_Model):
    n_rows: int
    n_empty: int = 0
    n_exact_dupes: int = 0
    n_synthetic: int = 0
    label_counts: dict[str, int] = Field(default_factory=dict)
    imbalance_ratio: float | None = Field(default=None, description="max/min по числу примеров на метку")
    input_chars: LengthStats = Field(default_factory=LengthStats)
    target_chars: LengthStats = Field(default_factory=LengthStats)
    input_tokens_est: LengthStats = Field(default_factory=LengthStats)
    target_tokens_est: LengthStats = Field(default_factory=LengthStats)
    language_share: dict[str, float] = Field(default_factory=dict)


FilterField = Literal["input_len", "target_len", "language", "label", "is_dupe"]
FilterOperator = Literal["<", "<=", ">", ">=", "==", "!=", "in", "not_in"]


class MapSchemaOp(_Model):
    """Приведение к формату задачи. kind: plain — колонки вход/цель; ner_bio — токены + BIO-теги;
    ner_spans — текст + список сущностей с позициями; qa — контекст + вопрос + ответы;
    chat — диалог (первый обмен репликами: вход — реплика пользователя, ответ — реплика ассистента)."""
    op: Literal["map_schema"] = "map_schema"
    kind: SchemaKind = "plain"
    input_column: str
    target_column: str
    question_column: str | None = None
    extra_input_columns: list[str] = Field(default_factory=list, description="Ещё колонки входа, в порядке подачи модели")
    options_columns: list[str] = Field(default_factory=list, description="Варианты ответа (колонка-список или несколько "
                                                                         "колонок); цель — ключ варианта → его текст")
    language_filter: dict[str, str] = Field(default_factory=dict, description="Колонка языка строки → ISO-код: "
                                                                             "остаются только строки этого языка")
    label_map: dict[str, str] | None = None


class DedupeOp(_Model):
    op: Literal["dedupe"] = "dedupe"
    mode: Literal["exact", "near"] = "exact"
    threshold: float = Field(default=0.9, ge=0.5, le=1.0)


class FilterOp(_Model):
    op: Literal["filter"] = "filter"
    field: FilterField
    operator: FilterOperator
    value: Any


class DropLabelNoiseOp(_Model):
    op: Literal["drop_label_noise"] = "drop_label_noise"
    max_fraction: float = Field(default=0.05, gt=0, le=0.5)


class RebalanceOp(_Model):
    op: Literal["rebalance"] = "rebalance"
    method: Literal["undersample", "oversample"]
    max_ratio: float = Field(default=3.0, ge=1.0, description="Целевое отношение max/min по меткам")


class CleanTextOp(_Model):
    """Нормализация текста входа и цели: HTML-теги и сущности, URL, повторы символов, пробелы."""
    op: Literal["clean_text"] = "clean_text"
    strip_html: bool = True
    remove_urls: bool = False
    max_char_repeat: int | None = Field(default=4, ge=2, le=20, description="«ааааа» → не длиннее N повторов")
    collapse_whitespace: bool = True


class MapLabelsOp(_Model):
    """Переименование/слияние меток (классификация): метка → метка."""
    op: Literal["map_labels"] = "map_labels"
    mapping: dict[str, str]


class AugmentOp(_Model):
    op: Literal["augment"] = "augment"
    method: Literal["llm", "deterministic", "hybrid"]
    path: str = Field(description="Файл с синтетическими примерами (jsonl) относительно папки сессии/запуска")
    sha256: str
    n_rows: int


RecipeOp = Annotated[
    Union[MapSchemaOp, CleanTextOp, MapLabelsOp, DedupeOp, FilterOp, DropLabelNoiseOp, RebalanceOp, AugmentOp],
    Field(discriminator="op"),
]


class SplitProportions(_Model):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    @model_validator(mode="after")
    def _sum_to_one(self):
        if abs(self.train + self.val + self.test - 1.0) > 1e-6:
            raise ValueError("Доли train/val/test должны давать в сумме 1")
        return self


class Recipe(_Model):
    dataset_id: str
    revision: str | None = None
    split_seed: int
    proportions: SplitProportions = Field(default_factory=SplitProportions)
    ops: list[RecipeOp] = Field(default_factory=list)
    profile_before: DataProfile | None = None
    profile_after: DataProfile | None = None
    removed_fraction: float = 0.0
    test_hash: str = ""
    built_by: Literal["rules", "agent"] = "rules"
    rationale: str = ""


# ---------------------------------------------------------------- конфигурация обучения

Backend = Literal["unsloth_lora", "peft_lora", "peft_ptuning"]


class LoraParams(_Model):
    r: int = Field(ge=1, le=256)
    alpha: int = Field(ge=1, le=512)
    dropout: float = Field(default=0.05, ge=0, le=0.5)
    target_modules: list[str] = Field(default_factory=list)
    variant: Literal["lora", "rslora", "dora"] = "lora"


class PtuningParams(_Model):
    num_virtual_tokens: int = Field(ge=1, le=512)
    encoder_hidden_size: int = Field(ge=8, le=4096)
    encoder_reparameterization_type: Literal["MLP", "LSTM"] = "MLP"
    encoder_num_layers: int = Field(default=2, ge=1, le=4)
    encoder_dropout: float = Field(default=0.0, ge=0, le=0.5)


class OptimParams(_Model):
    optimizer: str = "adamw_torch"
    lr: float = Field(gt=0, lt=1)
    scheduler: str = "cosine_with_min_lr"
    warmup: float = Field(default=0.03, ge=0, lt=1, description="Доля шагов прогрева")
    weight_decay: float = Field(default=0.0, ge=0, le=0.5)
    max_grad_norm: float = Field(default=1.0, gt=0)


class ScheduleParams(_Model):
    epochs: int = Field(ge=1, le=50)
    max_steps: int | None = Field(default=None, ge=1, description="Предел шагов первого раунда: бюджет GPU-часов "
                                                                   "меньше одной эпохи; None — по эпохам")
    micro_batch: int = Field(default=4, ge=1)
    grad_accum: int = Field(default=1, ge=1)
    logging_steps: int = Field(default=50, ge=1)
    eval_steps: int = Field(default=400, ge=1)
    patience: int = Field(default=3, ge=1)
    best_metric: str = "eval_loss"
    gradient_checkpointing: bool = False


class Seeds(_Model):
    split_seed: int
    train_seed: int


class Evidence(_Model):
    kind: Literal["run", "doc", "rule", "code"]
    ref: str = Field(description="run_id, doc_chunk_id, имя правила или функции")


class ParamRationale(_Model):
    param: str
    value: Any
    reason: str
    evidence: list[Evidence] = Field(default_factory=list)


class RunConfig(_Model):
    backend: Backend
    base_model_id: str
    quantization: Literal["none", "4bit"] = "none"
    max_seq_len: int = Field(ge=16)
    lora: LoraParams | None = None
    ptuning: PtuningParams | None = None
    optim: OptimParams
    schedule: ScheduleParams
    seeds: Seeds
    strategy: Literal["heuristic", "agent"] = "heuristic"
    rationales: list[ParamRationale] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_adapter(self):
        if (self.lora is None) == (self.ptuning is None):
            raise ValueError("Должно быть задано ровно одно из lora / ptuning")
        if self.backend == "peft_ptuning" and self.ptuning is None:
            raise ValueError("backend peft_ptuning требует ptuning")
        if self.backend != "peft_ptuning" and self.lora is None:
            raise ValueError(f"backend {self.backend} требует lora")
        return self


# ---------------------------------------------------------------- обучение и мониторинг

class MetricPoint(_Model):
    step: int
    epoch: float = 0.0
    train_loss: float | None = None
    val_loss: float | None = None
    val_metrics: dict[str, float] = Field(default_factory=dict)
    lr: float | None = None
    grad_norm: float | None = None
    tokens_per_sec: float | None = None
    gpu_mem_gb: float | None = None
    pred_label_share: dict[str, float] | None = Field(default=None, description="Доли предсказанных меток на val (классификация)")
    max_step: int | None = Field(default=None, description="Последний шаг текущего раунда")
    eta_s: float | None = Field(default=None, description="Осталось до конца раунда по фактическому темпу (с оценками на val)")
    time: datetime = Field(default_factory=utcnow)


AnomalyKind = Literal[
    "nan_loss", "plateau", "overfit", "grad_explosion",
    "throughput_drop", "oom_risk", "prediction_collapse",
]


class Anomaly(_Model):
    kind: AnomalyKind
    severity: Literal["soft", "hard"]
    step: int
    evidence: dict[str, Any] = Field(default_factory=dict)


class LrScalePatch(_Model):
    kind: Literal["lr_scale"] = "lr_scale"
    factor: float = Field(gt=0.05, lt=1.0, description="Только уменьшение lr")


class GradClipPatch(_Model):
    kind: Literal["grad_clip"] = "grad_clip"
    max_grad_norm: float = Field(gt=0.05, le=1.0)


class EvalEveryPatch(_Model):
    """Частота оценки на val (и сохранения чекпоинтов) — для более плотного наблюдения."""
    kind: Literal["eval_every"] = "eval_every"
    steps: int = Field(ge=5, le=100_000)


class LogEveryPatch(_Model):
    kind: Literal["log_every"] = "log_every"
    steps: int = Field(ge=1, le=10_000)


ConfigPatch = Annotated[Union[LrScalePatch, GradClipPatch, EvalEveryPatch, LogEveryPatch], Field(discriminator="kind")]


class MonitorDecision(_Model):
    """Одно решение агента мониторинга (журнал events.jsonl): вердикт, действие, патч, объяснение."""
    verdict: Literal["ok", "suspicious", "broken"]
    action: Literal["continue", "patch", "stop_early", "eval_now", "save_now"]
    patch: ConfigPatch | None = None
    explanation: str = Field(description="Объяснение для пользователя")

    @model_validator(mode="after")
    def _patch_matches_action(self):
        if (self.action == "patch") != (self.patch is not None):
            raise ValueError("patch задаётся тогда и только тогда, когда action == 'patch'")
        return self


RoundChoice = Literal["finish", "continue", "restart"]


class RoundReport(_Model):
    round: int
    epochs_done: float
    val_metrics: dict[str, float] = Field(default_factory=dict)
    baselines_val: dict[str, dict[str, float]] = Field(default_factory=dict)
    checkpoint_path: str | None = None
    adapter_best_path: str | None = None
    sufficient: bool = False
    diagnosis: str = ""
    recommendation: RoundChoice = "finish"
    choice: RoundChoice | None = None
    chosen_by: Literal["user", "system"] | None = None
    extra_epochs: int | None = None


class PredictionExample(_Model):
    input: str
    target: str
    prediction: str
    correct: bool | None = None


class EvalReport(_Model):
    split: Literal["test", "val"] = "test"
    n_examples: int
    metrics: dict[str, float]
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)
    baselines: dict[str, dict[str, float]] = Field(default_factory=dict)
    unparsed_fraction: float = 0.0
    latency_ms: float | None = Field(default=None, description="Средняя задержка одного запроса (батч 1)")
    examples_correct: list[PredictionExample] = Field(default_factory=list)
    examples_wrong: list[PredictionExample] = Field(default_factory=list)
    diagnosis: str = ""


class AutoConfirmation(_Model):
    card: str
    reason: str
    time: datetime = Field(default_factory=utcnow)


class Manifest(_Model):
    run_id: str
    status: Literal["created", "pilot", "running", "round_review", "interrupted", "failed", "finished"] = "created"
    spec: TaskSpec
    dataset: DatasetCandidate
    recipe_path: str = "recipe.json"
    config: RunConfig
    seeds: Seeds
    test_hash: str
    n_train: int = 0
    library_versions: dict[str, str] = Field(default_factory=dict)
    rounds: list[RoundReport] = Field(default_factory=list)
    auto_confirmations: list[AutoConfirmation] = Field(default_factory=list)
    adapter_path: str | None = None
    restart_of_run_id: str | None = None
    session_id: str | None = None
    pid: int | None = None
    resumed_after_interrupt: bool = False
    final_val_metrics: dict[str, float] = Field(default_factory=dict)
    test_metrics: dict[str, float] = Field(default_factory=dict)
    reuse_key: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class ExpertLimits(_Model):
    max_monitor_interventions: int = Field(default=3, ge=0, le=20)
    allowed_monitor_actions: list[Literal["continue", "patch", "stop_early"]] = Field(
        default_factory=lambda: ["continue", "patch", "stop_early"])
    max_rounds: int = Field(default=4, ge=1, le=20)


class SessionSettings(_Model):
    mode: Literal["auto", "expert"] = "auto"
    limits: ExpertLimits = Field(default_factory=ExpertLimits)
