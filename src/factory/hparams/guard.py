"""Защитный слой: выход стратегии (эвристика или A3) → проверенный RunConfig.

Стратегия решает: lr, эпохи (верхняя граница), для LoRA — r, alpha, вариант, набор модулей (attention/all_linear);
для P-tuning — число виртуальных токенов, размер и тип энкодера. Всё остальное решает код."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from factory.core.config import AppConfig
from factory.core.contracts import (
    DataProfile, Evidence, LoraParams, OptimParams, ParamRationale, PtuningParams, RunConfig,
    ScheduleParams, Seeds, TaskSpec,
)
from factory.core.registry import ModelInfo, TaskInfo
from factory.hparams import resources
from factory.hparams.retuning import PreviousAttempt


class LoraChoice(BaseModel):
    r: int
    alpha: int
    dropout: float = 0.05
    variant: Literal["lora", "rslora", "dora"] = "lora"
    modules: Literal["attention", "all_linear"] = "attention"


class PtuningChoice(BaseModel):
    num_virtual_tokens: int
    encoder_hidden_size: int
    encoder_reparameterization_type: Literal["MLP", "LSTM"] = "MLP"
    encoder_num_layers: int = 2
    encoder_dropout: float = 0.0


class StrategyChoice(BaseModel):
    """Выход стратегии подбора (вариант A или B)."""
    lr: float = Field(gt=0)
    epochs: int = Field(ge=1)
    lora: LoraChoice | None = None
    ptuning: PtuningChoice | None = None
    rationales: list[ParamRationale] = Field(default_factory=list)


@dataclass
class HParamContext:
    spec: TaskSpec
    task: TaskInfo
    model: ModelInfo
    profile: DataProfile
    n_train: int
    vram_gb: float
    cfg: AppConfig
    seeds: Seeds
    prompt_overhead_tokens: int = 64
    notes: list[str] = field(default_factory=list)
    previous_attempts: list[PreviousAttempt] = field(default_factory=list)

    @property
    def adapter_kind(self) -> Literal["lora", "ptuning"]:
        return "ptuning" if self.spec.constraints.adapter_method == "ptuning" else "lora"

    @property
    def forced_quantization(self) -> str | None:
        return "4bit" if self.spec.constraints.adapter_method == "qlora" else None


def unsloth_available() -> bool:
    return importlib.util.find_spec("unsloth") is not None


def _clamp(value, lo, hi, name: str, notes: list[str]):
    clamped = min(max(value, lo), hi)
    if clamped != value:
        notes.append(f"guard: {name}={value} вне диапазона [{lo}, {hi}] → {clamped}")
    return clamped


def assemble(choice: StrategyChoice, ctx: HParamContext, strategy: Literal["heuristic", "agent"]) -> RunConfig:
    """Проверяет выход стратегии и достраивает конфигурацию кодом."""
    rng = ctx.cfg.hparams.ranges
    notes = ctx.notes
    rationales = list(choice.rationales)
    lr = _clamp(choice.lr, *rng.lr, "lr", notes)
    epochs = int(_clamp(choice.epochs, *rng.epochs, "epochs", notes))

    lora = ptuning = None
    if ctx.adapter_kind == "lora":
        c = choice.lora or LoraChoice(r=8, alpha=16)
        if choice.lora is None:
            notes.append("guard: стратегия не задала LoRA — r=8, alpha=16")
        r = int(_clamp(c.r, *rng.lora_r, "lora.r", notes))
        alpha = int(_clamp(c.alpha, int(r * rng.lora_alpha_ratio[0]) or 1, int(r * rng.lora_alpha_ratio[1]), "lora.alpha", notes))
        lora = LoraParams(r=r, alpha=alpha, dropout=min(max(c.dropout, 0.0), 0.3),
                          target_modules=list(ctx.model.lora_modules[c.modules]), variant=c.variant)
        rationales.append(ParamRationale(param="lora.target_modules", value=lora.target_modules,
                                         reason=f"набор «{c.modules}» из реестра моделей",
                                         evidence=[Evidence(kind="code", ref="registry/models.yaml")]))
    else:
        if not ctx.model.ptuning:
            raise ValueError(f"Модель {ctx.model.id} не поддерживает P-tuning по реестру")
        p = choice.ptuning or PtuningChoice(num_virtual_tokens=20, encoder_hidden_size=128)
        ptuning = PtuningParams(
            num_virtual_tokens=int(_clamp(p.num_virtual_tokens, *rng.num_virtual_tokens, "ptuning.num_virtual_tokens", notes)),
            encoder_hidden_size=int(_clamp(p.encoder_hidden_size, rng.encoder_hidden_size[0],
                                           min(rng.encoder_hidden_size[1], ctx.model.hidden_size), "ptuning.encoder_hidden_size", notes)),
            encoder_reparameterization_type=p.encoder_reparameterization_type,
            encoder_num_layers=min(max(p.encoder_num_layers, 1), 4),
            encoder_dropout=min(max(p.encoder_dropout, 0.0), 0.5),
        )

    prompt_p99 = ctx.profile.input_tokens_est.p99 + ctx.prompt_overhead_tokens
    if ptuning:
        prompt_p99 += ptuning.num_virtual_tokens
    max_seq_len = resources.choose_max_seq_len(prompt_p99, ctx.profile.target_tokens_est.p99, ctx.model,
                                               ctx.cfg.hparams.max_seq_len_cap)
    plan = resources.plan_memory(ctx.model, max_seq_len, ctx.vram_gb, ctx.cfg.hparams.vram_safety, ctx.forced_quantization)
    if not plan["fits"]:
        notes.append(f"guard: по оценке модель не помещается в {ctx.vram_gb:.1f} ГБ даже с 4-bit и checkpointing — проверит пилот")
    logging_steps, eval_steps = resources.logging_and_eval_steps(ctx.n_train, plan["micro_batch"], plan["grad_accum"])

    # Unsloth — для LoRA и QLoRA поддерживаемых архитектур; DoRA и P-tuning — через PEFT
    backend = "peft_ptuning" if ptuning else (
        "unsloth_lora" if ctx.model.unsloth and lora.variant != "dora" and unsloth_available() else "peft_lora")
    rationales += [
        ParamRationale(param="max_seq_len", value=max_seq_len, reason="p99 длины входа+ответа по профилю, ≤ лимита модели",
                       evidence=[Evidence(kind="code", ref="resources.choose_max_seq_len")]),
        ParamRationale(param="quantization/micro_batch/grad_accum", value=[plan["quantization"], plan["micro_batch"], plan["grad_accum"]],
                       reason=f"по VRAM {ctx.vram_gb:.1f} ГБ, оценка пика {plan['estimate']['total_gb']} ГБ",
                       evidence=[Evidence(kind="code", ref="resources.plan_memory")]),
    ]
    return RunConfig(
        backend=backend, base_model_id=ctx.model.id, quantization=plan["quantization"], max_seq_len=max_seq_len,
        lora=lora, ptuning=ptuning,
        optim=OptimParams(lr=lr, warmup=0.03, weight_decay=0.0),
        schedule=ScheduleParams(epochs=epochs, micro_batch=plan["micro_batch"], grad_accum=plan["grad_accum"],
                                logging_steps=logging_steps, eval_steps=eval_steps, patience=3,
                                best_metric=f"eval_{ctx.task.primary_metric}",
                                gradient_checkpointing=plan["gradient_checkpointing"]),
        seeds=ctx.seeds, strategy=strategy, rationales=rationales,
    )


def validate_user_config(config: RunConfig, ctx: HParamContext) -> list[str]:
    """Проверка конфигурации, отредактированной пользователем (режим эксперта). Пустой список — всё в порядке."""
    rng = ctx.cfg.hparams.ranges
    problems = []
    if ctx.model.prequantized and config.quantization != "4bit":
        problems.append("веса модели уже в 4 битах — обучение только QLoRA (quantization 4bit)")
    if not rng.lr[0] <= config.optim.lr <= rng.lr[1]:
        problems.append(f"lr вне диапазона {rng.lr}")
    if not rng.epochs[0] <= config.schedule.epochs <= rng.epochs[1]:
        problems.append(f"epochs вне диапазона {rng.epochs}")
    if config.base_model_id != ctx.model.id:
        problems.append("базовая модель задаётся постановкой, а не конфигурацией")
    if config.lora:
        allowed = set(ctx.model.lora_modules["all_linear"])
        if not set(config.lora.target_modules) <= allowed:
            problems.append(f"target_modules должны быть из реестра: {sorted(allowed)}")
        if not rng.lora_r[0] <= config.lora.r <= rng.lora_r[1]:
            problems.append(f"lora.r вне диапазона {rng.lora_r}")
    if config.max_seq_len > ctx.model.max_context:
        problems.append(f"max_seq_len больше лимита модели {ctx.model.max_context}")
    est = resources.estimate_memory(ctx.model, config.max_seq_len, config.schedule.micro_batch, config.quantization,
                                    config.schedule.gradient_checkpointing)
    if ctx.vram_gb and est["total_gb"] > ctx.vram_gb:
        problems.append(f"оценка памяти {est['total_gb']:.1f} ГБ больше VRAM {ctx.vram_gb:.1f} ГБ")
    return problems
