"""Блок 3: стратегия A (эвристика) или B (агент A3) → guard → пилот. Флаг HPARAM_STRATEGY / hparams.strategy."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from factory.core.contracts import RunConfig
from factory.hparams import heuristic, resources
from factory.hparams.guard import HParamContext, assemble
from factory.hparams.pilot import PilotReport, run_pilot
from factory.runs import RunPaths

logger = logging.getLogger(__name__)


@dataclass
class Proposal:
    config: RunConfig
    strategy: Literal["heuristic", "agent"]
    pilot: PilotReport
    notes: list[str] = field(default_factory=list)
    fallback_reason: str | None = None


def baseline_primary(baselines: dict, primary: str) -> float | None:
    """Метрика исходной модели с промптом (0–1) — вход эвристики курсовой вместо BERTScore до обучения."""
    value = (baselines.get("base_prompt") or {}).get(primary)
    return None if value is None else max(0.0, min(1.0, float(value)))


async def propose(ctx: HParamContext, paths: RunPaths, baselines_val: dict,
                  strategy: Literal["heuristic", "agent"]) -> Proposal:
    used, fallback = strategy, None
    choice = None
    if strategy == "agent":
        from factory.agents import hparam
        try:
            choice = await hparam.run(hparam.HParamDeps(ctx=ctx, paths=paths, baselines_val=baselines_val))
        except Exception as e:
            fallback = f"{type(e).__name__}: {e}"
            logger.warning("A3 недоступен (%s) — возврат на эвристику", fallback)
            used = "heuristic"
    if choice is None:
        choice = heuristic.suggest(ctx, baseline_primary(baselines_val, ctx.task.primary_metric))
    config = assemble(choice, ctx, used)
    report = await run_pilot(paths, config, ctx.n_train, ctx.cfg.hparams.pilot)
    config = fit_budget(report, ctx)
    return Proposal(config=config, strategy=used, pilot=report, notes=list(ctx.notes), fallback_reason=fallback)


def fit_budget(report: PilotReport, ctx: HParamContext) -> RunConfig:
    """Бюджет GPU-часов постановки: если оценка пилота (верхняя граница) больше — код урезает число эпох
    пропорционально; если не влезает и одна эпоха — первый раунд ограничивается числом шагов в пределах бюджета."""
    config, budget = report.config, ctx.spec.constraints.gpu_hours
    est = report.result.get("estimated_time_s") if report.ok else None
    if not budget or not est or est <= budget * 3600:
        return config
    epochs = config.schedule.epochs
    per_epoch = est / epochs
    fitted = int(budget * 3600 / per_epoch)
    if fitted >= 1:
        config = config.model_copy(update={"schedule": config.schedule.model_copy(update={"epochs": fitted})})
        report.result["estimated_time_s"] = round(per_epoch * fitted, 1)
        ctx.notes.append(f"бюджет {budget} GPU-ч: эпох {epochs} → {fitted} (оценка пилота {est / 3600:.1f} ч)")
        return config
    spe = report.result.get("steps_per_epoch") or resources.steps_per_epoch(
        ctx.n_train, config.schedule.micro_batch, config.schedule.grad_accum)
    steps = max(1, int(spe * budget * 3600 / per_epoch))
    config = config.model_copy(update={"schedule": config.schedule.model_copy(update={"epochs": 1, "max_steps": steps})})
    report.result["estimated_time_s"] = round(budget * 3600, 1)
    ctx.notes.append(f"бюджет {budget} GPU-ч: даже одна эпоха (~{per_epoch / 3600:.1f} ч) не укладывается — "
                     f"первый раунд ограничен {steps} шагами (~{steps / spe:.0%} эпохи)")
    return config
