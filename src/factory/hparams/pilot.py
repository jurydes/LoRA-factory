"""Пилот: короткий прогон ловит NaN/OOM, меряет скорость и память → оценка времени (верхняя граница).
Авто-исправления, не более двух попыток: OOM → меньше micro-batch и больше накопление (эффективный батч тот же)
→ gradient checkpointing → QLoRA; NaN → lr/2 и больше warmup."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from factory.core.config import PilotConfig
from factory.core.contracts import RunConfig
from factory.core.storage import write_json
from factory.hparams.resources import steps_per_epoch
from factory.runs import RunPaths
from factory.training import launcher

logger = logging.getLogger(__name__)


@dataclass
class PilotReport:
    status: str
    config: RunConfig
    attempts: list[dict] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    result: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def pilot_steps(n_train: int, cfg: RunConfig, pc: PilotConfig) -> int:
    spe = steps_per_epoch(n_train, cfg.schedule.micro_batch, cfg.schedule.grad_accum)
    total = spe * cfg.schedule.epochs
    return int(min(pc.max_steps, max(pc.min_steps, math.ceil(total * pc.fraction))))


def fix_oom(cfg: RunConfig) -> tuple[RunConfig, str] | None:
    s = cfg.schedule
    if s.micro_batch > 1:
        mb = s.micro_batch // 2
        return cfg.model_copy(update={"schedule": s.model_copy(update={"micro_batch": mb, "grad_accum": s.grad_accum * 2})}), \
            f"OOM → micro_batch {s.micro_batch}→{mb}, grad_accum {s.grad_accum}→{s.grad_accum * 2}"
    if not s.gradient_checkpointing:
        return cfg.model_copy(update={"schedule": s.model_copy(update={"gradient_checkpointing": True})}), \
            "OOM → gradient checkpointing"
    if cfg.quantization != "4bit" and cfg.lora is not None:
        backend = cfg.backend if cfg.backend == "unsloth_lora" else "peft_lora"
        return cfg.model_copy(update={"quantization": "4bit", "backend": backend}), "OOM → QLoRA (4-bit)"
    return None


def fix_nan(cfg: RunConfig) -> tuple[RunConfig, str]:
    o = cfg.optim
    warmup = min(0.2, max(o.warmup * 2, 0.05))
    return cfg.model_copy(update={"optim": o.model_copy(update={"lr": o.lr / 2, "warmup": warmup})}), \
        f"NaN → lr {o.lr:g}→{o.lr / 2:g}, warmup {o.warmup:g}→{warmup:g}"


async def run_pilot(paths: RunPaths, cfg: RunConfig, n_train: int, pc: PilotConfig) -> PilotReport:
    report = PilotReport(status="error", config=cfg)
    # статусы прошлого пилота (пилот A3, «Подобрать заново») — иначе последним по имени окажется чужой статус,
    # и обучение возьмёт из него скорость и оценку времени другой конфигурации
    for old in paths.pilot.glob("status_*.json"):
        old.unlink()
    for attempt in range(pc.max_fix_attempts + 1):
        write_json(paths.pilot / "config.json", cfg)
        steps = pilot_steps(n_train, cfg, pc)
        result = await launcher.run(paths, "pilot", paths.pilot / f"status_{attempt}.json", "--steps", str(steps))
        report.attempts.append(result)
        status = result.get("status")
        if status == "ok":
            report.status, report.config, report.result = "ok", cfg, result
            return report
        if attempt == pc.max_fix_attempts:
            break
        if status == "oom":
            fixed = fix_oom(cfg)
            if fixed is None:
                break
        elif status == "nan":
            fixed = fix_nan(cfg)
        else:
            break  # прочие ошибки авто-исправлением не лечатся
        cfg, note = fixed
        report.fixes.append(note)
        logger.info("Пилот: %s", note)
    report.status = report.attempts[-1].get("status", "error") if report.attempts else "error"
    report.config = cfg
    report.result = report.attempts[-1] if report.attempts else {}
    return report
