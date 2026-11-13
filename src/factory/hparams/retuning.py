"""Контекст повторного подбора и защита от повторения уже обученной конфигурации."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from factory.core.contracts import Evidence, ParamRationale, RoundReport, RunConfig
from factory.core.storage import read_model
from factory.runs import RunPaths, load_manifest

if TYPE_CHECKING:
    from factory.hparams.guard import HParamContext


def training_signature(config: RunConfig) -> str:
    """Seed, объяснения и частота логов не делают гиперпараметры новыми."""
    return json.dumps({
        "model": config.base_model_id, "quantization": config.quantization,
        "max_seq_len": config.max_seq_len,
        "optim": config.optim.model_dump(),
        "lora": ({**config.lora.model_dump(), "target_modules": sorted(config.lora.target_modules)}
                 if config.lora else None),
        "ptuning": config.ptuning.model_dump() if config.ptuning else None,
        "epochs": config.schedule.epochs if config.schedule.max_steps is None else None,
        "max_steps": config.schedule.max_steps,
        "effective_batch": config.schedule.micro_batch * config.schedule.grad_accum,
    }, sort_keys=True)


@dataclass
class PreviousAttempt:
    run_id: str
    config: RunConfig
    rounds: list[RoundReport]

    def summary(self) -> dict:
        return {"run_id": self.run_id, "config": json.loads(training_signature(self.config)),
                "rounds": [r.model_dump(include={"round", "epochs_done", "val_metrics", "baselines_val",
                                                "sufficient", "diagnosis", "recommendation"})
                           for r in self.rounds]}


def load_previous_attempts(runs_dir: Path, run_id: str | None) -> list[PreviousAttempt]:
    """Цепочка перезапусков, включая незавершённые запуски; test не используется."""
    attempts, seen = [], set()
    while run_id and run_id not in seen:
        seen.add(run_id)
        paths = RunPaths(runs_dir / run_id)
        manifest = load_manifest(paths)
        configs = [read_model(paths.round_dir(r.round) / "config.json", RunConfig) or manifest.config
                   for r in reversed(manifest.rounds)] + [manifest.config]
        signatures = set()
        for config in configs:
            signature = training_signature(config)
            if signature not in signatures:
                attempts.append(PreviousAttempt(run_id, config, manifest.rounds))
                signatures.add(signature)
        run_id = manifest.restart_of_run_id
    return attempts


def repeats_previous(config: RunConfig, ctx: HParamContext) -> bool:
    return any(training_signature(config) == training_signature(a.config) for a in ctx.previous_attempts)


def ensure_new_config(config: RunConfig, ctx: HParamContext) -> RunConfig:
    """Если стратегия повторилась, пробуем ещё не проверенный lr в разрешённом диапазоне."""
    if not repeats_previous(config, ctx):
        return config
    previous = ctx.previous_attempts[0]
    lo, hi = ctx.cfg.hparams.ranges.lr
    # Сначала меньший шаг: консервативная альтернатива, без обещания роста качества.
    for power in range(1, len(ctx.previous_attempts) + 3):
        for factor in (2.0 ** -power, 2.0 ** power):
            lr = min(hi, max(lo, previous.config.optim.lr * factor))
            candidate = config.model_copy(update={"optim": config.optim.model_copy(update={"lr": lr})})
            if repeats_previous(candidate, ctx):
                continue
            reason = (f"Повторный подбор после {previous.run_id}: стратегия повторила уже проверенную конфигурацию; "
                      f"выбран ещё не проверенный lr {config.optim.lr:g} → {lr:g}. "
                      "Это новая попытка, улучшение качества требует проверки на val.")
            ctx.notes.append(reason)
            candidate.rationales = [r for r in candidate.rationales if r.param != "lr"] + [
                ParamRationale(param="lr", value=lr, reason=reason,
                               evidence=[Evidence(kind="run", ref=previous.run_id)])]
            return candidate
    raise ValueError("Повторный подбор не нашёл новых параметров в допустимом диапазоне lr. "
                     "Измените конфигурацию вручную или пересмотрите данные и модель.")
