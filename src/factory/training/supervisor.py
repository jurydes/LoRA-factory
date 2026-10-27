"""Надзор за раундом обучения (в процессе сервиса): метрики → детекторы → A4 для мягких отклонений;
жёсткие сбои — правилами: NaN → resume с lr/2 (один раз); OOM → лестница пилота; зависание → уведомление."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from factory.core import activity
from factory.core.config import AppConfig
from factory.core.contracts import (
    Anomaly,
    Manifest,
    MetricPoint,
    RunConfig,
    SessionSettings,
)
from factory.core.registry import Registry
from factory.core.storage import (
    append_jsonl,
    read_json,
    read_jsonl,
    read_model,
    write_json,
)
from factory.hparams.pilot import fix_oom
from factory.runs import RunPaths
from factory.training import detectors, launcher

logger = logging.getLogger(__name__)

Notify = Callable[[str], Awaitable[None] | None]


def send_command(paths: RunPaths, cmd: str, patch: dict | None = None) -> str:
    """Команда процессу обучения через control.json (stop_and_save, abort, patch)."""
    data = read_json(paths.control, {"commands": []}) or {"commands": []}
    cid = uuid.uuid4().hex[:8]
    data["commands"].append({"id": cid, "cmd": cmd, "patch": patch, "time": time.time()})
    write_json(paths.control, data)
    return cid


def checkpoint_state(ckpt: Path) -> tuple[int, float, float | None]:
    """(global_step, epoch, последний lr) из trainer_state.json чекпоинта."""
    state = json.loads((ckpt / "trainer_state.json").read_text(encoding="utf-8"))
    lr = next((h["learning_rate"] for h in reversed(state.get("log_history", [])) if "learning_rate" in h), None)
    return int(state.get("global_step", 0)), float(state.get("epoch", 0.0)), lr


def checkpoint_offsets(ckpt: Path, events: list[dict], base_step: int, base_epoch: float) -> tuple[int, float]:
    """Смещение (шаг, эпоха) процесса, создавшего чекпоинт раунда: номер чекпоинта и его trainer_state считают шаги
    от старта процесса, а раунд мог продолжаться не раз (OOM, NaN, перезапуск сервиса). Берётся старт процесса —
    последнее событие round_started до сохранения чекпоинта; в старых событиях нет step_offset — он выводится из
    эпохи и шагов на эпоху. Нет событий — начало раунда (base_*)."""
    saved = ckpt.stat().st_mtime
    started = [e for e in events if e.get("type") == "round_started" and e.get("time", 0) <= saved]
    if not started:
        return base_step, base_epoch
    e = started[-1]
    epoch_offset = float(e.get("epoch_offset") or base_epoch)
    if e.get("step_offset") is not None:
        return int(e["step_offset"]), epoch_offset
    step, epoch, _ = checkpoint_state(ckpt)
    per_epoch = step / epoch if epoch else 0
    return base_step + round((epoch_offset - base_epoch) * per_epoch), epoch_offset


def is_intervention(decision: dict) -> bool:
    """Вмешательство в оптимизацию (идёт в лимит): поправка lr/градиента; частота оценки и остановка — нет
    (остановка завершает раунд с лучшим адаптером и не должна упираться в лимит поправок)."""
    patch = decision.get("patch") or {}
    return decision.get("action") == "patch" and patch.get("kind") in ("lr_scale", "grad_clip")


@dataclass
class RoundOutcome:
    status: str  # ok | aborted | failed
    result: dict = field(default_factory=dict)
    config: RunConfig | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class RoundPlan:
    round_k: int
    epochs: float
    resume_from: Path | None = None
    lr: float | None = None
    step_offset: int = 0
    epoch_offset: float = 0.0


class Supervisor:
    def __init__(self, paths: RunPaths, cfg: AppConfig, registry: Registry, settings: SessionSettings,
                 notify: Notify, pilot_tps: float | None, vram_gb: float | None, epoch_s: float | None = None):
        self.paths = paths
        self.epoch_s = epoch_s  # секунд на эпоху по пилоту: оценка времени, пока нет точек раунда
        self.cfg = cfg
        self.registry = registry
        self.settings = settings
        self.notify = notify
        manifest = read_model(paths.manifest, Manifest)
        self.manifest = manifest
        self.task = registry.task(manifest.spec.task_type)
        self.det_ctx = detectors.DetectorContext(primary_metric=self.task.primary_metric,
                                                 thresholds=cfg.training.detectors,
                                                 pilot_tokens_per_sec=pilot_tps, vram_total_gb=vram_gb)
        self.interventions = sum(1 for e in read_jsonl(paths.events)[0]
                                 if e.get("type") == "monitor_decision" and is_intervention(e))
        self.nan_retry_used = False
        self.loop_ended = False  # цикл обучения раунда закончился (идёт итоговая оценка) — команды не применятся
        self.pending: set[str] = set()  # отправленные, но ещё не подтверждённые процессом команды

    async def _say(self, text: str):
        r = self.notify(text)
        if asyncio.iscoroutine(r):
            await r

    def _event(self, **e):
        append_jsonl(self.paths.events, {"time": time.time(), **e})

    async def run_round(self, plan: RoundPlan, config: RunConfig, attach_pid: int | None = None) -> RoundOutcome:
        """Раунд обучения. attach_pid — процесс раунда уже идёт (сервис перезапускался): только наблюдаем."""
        notes: list[str] = []
        oom_fixes = 0
        while True:
            rdir = self.paths.round_dir(plan.round_k)
            rdir.mkdir(parents=True, exist_ok=True)
            write_json(rdir / "config.json", config)
            extra = ["--round", str(plan.round_k), "--epochs", str(plan.epochs),
                     "--step-offset", str(plan.step_offset), "--epoch-offset", str(plan.epoch_offset)]
            if plan.resume_from:
                extra += ["--resume-from", str(plan.resume_from)]
            if plan.lr is not None:
                extra += ["--lr", str(plan.lr)]
            status_file = rdir / "status.json"
            if attach_pid is not None:
                pid, attach_pid = attach_pid, None
                alive = lambda pid=pid: launcher.pid_alive(pid)  # noqa: E731
                self._event(type="round_reattached", round=plan.round_k, pid=pid)
            else:
                # команды и heartbeat прошлого раунда не должны действовать на новый процесс
                self.paths.control.unlink(missing_ok=True)
                self.paths.heartbeat.unlink(missing_ok=True)
                proc = launcher.spawn(self.paths, "train", status_file, *extra)
                await activity.emit(f"обучение, раунд {plan.round_k}: {plan.epochs:g} эп.", "рабочий процесс")
                pid, alive = proc.pid, (lambda proc=proc: proc.poll() is None)
                self.manifest.pid = pid
                self.manifest.status = "running"
                write_json(self.paths.manifest, self.manifest)
                self._event(type="round_started", round=plan.round_k, pid=pid, epochs=plan.epochs,
                            step_offset=plan.step_offset, epoch_offset=plan.epoch_offset,
                            target_epoch=plan.epoch_offset + plan.epochs,
                            resume_from=str(plan.resume_from) if plan.resume_from else None,
                            estimated_s=round(self.epoch_s * plan.epochs) if self.epoch_s else None)
            result = await self._watch(alive, status_file, plan, config)
            status = result.get("status")
            self._event(type="round_process_exit", round=plan.round_k, status=status, message=result.get("message"))

            if status == "ok":
                return RoundOutcome("ok", result, config, notes)
            if status == "aborted":
                return RoundOutcome("aborted", result, config, notes)
            ckpt = Path(result["checkpoint"]) if result.get("checkpoint") else None
            if status == "nan" and not self.nan_retry_used:
                self.nan_retry_used = True
                plan, config = self._replan(plan, config, ckpt, lr_factor=0.5)
                lr = plan.lr if plan.lr is not None else config.optim.lr
                note = (f"NaN в loss → продолжение с последнего чекпоинта с lr/2 = {lr:g}" if plan.resume_from
                        else f"NaN в loss → перезапуск раунда с lr/2 = {lr:g}")
                notes.append(note)
                await self._say(note)
                self._event(type="hard_rule", rule="nan_resume", lr=lr)
                continue
            if status == "oom" and oom_fixes < self.cfg.hparams.pilot.max_fix_attempts:
                fixed = fix_oom(config)
                if fixed:
                    oom_fixes += 1
                    config, note = fixed
                    notes.append(note)
                    await self._say(f"Нехватка памяти: {note}")
                    self._event(type="hard_rule", rule="oom_ladder", note=note)
                    plan, config = self._replan(plan, config, ckpt, lr_factor=1.0)
                    continue
            return RoundOutcome("failed", result, config, notes)

    def _replan(self, plan: RoundPlan, config: RunConfig, ckpt: Path | None,
                lr_factor: float) -> tuple[RoundPlan, RunConfig]:
        """Новый план раунда после сбоя: с чекпоинта (lr — из его trainer_state × lr_factor) или раунд заново."""
        if ckpt is not None and ckpt.exists():
            step, epoch, lr_ckpt = checkpoint_state(ckpt)
            lr = (lr_ckpt if lr_ckpt is not None else (plan.lr or config.optim.lr)) * lr_factor
            return RoundPlan(plan.round_k, max(plan.epochs - epoch, 0.5), resume_from=ckpt, lr=lr,
                             step_offset=plan.step_offset + step, epoch_offset=plan.epoch_offset + epoch), config
        if plan.resume_from is not None:  # раунд-продолжение без нового чекпоинта: тот же старт
            lr = (plan.lr or config.optim.lr) * lr_factor
            return RoundPlan(plan.round_k, plan.epochs, plan.resume_from, lr, plan.step_offset, plan.epoch_offset), config
        if lr_factor != 1.0:
            config = config.model_copy(update={"optim": config.optim.model_copy(update={"lr": config.optim.lr * lr_factor})})
        return RoundPlan(plan.round_k, plan.epochs, None, None, plan.step_offset, plan.epoch_offset), config

    async def _watch(self, alive, status_file: Path, plan: RoundPlan, config: RunConfig) -> dict:
        offset = self.paths.metrics.stat().st_size if self.paths.metrics.exists() else 0
        ev_offset = self.paths.events.stat().st_size if self.paths.events.exists() else 0
        self.loop_ended, self.pending = False, set()
        window: list[MetricPoint] = []
        last_decision_step = -10 ** 9
        hung_reported = slow_reported = False
        handled: set[tuple[str, int]] = set()
        next_check_at = self.cfg.training.monitor_check_every_evals
        evals = 0  # оценок на val за раунд — нарастающим итогом, а не по окну (окно в 200 точек их теряет)
        while alive():
            await asyncio.sleep(self.cfg.training.poll_interval_s)
            events, ev_offset = read_jsonl(self.paths.events, ev_offset)
            self.loop_ended |= any(e.get("type") == "train_loop_end" for e in events)
            records, offset = read_jsonl(self.paths.metrics, offset)
            if records:
                points = [MetricPoint.model_validate(r) for r in records]
                evals += sum(1 for p in points if p.val_metrics or p.val_loss is not None)
                window = (window + points)[-200:]
                anomalies = [a for a in detectors.run_all(window, self.det_ctx) if (a.kind, a.step) not in handled]
                for a in anomalies:
                    handled.add((a.kind, a.step))
                    self._event(type="anomaly", **a.model_dump(mode="json"))
                slow = next((a for a in anomalies if a.kind == "throughput_drop"), None)
                if slow and not slow_reported:  # дело железа, а не обучения: агенту нечего менять — сказать пользователю
                    slow_reported = True
                    await self._say(
                        f"Обучение замедлилось: {slow.evidence.get('tokens_per_sec')} токенов/с против "
                        f"{slow.evidence.get('pilot')} на пробном прогоне. Обычно видеопамять заняли другие программы "
                        "(редакторы, браузеры, игры): когда её не хватает, Windows без ошибки выносит часть данных "
                        "в оперативную память и всё замедляется в разы. Закройте их — скорость вернётся.")
                soft = [a for a in anomalies if a.severity == "soft" and a.kind != "throughput_drop"]
                step = window[-1].step
                due = (self.cfg.training.monitor_check_every_evals and evals >= next_check_at)
                if (soft or due) and not self.loop_ended and step - last_decision_step >= self.cfg.training.monitor_cooldown_steps:
                    last_decision_step = step
                    if due:
                        next_check_at = evals + self.cfg.training.monitor_check_every_evals
                    trigger = "аномалии детекторов" if soft else f"плановая проверка после {evals}-й оценки на val"
                    await self._consult_monitor(soft, window, config, trigger)
            hb = self.paths.heartbeat
            status = read_json(status_file, {}) or {}
            if status.get("status") == "running" and hb.exists():
                stale = time.time() - hb.stat().st_mtime
                if stale > self.cfg.training.heartbeat_timeout_s and not hung_reported:
                    hung_reported = True
                    self._event(type="hung", stale_s=stale)
                    await self._say(f"Процесс обучения не делает шагов уже {stale / 60:.0f} мин. "
                                    "Можно подождать или остановить обучение с сохранением.")
        status = read_json(status_file) or {}
        if status.get("status") in (None, "queued", "running"):  # процесс умер, не дописав итог
            return {"status": "error", "message": "процесс обучения завершился без итогового статуса"}
        return status

    async def _consult_monitor(self, soft: list[Anomaly], window: list[MetricPoint], config: RunConfig, trigger: str):
        """A4 смотрит на живой процесс через инструменты и при необходимости сразу управляет им."""
        from factory.agents import monitor
        limits = self.settings.limits
        deps = monitor.MonitorDeps(
            task_type=self.task.key, primary_metric=self.task.primary_metric,
            allowed_actions=list(limits.allowed_monitor_actions),
            interventions_left=limits.max_monitor_interventions - self.interventions,
            live=LiveView(self, window, config), eval_steps=config.schedule.eval_steps)
        await activity.emit(f"проверяет обучение: {trigger}", activity.ROLES["monitor"])
        try:
            verdict = await monitor.review(deps, trigger, [a.model_dump(mode="json") for a in soft])
        except Exception as e:
            self._event(type="monitor_unavailable", error=str(e))
            return
        self.interventions += sum(1 for d in deps.actions if is_intervention(d.model_dump(mode="json")))
        self._event(type="monitor_review", trigger=trigger, verdict=verdict.verdict, explanation=verdict.explanation,
                    actions=[d.model_dump(mode="json") for d in deps.actions])
        if deps.actions or verdict.verdict != "ok":
            await self._say(f"Агент мониторинга ({trigger}): {verdict.explanation}")

    def _acks(self) -> dict[str, dict]:
        return {e.get("id"): e for e in read_jsonl(self.paths.events)[0] if e.get("type") == "control_applied"}

    async def send_and_confirm(self, cmd: str, patch: dict | None, decision) -> dict:
        """Команда процессу обучения и ожидание подтверждения (control_applied в events.jsonl).

        Пока прошлая команда не подтверждена, новая не отправляется: неподтверждённая лежит в очереди и применится,
        и повтор той же поправки сложился бы с ней (lr × 0.5 × 0.5…). sent=False — не отправлено и не вмешательство."""
        self.pending -= set(self._acks())
        if self.loop_ended:
            return {"applied": False, "sent": False, "command": cmd,
                    "note": "цикл обучения раунда уже закончился (идёт итоговая оценка) — команды не применяются"}
        if self.pending:
            return {"applied": False, "sent": False, "command": cmd,
                    "note": "прошлая команда ещё не подтверждена процессом и применится сама — новая не отправлена"}
        cid = send_command(self.paths, cmd, patch)
        self.pending.add(cid)
        self._event(type="monitor_decision", command_id=cid, **decision.model_dump(mode="json"))
        deadline = time.monotonic() + self.cfg.training.monitor_ack_timeout_s
        while time.monotonic() < deadline:
            ack = self._acks().get(cid)
            if ack:
                self.pending.discard(cid)
                return {"applied": True, "at_step": ack.get("step"), "command": cmd}
            await asyncio.sleep(1.0)
        return {"applied": False, "sent": True, "command": cmd,
                "note": "процесс ещё не подтвердил (идёт оценка или долгий шаг) — команда в очереди и применится; не повторяй её"}


class LiveView:
    """Окно в живой процесс для A4: метрики, конфигурация, история, команды."""

    def __init__(self, sup: Supervisor, window: list[MetricPoint], config: RunConfig):
        self.sup, self.config = sup, config
        self.start_step = window[0].step if window else 0  # только точки текущего раунда
        self.last_seen = window[-1].step if window else 0

    def _points(self) -> list[MetricPoint]:
        """Свежие точки прямо из metrics.jsonl живого процесса (не снимок на момент вызова агента)."""
        return [MetricPoint.model_validate(r) for r in read_jsonl(self.sup.paths.metrics)[0] if r["step"] >= self.start_step]

    def recent_metrics(self, n: int) -> list[dict]:
        points = self._points()
        if points:
            self.last_seen = max(self.last_seen, points[-1].step)
        return [p.model_dump(mode="json", exclude={"time"}, exclude_none=True) for p in points[-n:]]

    def val_curve(self) -> list[dict]:
        return [{"step": p.step, "val_loss": p.val_loss, **p.val_metrics}
                for p in self._points() if p.val_loss is not None or p.val_metrics]

    async def wait_new(self, seconds: float) -> list[dict]:
        await asyncio.sleep(seconds)
        new = [p for p in self._points() if p.step > self.last_seen]
        if new:
            self.last_seen = new[-1].step
        return [p.model_dump(mode="json", exclude={"time"}, exclude_none=True) for p in new[-30:]]

    def config_summary(self) -> dict:
        c = self.config
        return {"lr": c.optim.lr, "scheduler": c.optim.scheduler, "warmup": c.optim.warmup,
                "max_grad_norm": c.optim.max_grad_norm, "epochs": c.schedule.epochs,
                "micro_batch": c.schedule.micro_batch, "grad_accum": c.schedule.grad_accum,
                "eval_steps": c.schedule.eval_steps, "logging_steps": c.schedule.logging_steps,
                "adapter": c.lora.model_dump(include={"r", "alpha", "variant"}) if c.lora else "ptuning",
                "quantization": c.quantization, "backend": c.backend}

    def event_history(self, n: int) -> list[dict]:
        keep = {"anomaly", "monitor_decision", "monitor_review", "control_applied", "hard_rule"}
        return [e for e in read_jsonl(self.sup.paths.events)[0] if e.get("type") in keep][-n:]

    async def send(self, cmd: str, patch: dict | None, decision) -> dict:
        return await self.sup.send_and_confirm(cmd, patch, decision)
