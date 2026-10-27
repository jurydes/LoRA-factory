"""Детекторы аномалий обучения: чистые функции над окном MetricPoint (пороги — config.yaml)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

from factory.core.config import DetectorThresholds
from factory.core.contracts import Anomaly, MetricPoint


@dataclass
class DetectorContext:
    primary_metric: str
    thresholds: DetectorThresholds
    pilot_tokens_per_sec: float | None = None
    vram_total_gb: float | None = None


def _bad(x: float | None) -> bool:
    return x is not None and not math.isfinite(x)


def train_points(points: list[MetricPoint]) -> list[MetricPoint]:
    return [p for p in points if p.train_loss is not None]


def eval_points(points: list[MetricPoint]) -> list[MetricPoint]:
    return [p for p in points if p.val_loss is not None or p.val_metrics]


def nan_loss(points: list[MetricPoint], ctx: DetectorContext) -> Anomaly | None:
    """Жёсткий: loss ушёл в NaN/inf. Мягкий: резкий скачок loss относительно медианы окна."""
    tr = train_points(points)
    if not tr:
        return None
    last = tr[-1]
    if _bad(last.train_loss) or _bad(last.val_loss):
        return Anomaly(kind="nan_loss", severity="hard", step=last.step, evidence={"train_loss": str(last.train_loss)})
    window = [p.train_loss for p in tr[-ctx.thresholds.window - 1:-1] if p.train_loss is not None and math.isfinite(p.train_loss)]
    if len(window) >= 3:
        med = median(window)
        if med > 0 and last.train_loss > ctx.thresholds.loss_spike_factor * med:
            return Anomaly(kind="nan_loss", severity="soft", step=last.step,
                           evidence={"train_loss": last.train_loss, "window_median": med, "note": "скачок loss"})
    return None


def plateau(points: list[MetricPoint], ctx: DetectorContext) -> Anomaly | None:
    """Метрика на val не растёт N оценок подряд, а train loss ещё падает."""
    ev = [p for p in eval_points(points) if ctx.primary_metric in p.val_metrics]
    n = ctx.thresholds.plateau_evals
    if len(ev) < n + 1:
        return None
    best_before = max(p.val_metrics[ctx.primary_metric] for p in ev[:-n])
    recent = [p.val_metrics[ctx.primary_metric] for p in ev[-n:]]
    if max(recent) > best_before + ctx.thresholds.plateau_min_delta:
        return None
    tr = train_points(points)
    span = [p.train_loss for p in tr if p.step >= ev[-n].step and p.train_loss is not None]
    if len(span) >= 2 and span[-1] < span[0]:
        return Anomaly(kind="plateau", severity="soft", step=ev[-1].step,
                       evidence={"best_before": best_before, "recent": recent, "train_loss_from_to": [span[0], span[-1]]})
    return None


def overfit(points: list[MetricPoint], ctx: DetectorContext) -> Anomaly | None:
    """val loss растёт N оценок подряд при падающем train loss."""
    ev = [p for p in eval_points(points) if p.val_loss is not None and math.isfinite(p.val_loss)]
    n = ctx.thresholds.overfit_evals
    if len(ev) < n + 1:
        return None
    tail = [p.val_loss for p in ev[-(n + 1):]]
    if not all(b > a for a, b in zip(tail, tail[1:])):
        return None
    tr = [p.train_loss for p in train_points(points) if p.step >= ev[-(n + 1)].step and p.train_loss is not None]
    if len(tr) >= 2 and tr[-1] < tr[0]:
        return Anomaly(kind="overfit", severity="soft", step=ev[-1].step,
                       evidence={"val_loss": tail, "train_loss_from_to": [tr[0], tr[-1]]})
    return None


def grad_explosion(points: list[MetricPoint], ctx: DetectorContext) -> Anomaly | None:
    g = [p for p in points if p.grad_norm is not None]
    if not g:
        return None
    last = g[-1]
    if _bad(last.grad_norm):
        return Anomaly(kind="grad_explosion", severity="hard", step=last.step, evidence={"grad_norm": str(last.grad_norm)})
    window = [p.grad_norm for p in g[-ctx.thresholds.window - 1:-1] if math.isfinite(p.grad_norm)]
    med = median(window) if len(window) >= 3 else None
    if last.grad_norm > ctx.thresholds.grad_explosion_abs or (med and last.grad_norm > ctx.thresholds.grad_explosion_factor * med):
        return Anomaly(kind="grad_explosion", severity="soft", step=last.step,
                       evidence={"grad_norm": last.grad_norm, "window_median": med})
    return None


def throughput_drop(points: list[MetricPoint], ctx: DetectorContext) -> Anomaly | None:
    """Скорость (токены/с) упала относительно пилота; смотрим медиану последних точек."""
    if not ctx.pilot_tokens_per_sec:
        return None
    tps = [p.tokens_per_sec for p in points if p.tokens_per_sec]
    if len(tps) < 3:
        return None
    current = median(tps[-ctx.thresholds.window:])
    if current < (1 - ctx.thresholds.throughput_drop) * ctx.pilot_tokens_per_sec:
        return Anomaly(kind="throughput_drop", severity="soft", step=points[-1].step,
                       evidence={"tokens_per_sec": current, "pilot": ctx.pilot_tokens_per_sec})
    return None


def oom_risk(points: list[MetricPoint], ctx: DetectorContext) -> Anomaly | None:
    if not ctx.vram_total_gb:
        return None
    mem = [p.gpu_mem_gb for p in points if p.gpu_mem_gb]
    if mem and mem[-1] > ctx.thresholds.oom_risk_share * ctx.vram_total_gb:
        return Anomaly(kind="oom_risk", severity="soft", step=points[-1].step,
                       evidence={"gpu_mem_gb": mem[-1], "vram_total_gb": ctx.vram_total_gb})
    return None


def prediction_collapse(points: list[MetricPoint], ctx: DetectorContext) -> Anomaly | None:
    """Почти все предсказания на val — один класс."""
    ev = [p for p in points if p.pred_label_share]
    if not ev:
        return None
    share = ev[-1].pred_label_share
    label, top = max(share.items(), key=lambda kv: kv[1])
    if top >= ctx.thresholds.collapse_share and len(share) >= 1:
        return Anomaly(kind="prediction_collapse", severity="soft", step=ev[-1].step,
                       evidence={"label": label, "share": top})
    return None


DETECTORS = [nan_loss, plateau, overfit, grad_explosion, throughput_drop, oom_risk, prediction_collapse]


def run_all(points: list[MetricPoint], ctx: DetectorContext) -> list[Anomaly]:
    out = []
    for det in DETECTORS:
        a = det(points, ctx)
        if a is not None:
            out.append(a)
    return out
