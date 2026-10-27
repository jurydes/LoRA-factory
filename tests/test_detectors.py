"""По тесту на каждый детектор: срабатывает на типичной картине и молчит на нормальной."""
import math

from factory.core.config import DetectorThresholds
from factory.core.contracts import MetricPoint
from factory.training import detectors as d

CTX = d.DetectorContext(primary_metric="macro_f1", thresholds=DetectorThresholds(), pilot_tokens_per_sec=1000.0,
                        vram_total_gb=12.0)


def tr(step, loss, **kw):
    return MetricPoint(step=step, train_loss=loss, **kw)


def ev(step, val_loss=None, f1=None, share=None):
    return MetricPoint(step=step, val_loss=val_loss, val_metrics={"macro_f1": f1} if f1 is not None else {},
                       pred_label_share=share)


def test_nan_loss_hard():
    a = d.nan_loss([tr(1, 2.0), tr(2, float("nan"))], CTX)
    assert a and a.kind == "nan_loss" and a.severity == "hard"


def test_nan_loss_spike_soft():
    pts = [tr(i, 1.0) for i in range(1, 7)] + [tr(7, 5.0)]
    a = d.nan_loss(pts, CTX)
    assert a and a.severity == "soft"


def test_nan_loss_quiet_on_normal():
    assert d.nan_loss([tr(i, 2.0 - i * 0.1) for i in range(1, 8)], CTX) is None


def test_plateau():
    pts = []
    for i, f1 in enumerate([0.5, 0.6, 0.6, 0.6, 0.6]):
        pts += [tr(i * 10 + 1, 1.0 - i * 0.1), ev(i * 10 + 5, val_loss=0.9, f1=f1)]
    a = d.plateau(pts, CTX)
    assert a and a.kind == "plateau"


def test_plateau_quiet_when_improving():
    pts = []
    for i, f1 in enumerate([0.5, 0.55, 0.6, 0.65, 0.7]):
        pts += [tr(i * 10 + 1, 1.0 - i * 0.1), ev(i * 10 + 5, val_loss=0.9, f1=f1)]
    assert d.plateau(pts, CTX) is None


def test_overfit():
    pts = []
    for i, vl in enumerate([0.8, 0.85, 0.9, 1.0]):
        pts += [tr(i * 10 + 1, 1.0 - i * 0.2), ev(i * 10 + 5, val_loss=vl, f1=0.5)]
    a = d.overfit(pts, CTX)
    assert a and a.kind == "overfit"


def test_overfit_quiet_when_val_falls():
    pts = []
    for i, vl in enumerate([1.0, 0.9, 0.85, 0.8]):
        pts += [tr(i * 10 + 1, 1.0 - i * 0.2), ev(i * 10 + 5, val_loss=vl, f1=0.5)]
    assert d.overfit(pts, CTX) is None


def test_grad_explosion():
    pts = [tr(i, 1.0, grad_norm=1.0) for i in range(1, 7)] + [tr(7, 1.0, grad_norm=50.0)]
    a = d.grad_explosion(pts, CTX)
    assert a and a.severity == "soft"
    a = d.grad_explosion([tr(1, 1.0, grad_norm=math.inf)], CTX)
    assert a and a.severity == "hard"
    assert d.grad_explosion([tr(i, 1.0, grad_norm=1.0) for i in range(1, 8)], CTX) is None


def test_throughput_drop():
    pts = [tr(i, 1.0, tokens_per_sec=300.0) for i in range(1, 6)]
    assert d.throughput_drop(pts, CTX).kind == "throughput_drop"
    assert d.throughput_drop([tr(i, 1.0, tokens_per_sec=950.0) for i in range(1, 6)], CTX) is None


def test_oom_risk():
    assert d.oom_risk([tr(1, 1.0, gpu_mem_gb=11.8)], CTX).kind == "oom_risk"
    assert d.oom_risk([tr(1, 1.0, gpu_mem_gb=6.0)], CTX) is None


def test_prediction_collapse():
    assert d.prediction_collapse([ev(5, share={"позитив": 0.97, "негатив": 0.03})], CTX).kind == "prediction_collapse"
    assert d.prediction_collapse([ev(5, share={"позитив": 0.5, "негатив": 0.5})], CTX) is None


def test_run_all_collects():
    pts = [tr(1, 1.0, gpu_mem_gb=11.9), tr(2, float("nan"))]
    kinds = {a.kind for a in d.run_all(pts, CTX)}
    assert {"nan_loss", "oom_risk"} <= kinds
