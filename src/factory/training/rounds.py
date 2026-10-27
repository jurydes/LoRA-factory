"""Раунды обучения: детерминированная проверка «достаточно ли» и плейсхолдеры чисел для текстов диагноза."""
from __future__ import annotations

import re


def best_baseline(baselines: dict[str, dict[str, float]], primary: str) -> tuple[str | None, float | None]:
    scored = [(name, m[primary]) for name, m in baselines.items() if primary in m]
    if not scored:
        return None, None
    return max(scored, key=lambda kv: kv[1])


def is_sufficient(primary: str, val_metrics: dict[str, float], baselines: dict[str, dict[str, float]],
                  target: float | None, prev_value: float | None, min_gain: float) -> tuple[bool, str]:
    """Цель target_primary_metric достигнута; без цели — метрика выше лучшего baseline и последний раунд
    почти не дал прироста (prev_value — значение в конце прошлого раунда или на предыдущей оценке)."""
    value = val_metrics.get(primary)
    if value is None:
        return False, f"нет значения {primary} на val"
    if target is not None:
        if value >= target:
            return True, f"цель достигнута: {primary}={value:.4f} ≥ {target}"
        return False, f"цель не достигнута: {primary}={value:.4f} < {target}"
    name, base = best_baseline(baselines, primary)
    if base is not None and value <= base:
        return False, f"{primary}={value:.4f} не выше лучшего baseline {name}={base:.4f}"
    if prev_value is None:
        return True, f"{primary}={value:.4f} выше лучшего baseline ({name}); прирост оценить не по чему"
    gain = value - prev_value
    if gain < min_gain:
        return True, f"{primary}={value:.4f} выше baseline, прирост за раунд {gain:+.4f} < {min_gain}"
    return False, f"{primary} ещё растёт: прирост за раунд {gain:+.4f}"


PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_.\-]+)\}")
INVENTED_NUMBER = re.compile(r"(?<![\w{.])\d+[.,]\d+|\d+\s?%")


def flatten_values(**groups: dict) -> dict[str, float]:
    """{"val": {"macro_f1": .8}, "baseline": {"majority": {...}}} → {"val.macro_f1": .8, "baseline.majority.macro_f1": ...}"""
    out: dict[str, float] = {}

    def walk(prefix: str, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            out[prefix] = float(obj)

    for name, values in groups.items():
        walk(name, values)
    return out


def check_placeholders(text: str, values: dict[str, float]) -> list[str]:
    problems = []
    unknown = [p for p in PLACEHOLDER.findall(text) if p not in values]
    if unknown:
        problems.append(f"неизвестные плейсхолдеры: {unknown}; доступны: {sorted(values)[:40]}")
    stripped = PLACEHOLDER.sub("", text)
    if INVENTED_NUMBER.search(stripped):
        problems.append("дробные числа и проценты пиши только плейсхолдерами {…}, не цифрами")
    return problems


def render(text: str, values: dict[str, float]) -> str:
    def sub(m):
        v = values.get(m.group(1))
        if v is None:
            return m.group(0)
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
    return PLACEHOLDER.sub(sub, text)
