"""A2 DataPrep: ReAct-агент подготовки данных. Работает только с выборкой train через закрытый набор
инструментов; итог — воспроизводимый рецепт, который код потом применяет ко всему train.

Цикл: check_quality → исследование (find_examples, length_outliers, near_duplicates, language_report,
label_issues, sample) → операция (сначала apply=false, затем apply=true; undo — откат) → check_quality …
Рецепт фиксируется, только когда все цели качества выполнены или на каждую невыполненную дан обоснованный отказ."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits

from factory.core import llm
from factory.core.config import DataPrepConfig, PrepRules
from factory.core.contracts import (
    AugmentOp, CleanTextOp, DataProfile, DedupeOp, DropLabelNoiseOp, FilterOp, MapLabelsOp, RebalanceOp, RecipeOp,
    TaskFamily,
)
from factory.prep import lang
from factory.prep import ops as ops_mod
from factory.prep.augment import synthesize
from factory.prep.profile import profile_frame, profile_summary
from factory.prep.quality import failing, junk_mask, length_outlier_mask, quality_report

logger = logging.getLogger(__name__)
MAX_CHARS = 240


@dataclass
class PrepWorkspace:
    sample: pd.DataFrame
    family: TaskFamily
    language: str
    labels: list[str] | None
    task_description: str
    cfg: DataPrepConfig
    mode: Literal["auto", "expert"]
    synth_dir: Path
    seed: int
    findings: list[str]
    ops: list[RecipeOp] = field(default_factory=list)
    current: pd.DataFrame | None = None
    profile_before: DataProfile | None = None
    augment_proposal: dict | None = None
    committed: bool = False
    rationale: str = ""
    waivers: dict[str, str] = field(default_factory=dict)
    full_train_rows: int = 0
    history: list[pd.DataFrame] = field(default_factory=list)
    quality_checks: int = 0
    version: int = 0  # растёт при каждом изменении выборки (операция или undo)
    seen: dict[str, int] = field(default_factory=dict)  # инструмент наблюдения → версия при последнем вызове

    def repeat_note(self, tool: str) -> dict:
        """Повтор наблюдения без изменений в данных ничего нового не даст — агент ходил по кругу
        «посмотреть мусор → проверить цели» десятки запросов."""
        stale = self.seen.get(tool) == self.version
        self.seen[tool] = self.version
        if not stale:
            return {}
        return {"note": "данные не менялись с прошлого вызова — результат тот же. Не повторяй: исправь операцией "
                        "или, если цель для этой задачи не нужна или неисправима, обоснуй это в waivers и вызови commit_recipe"}

    def __post_init__(self):
        if self.current is None:
            self.current = self.sample.copy()
        self.profile_before = profile_frame(self.sample, self.family, self.seed)

    @property
    def rules(self) -> PrepRules:
        return self.cfg.rules

    def record(self, op: RecipeOp, new_df: pd.DataFrame) -> dict:
        before = len(self.current)
        self.history.append(self.current)
        self.ops.append(op)
        self.current = new_df
        self.version += 1
        return {"applied": op.op, "rows_before": before, "rows_after": len(new_df)}

    def report(self) -> dict[str, dict]:
        return quality_report(self.current, self.family, self.rules, self.language, self.labels, self.seed)


def _rows(df: pd.DataFrame, n: int) -> list[dict]:
    return df.head(max(1, min(n, 10)))[["input", "target"]].astype(str).map(lambda s: s[:MAX_CHARS]).to_dict(orient="records")


class PrepOutcome(BaseModel):
    rationale: str = Field(description="Что сделано и почему — для пользователя, 2–5 предложений")


INSTRUCTIONS = """Ты — агент подготовки данных для дообучения LLM. Ты видишь только выборку train через инструменты,
которые возвращают сводки. Цель — воспроизводимый рецепт чистки, после которого выполнены ЦЕЛИ КАЧЕСТВА
(check_quality): нет пустых строк и точных дублей, мало почти-дублей, мусора (HTML, URL, строки без букв) и выбросов
по длине, нужный язык преобладает, классы не слишком несбалансированы, метки из постановки, мало шума меток.

Работай циклом, пока цели не выполнены:
1) check_quality — что не так;
2) исследуй причину: find_examples (поиск по шаблону), length_outliers, near_duplicates, language_report,
   label_issues, inspect_label, sample — смотри на реальные примеры, прежде чем удалять;
3) исправь операцией: сначала apply=false (оценить эффект), потом apply=true; ошибся — undo;
   clean_text чинит мусор, не удаляя строки; filter удаляет; map_labels переименовывает/сливает метки;
   dedupe, drop_label_noise, rebalance; augment — синтез, если данных мало или редкие классы почти пусты;
4) снова check_quality.
Не удаляй больше, чем нужно: лучше почистить, чем выбросить. Не проверяй «фактическую верность» текстов.
Если цель не нужна или вредна для этой задачи (например, дисбаланс отражает реальность, а балансировка уничтожит
данные) — оставь её невыполненной и дай обоснование в waivers при commit_recipe. В конце — commit_recipe."""

# повторы инструментов: commit_recipe отклоняется, пока цели не выполнены, — агент должен успеть исправить
agent = Agent(deps_type=PrepWorkspace, output_type=PrepOutcome, instructions=INSTRUCTIONS, retries={"output": 3, "tools": 8})


@agent.instructions
def _context(ctx: RunContext[PrepWorkspace]) -> str:
    w = ctx.deps
    return (f"Задача: {w.task_description}. Тип: {w.family}; язык: {w.language}; метки: {w.labels or '—'}.\n"
            f"Выборка train: {len(w.sample)} строк из {w.full_train_rows}; синтез (augment) — не больше "
            f"{w.cfg.augment_max_rows} строк на весь train. Находки правил: {'; '.join(w.findings) or 'нет'}.\n"
            f"Уже в рецепте (из правил): {[o.op for o in w.ops]}.")


# ---------------------------------------------------------------- наблюдение

@agent.tool
def check_quality(ctx: RunContext[PrepWorkspace]) -> dict:
    """Цели качества на текущей выборке: значение, порог, выполнена ли. Вызывай до и после исправлений."""
    w = ctx.deps
    w.quality_checks += 1
    goals = w.report()
    return {"goals": goals, "failing": failing(goals), "rows": len(w.current), "ops_so_far": [o.op for o in w.ops]} \
        | w.repeat_note("check_quality")


@agent.tool
def profile(ctx: RunContext[PrepWorkspace]) -> dict:
    """Профиль текущей выборки: размеры, баланс, длины, дубли, язык."""
    return profile_summary(profile_frame(ctx.deps.current, ctx.deps.family, ctx.deps.seed)) | ctx.deps.repeat_note("profile")


@agent.tool
def sample(ctx: RunContext[PrepWorkspace], n: int = 5, label: str | None = None) -> list[dict]:
    """Случайные строки текущей выборки (тексты обрезаны). label — только строки с этой меткой."""
    df = ctx.deps.current
    if label is not None:
        df = df[df["target"] == label]
    return _rows(df.sample(n=min(max(n, 1), 10, len(df)), random_state=ctx.deps.seed) if len(df) else df, n)


@agent.tool
def inspect_label(ctx: RunContext[PrepWorkspace], label: str, n: int = 6) -> dict:
    """Метка: сколько примеров, типичные длины и примеры — чтобы понять, не мусорная ли она и не дубль ли другой."""
    df = ctx.deps.current[ctx.deps.current["target"] == label]
    return {"count": len(df), "input_chars_p50": float(df["input"].str.len().median()) if len(df) else 0,
            "examples": _rows(df.sample(n=min(n, len(df)), random_state=ctx.deps.seed) if len(df) else df, n)}


@agent.tool
def find_examples(ctx: RunContext[PrepWorkspace], pattern: str, field: Literal["input", "target"] = "input", n: int = 6) -> dict:
    """Поиск по регулярному выражению (без учёта регистра): сколько строк совпало и примеры."""
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise ModelRetry(f"Некорректное регулярное выражение: {e}")
    df = ctx.deps.current
    hits = df[df[field].map(lambda t: bool(rx.search(str(t))))]
    return {"matches": len(hits), "share": round(len(hits) / max(len(df), 1), 4), "examples": _rows(hits, n)}


@agent.tool
def junk_examples(ctx: RunContext[PrepWorkspace], n: int = 6) -> dict:
    """Строки с мусором: HTML-разметка, URL, почти нет букв."""
    df = ctx.deps.current
    mask = junk_mask(df["input"]) | (junk_mask(df["target"], answers=True) if ctx.deps.family != "classification" else False)
    return {"count": int(mask.sum()), "examples": _rows(df[mask], n)} | ctx.deps.repeat_note("junk_examples")


@agent.tool
def length_outliers(ctx: RunContext[PrepWorkspace], n: int = 5) -> dict:
    """Выбросы по длине и квантили длин (символы) входа и ответа."""
    df = ctx.deps.current
    mask = length_outlier_mask(df, ctx.deps.rules)
    q = lambda s: {k: float(v) for k, v in s.str.len().quantile([0.5, 0.9, 0.99, 1.0]).items()}  # noqa: E731
    return {"count": int(mask.sum()), "input_quantiles": q(df["input"]), "target_quantiles": q(df["target"]),
            "examples": _rows(df[mask], n)} | ctx.deps.repeat_note("length_outliers")


@agent.tool
def near_duplicates(ctx: RunContext[PrepWorkspace], threshold: float = 0.9, n: int = 5) -> dict:
    """Почти-дубли входа (MinHash, threshold 0.5–1): сколько и примеры пар «оригинал — дубль»."""
    df = ctx.deps.current.reset_index(drop=True)
    mask = ops_mod.near_duplicate_mask(df, threshold)
    norm = df["input"].map(ops_mod.normalize_text)
    pairs = []
    for i in df.index[mask][:n]:
        words = set(norm[i].split())
        j = max((k for k in range(i) if not mask[k]), key=lambda k: len(words & set(norm[k].split())), default=None)
        if j is not None:
            pairs.append({"original": df.at[j, "input"][:MAX_CHARS], "duplicate": df.at[i, "input"][:MAX_CHARS]})
    return {"count": int(mask.sum()), "share": round(mask.mean(), 4), "pairs": pairs}


@agent.tool
def language_report(ctx: RunContext[PrepWorkspace], n: int = 5) -> dict:
    """Распределение языков (по выборке до 400 строк) и примеры строк не на языке задачи."""
    w = ctx.deps
    df = w.current.sample(n=min(400, len(w.current)), random_state=w.seed) if len(w.current) else w.current
    langs = df["input"].map(lang.detect)
    other = df[langs != w.language]
    return {"share": lang.language_share(df["input"].tolist()), "examples_other": _rows(other, n)}


@agent.tool
def label_issues(ctx: RunContext[PrepWorkspace], top: int = 8) -> dict:
    """Подозрительные метки (cleanlab по out-of-sample вероятностям TF-IDF+LogReg). Только классификация."""
    w = ctx.deps
    if w.family != "classification":
        return {"error": "только для классификации"}
    cur = w.current.reset_index(drop=True)
    idx = ops_mod.label_issue_indices(cur, w.seed)
    return {"n_suspicious": len(idx), "share": round(len(idx) / max(len(cur), 1), 4),
            "top_examples": _rows(cur.loc[idx[:min(top, 10)]], top)}


@agent.tool
def diff(ctx: RunContext[PrepWorkspace]) -> dict:
    """Профиль до и после применённых операций."""
    w = ctx.deps
    return {"before": profile_summary(w.profile_before),
            "after": profile_summary(profile_frame(w.current, w.family, w.seed)),
            "ops": [o.op for o in w.ops]}


# ---------------------------------------------------------------- операции рецепта

def _try(w: PrepWorkspace, op: RecipeOp, new: pd.DataFrame, apply: bool, extra: dict | None = None) -> dict:
    if apply:
        unchanged = len(new) == len(w.current) and new[["input", "target"]].reset_index(drop=True).equals(
            w.current[["input", "target"]].reset_index(drop=True))
        if unchanged:  # пустая операция не попадает в рецепт
            return {"applied": False, "note": "операция ничего не меняет — в рецепт не добавлена, не повторяй её"}
        return w.record(op, new) | (extra or {})
    return {"would_remove": len(w.current) - len(new), "rows_after": len(new)} | (extra or {})


@agent.tool
def clean_text(ctx: RunContext[PrepWorkspace], strip_html: bool = True, remove_urls: bool = False,
               max_char_repeat: int | None = 4, collapse_whitespace: bool = True, apply: bool = False) -> dict:
    """Почистить тексты (вход и ответ) без удаления строк: HTML-теги и сущности, URL, «ааааа», лишние пробелы."""
    w = ctx.deps
    op = CleanTextOp(strip_html=strip_html, remove_urls=remove_urls, max_char_repeat=max_char_repeat,
                     collapse_whitespace=collapse_whitespace)
    new = ops_mod.apply_clean_text(w.current, op)
    changed = int((new["input"] != w.current["input"]).sum() + (new["target"] != w.current["target"]).sum())
    return _try(w, op, new, apply, {"changed_texts": changed, "example_after": _rows(new[new["input"] != w.current["input"]], 2)})


@agent.tool
def map_labels(ctx: RunContext[PrepWorkspace], mapping: dict[str, str], apply: bool = False) -> dict:
    """Переименовать/слить метки (только классификация), например {"positive": "позитив", "pos": "позитив"}."""
    w = ctx.deps
    if w.family != "classification":
        return {"error": "только для классификации"}
    unknown = [k for k in mapping if k not in set(w.current["target"])]
    if unknown:
        raise ModelRetry(f"Таких меток нет в данных: {unknown}")
    op = MapLabelsOp(mapping=mapping)
    new = ops_mod.apply_map_labels(w.current, op)
    return _try(w, op, new, apply, {"labels_after": new["target"].value_counts().to_dict()})


@agent.tool
def dedupe(ctx: RunContext[PrepWorkspace], mode: Literal["exact", "near"] = "exact", threshold: float = 0.9, apply: bool = False) -> dict:
    """Удаление дублей: exact (нормализованный текст) или near (MinHash по 3-граммам слов, threshold 0.5–1)."""
    op = DedupeOp(mode=mode, threshold=threshold)
    return _try(ctx.deps, op, ops_mod.apply_dedupe(ctx.deps.current, op), apply)


@agent.tool
def filter(ctx: RunContext[PrepWorkspace], field: Literal["input_len", "target_len", "language", "label", "is_dupe"],
           operator: Literal["<", "<=", ">", ">=", "==", "!=", "in", "not_in"], value: Any, apply: bool = False) -> dict:
    """Фильтр: строка ОСТАЁТСЯ, если условие истинно. Поля: длины в символах, язык (ISO), метка, признак дубля."""
    op = FilterOp(field=field, operator=operator, value=value)
    return _try(ctx.deps, op, ops_mod.apply_filter(ctx.deps.current, op), apply)


@agent.tool
def drop_label_noise(ctx: RunContext[PrepWorkspace], max_fraction: float = 0.03, apply: bool = False) -> dict:
    """Удаление самых подозрительных меток, не более max_fraction train (ограничено конфигом)."""
    w = ctx.deps
    if w.family != "classification":
        return {"error": "только для классификации"}
    cap = w.cfg.max_label_noise_fraction
    if max_fraction > cap:
        raise ModelRetry(f"max_fraction не может превышать {cap}")
    op = DropLabelNoiseOp(max_fraction=max_fraction)
    return _try(w, op, ops_mod.apply_drop_label_noise(w.current, op, w.seed), apply)


@agent.tool
def rebalance(ctx: RunContext[PrepWorkspace], method: Literal["undersample", "oversample"], max_ratio: float = 3.0, apply: bool = False) -> dict:
    """Балансировка классов до отношения max/min ≤ max_ratio: undersample крупных или повтор мелких."""
    w = ctx.deps
    if w.family != "classification":
        return {"error": "только для классификации"}
    op = RebalanceOp(method=method, max_ratio=max_ratio)
    new = ops_mod.apply_rebalance(w.current, op, w.seed)
    counts = new["target"].value_counts()
    return _try(w, op, new, apply, {"imbalance_after": round(float(counts.max() / max(counts.min(), 1)), 2)})


@agent.tool
async def augment(ctx: RunContext[PrepWorkspace], method: Literal["llm", "deterministic", "hybrid"], n_rows: int,
                  confidence: float, reason: str) -> dict:
    """Синтез данных в стиле датасета (помечаются synthetic, только train). confidence 0–1 — уверенность, что это нужно.
    n_rows — строк, добавляемых ко ВСЕМУ train (не к выборке); в выборке эффект виден пропорционально её доле в train.
    llm — генерация по образцам параллельными батчами; deterministic — синонимы и перестановки; hybrid — оба.
    В режиме АВТО при высокой уверенности выполняется сразу, иначе уходит пользователю на одобрение."""
    w = ctx.deps
    n_rows = max(1, min(n_rows, w.cfg.augment_max_rows))
    if w.mode == "auto" and confidence >= w.cfg.augment_confidence:
        op: AugmentOp = await synthesize(method, w.current, w.family, n_rows, w.task_description, w.language,
                                         w.synth_dir, w.cfg.augment_workers, w.seed)
        if op.n_rows == 0:
            return {"generated": 0, "note": "все сгенерированные строки отсеяны проверками (копии, язык, длина)"}
        # синтетика добавится ко всему train, а агент смотрит на выборку: в выборку — та же доля строк, иначе
        # 1500 строк на выборке из 3000 «исправили» бы дисбаланс, которого на train из 200 тыс. почти не коснутся
        synth = ops_mod.load_synthetic(w.synth_dir / op.path)
        share = len(w.sample) / max(w.full_train_rows, len(w.sample))
        shown = synth.sample(n=round(len(synth) * share), random_state=w.seed)
        view = pd.concat([w.current, shown], ignore_index=True)
        return w.record(op, view) | {"generated": op.n_rows, "added_to_sample_view": len(shown),
                                     "note": f"выборка — {share:.1%} train, поэтому в ней видно {len(shown)} из {op.n_rows} строк"}
    w.augment_proposal = {"method": method, "n_rows": n_rows, "confidence": confidence, "reason": reason}
    return {"status": "предложение сохранено и будет показано пользователю на одобрение"}


@agent.tool
def undo(ctx: RunContext[PrepWorkspace]) -> dict:
    """Отменить последнюю применённую операцию."""
    w = ctx.deps
    if not w.history:
        return {"error": "нечего отменять"}
    w.current = w.history.pop()
    w.version += 1
    return {"undone": w.ops.pop().op, "rows": len(w.current)}


@agent.tool
def commit_recipe(ctx: RunContext[PrepWorkspace], rationale: str, waivers: dict[str, str] | None = None) -> dict:
    """Зафиксировать рецепт. Каждая невыполненная цель check_quality должна быть в waivers с обоснованием
    (почему её не нужно или нельзя выполнять для этой задачи); иначе — продолжай исправлять."""
    w = ctx.deps
    waivers = {k: v for k, v in (waivers or {}).items() if len(str(v).strip()) >= 15}
    open_goals = [g for g in failing(w.report()) if g not in waivers]
    if open_goals:
        raise ModelRetry(f"Цели не выполнены: {open_goals}. Исправь их операциями или обоснуй отказ в waivers "
                         "(по 1–2 предложения на цель).")
    w.committed, w.rationale, w.waivers = True, rationale, waivers
    return {"committed": [o.op for o in w.ops], "waived": list(waivers)}


@agent.output_validator
def _require_commit(ctx: RunContext[PrepWorkspace], out: PrepOutcome) -> PrepOutcome:
    if not ctx.deps.committed:
        raise ModelRetry("Сначала доведи цели качества и вызови commit_recipe")
    return out


async def run(workspace: PrepWorkspace) -> PrepOutcome:
    limits = UsageLimits(request_limit=workspace.cfg.request_limit, tool_calls_limit=workspace.cfg.tool_calls_limit)
    result = await llm.run_agent(agent, "dataprep", "Подготовь рецепт чистки данных: доведи цели качества.",
                                 deps=workspace, usage_limits=limits)
    out = result.output
    if workspace.waivers:
        out.rationale += " Сознательно не выполнено: " + "; ".join(f"{k} — {v}" for k, v in workspace.waivers.items())
    return out
