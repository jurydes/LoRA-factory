"""Блок 2: обязательная часть без LLM (разбиение, формат, профиль) → правила → A2 при необходимости →
рецепт применяется ко всему train обычным кодом."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

from factory.core.config import AppConfig
from factory.core.contracts import (
    CleanTextOp,
    DatasetCandidate,
    MapSchemaOp,
    Recipe,
    SplitProportions,
    TaskSpec,
)
from factory.core.registry import TaskInfo
from factory.core.storage import read_model, write_json
from factory.prep import ops as ops_mod
from factory.prep import rules as rules_mod
from factory.prep.loading import load_candidate_frame
from factory.prep.profile import profile_frame
from factory.prep.quality import failing, quality_report
from factory.prep.schema import drop_unusable, to_task_frame
from factory.prep.split import save_splits, split_frame

logger = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass
class PrepResult:
    recipe: Recipe
    mass_removal: bool
    augment_proposal: dict | None
    agent_error: str | None = None
    labels: list[str] | None = None  # фактический набор меток train (классификация)


def map_schema_op(candidate: DatasetCandidate, spec: TaskSpec) -> MapSchemaOp:
    compat = candidate.compat
    if not compat or not compat.input_column or not compat.target_column:
        raise ValueError(f"Для датасета {candidate.hf_id} не определены колонки входа и цели")
    from factory.search.compat import schema_op
    return schema_op(compat, {k: v for k, v in compat.label_map.items() if k != v})


def mandatory_part(candidate: DatasetCandidate, spec: TaskSpec, task: TaskInfo, data_dir: Path,
                   proportions: SplitProportions, split_seed: int) -> tuple[pd.DataFrame, MapSchemaOp, str, int]:
    """Загрузка → формат задачи → строки без входа/ответа убраны → разбиение → test закрыт.
    Возвращает (train, map_schema, test_hash, убрано строк без входа или ответа)."""
    raw = load_candidate_frame(candidate)
    schema = map_schema_op(candidate, spec)
    df, dropped = drop_unusable(to_task_frame(raw, schema, task.format))
    if dropped:
        logger.info("Убрано %d строк без входа или ответа из %d", dropped, len(df) + dropped)
    if df.empty:
        raise ValueError(f"В датасете {candidate.hf_id} не осталось примеров со входом и ответом")
    train, val, test = split_frame(df, proportions, split_seed, stratify=task.family == "classification")
    test_hash = save_splits(data_dir, train, val, test)
    (data_dir / "train.raw.parquet").write_bytes((data_dir / "train.parquet").read_bytes())
    logger.info("Разбиение: train=%d val=%d test=%d", len(train), len(val), len(test))
    return train, schema, test_hash, dropped


async def prepare(candidate: DatasetCandidate, spec: TaskSpec, task: TaskInfo, work_dir: Path, cfg: AppConfig,
                  mode: Literal["auto", "expert"], split_seed: int, progress: Progress | None = None,
                  use_agent: bool = True) -> PrepResult:
    say = progress or (lambda _: None)
    data_dir = work_dir / "data"
    p = cfg.dataprep.proportions
    proportions = SplitProportions(train=p[0], val=p[1], test=p[2])

    say("Загружаю датасет, привожу к формату задачи и разбиваю на train/val/test (test сразу закрыт)…")
    train, schema, test_hash, dropped = await asyncio.to_thread(
        mandatory_part, candidate, spec, task, data_dir, proportions, split_seed)
    profile_before = await asyncio.to_thread(profile_frame, train, task.family, split_seed)
    verdict = rules_mod.assess(profile_before, task.family, cfg.dataprep.rules, spec.language)
    if dropped:
        verdict.findings.insert(0, f"до разбиения убрано {dropped} строк без входа или ответа (нет метки, "
                                   "нет ответа, несколько меток)")
    sample = train.sample(n=min(cfg.dataprep.sample_size, len(train)), random_state=split_seed).reset_index(drop=True)
    for op in verdict.ops:  # операции правил — на выборке, чтобы оценить, что ещё осталось
        sample = ops_mod.apply_op(sample, op, split_seed, work_dir)
    labels_for_check = spec.labels if task.format == "label" else None
    goals = await asyncio.to_thread(quality_report, sample, task.family, cfg.dataprep.rules, spec.language,
                                    labels_for_check, split_seed)
    open_goals = failing(goals)
    if open_goals:
        verdict.findings.append(f"после правил не выполнены цели качества: {', '.join(open_goals)}")
    ops = [schema] + verdict.ops
    built_by, rationale, proposal, agent_error = "rules", "; ".join(verdict.findings) or "проблем не найдено", None, None

    if (verdict.call_agent or open_goals) and use_agent:
        say("Данным нужна чистка — запускаю агента подготовки данных на выборке train "
            f"({', '.join(open_goals) or 'неоднозначная статистика'})…")
        from factory.agents import dataprep

        raw_sample = train.sample(n=min(cfg.dataprep.sample_size, len(train)), random_state=split_seed).reset_index(drop=True)
        ws = dataprep.PrepWorkspace(
            sample=raw_sample, family=task.family, language=spec.language, labels=labels_for_check,
            task_description=spec.description, cfg=cfg.dataprep, mode=mode, synth_dir=work_dir,
            seed=split_seed, findings=verdict.findings, full_train_rows=len(train))
        # операции правил применяются к выборке до агента, чтобы он видел их эффект
        for op in verdict.ops:
            ws.record(op, ops_mod.apply_op(ws.current, op, split_seed, work_dir))
        try:
            outcome = await dataprep.run(ws)
            ops, built_by, rationale, proposal = [schema] + ws.ops, "agent", outcome.rationale, ws.augment_proposal
        except Exception as e:
            agent_error = f"{type(e).__name__}: {e}"
            logger.warning("Агент подготовки не справился (%s) — рецепт только из правил", agent_error)
            rationale += f". Агент недоступен ({type(e).__name__}) — рецепт только из правил"
            if "junk_text" in open_goals:  # безопасная детерминированная чистка: строки не удаляются
                ops.append(CleanTextOp(strip_html=True, remove_urls=True))
                rationale += "; текст почищен от HTML и URL"

    recipe = Recipe(dataset_id=candidate.hf_id, revision=candidate.revision, split_seed=split_seed,
                    proportions=proportions, ops=ops, profile_before=profile_before, test_hash=test_hash,
                    built_by=built_by, rationale=rationale)
    recipe = await asyncio.to_thread(apply_and_save, recipe, data_dir, work_dir, task)
    mass = recipe.removed_fraction > cfg.dataprep.confirm_removed_fraction
    labels = None
    if task.format == "label" and recipe.profile_after:
        labels = sorted(recipe.profile_after.label_counts)
    elif task.format == "entities":  # типы сущностей — из разметки train
        from factory.training.metrics import parse_entities
        train = pd.read_parquet(data_dir / "train.parquet")
        labels = sorted({t for v in train["target"] for t, _ in parse_entities(v)[0]})
    return PrepResult(recipe=recipe, mass_removal=mass, augment_proposal=proposal, agent_error=agent_error, labels=labels)


def apply_and_save(recipe: Recipe, data_dir: Path, work_dir: Path, task: TaskInfo) -> Recipe:
    """Применяет рецепт к сохранённому исходному train и пишет train.parquet + recipe.json."""
    raw_train = pd.read_parquet(data_dir / "train.raw.parquet")
    train = ops_mod.apply_recipe(raw_train, recipe, work_dir)
    real_after = int((~train["synthetic"]).sum())
    removed = 1 - len(set(train.loc[~train["synthetic"], "_id"])) / max(len(set(raw_train["_id"])), 1)
    train.to_parquet(data_dir / "train.parquet", index=False)
    recipe = recipe.model_copy(update={
        "profile_after": profile_frame(train, task.family, recipe.split_seed),
        "removed_fraction": round(max(removed, 0.0), 4),
    })
    write_json(work_dir / "recipe.json", recipe)
    logger.info("Рецепт применён: %d → %d реальных строк train (+%d синтетических)",
                len(raw_train), real_after, len(train) - real_after)
    return recipe


def load_recipe(work_dir: Path) -> Recipe | None:
    return read_model(work_dir / "recipe.json", Recipe)
