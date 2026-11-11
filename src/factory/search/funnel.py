"""Воронка поиска датасета: «дешёвое на многом, дорогое на малом» (§3.1)."""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

from factory.core import llm
from factory.core.config import AppConfig, get_config
from factory.core.contracts import DatasetCandidate, StageScores, TaskSpec
from factory.core.registry import Registry
from factory.prep.loading import hf_sample, read_user_file
from factory.search import compat as compat_mod
from factory.search.catalog import load_catalog
from factory.search.llm_eval import assess
from factory.search.retrieval import hybrid_search

logger = logging.getLogger(__name__)

Progress = Callable[[str], None]

FALLBACK_EXPAND_PROMPT = "Переформулируй задачу пользователя как запрос для поиска датасетов (3 предложения): {query}"


async def expand_query(spec: TaskSpec) -> str:
    """Расширение запроса промптом из курсовой (dataset_work.EXPAND_QUERY_PROMPT) через общий клиент."""
    try:
        from dataset_work import EXPAND_QUERY_PROMPT
    except Exception:
        EXPAND_QUERY_PROMPT = FALLBACK_EXPAND_PROMPT
    labels = f" Метки: {', '.join(spec.labels)}." if spec.labels else ""
    query = f"{spec.description} (тип задачи: {spec.task_type}, язык: {spec.language}).{labels}"
    try:
        return await llm.complete("query_expansion", EXPAND_QUERY_PROMPT.format(query=query))
    except Exception as e:
        logger.warning("Расширение запроса не удалось (%s), ищу по исходной формулировке", e)
        return query


def _check_one(record: dict, stages: dict, spec: TaskSpec, registry: Registry, cfg: AppConfig):
    task = registry.task(spec.task_type)
    try:
        meta = compat_mod.hf_metadata(record["id"])
        if meta["has_loading_script"] or meta["gated"]:
            return None, None
        sample, extra = hf_sample(record["id"], max(50, cfg.search.llm_examples * 10), meta["revision"],
                                  language=spec.language, split_sizes=meta["split_sizes"])
        meta.update(extra)
    except Exception as e:
        logger.info("Датасет %s пропущен: %s", record["id"], e)
        return None, None
    cand = compat_mod.candidate_from_sample(
        record["id"], spec, task, sample, meta, description=record.get("description", ""),
        min_lang_examples=cfg.search.min_lang_examples)
    cand.stage_scores = StageScores(**stages)
    return cand, sample


async def search(spec: TaskSpec, registry: Registry, cfg: AppConfig | None = None,
                 progress: Progress | None = None, exclude: set[str] | None = None) -> list[DatasetCandidate]:
    """Полная воронка: расширение запроса → BM25+dense → RRF → cross-encoder → совместимость → LLM-оценка."""
    cfg = cfg or get_config()
    say = progress or (lambda _: None)
    if cfg.offline:
        say("Закрытый контур: загрузите свой датасет.")
        return []
    exclude = exclude or set()

    query = await expand_query(spec)
    say(f"Ищу по запросу: {query}")
    catalog = await asyncio.to_thread(load_catalog, spec.language, cfg)
    ranked = await asyncio.to_thread(hybrid_search, catalog, query, cfg.search)
    ranked = [(i, s) for i, s in ranked if catalog.records[i]["id"] not in exclude]
    if not ranked:
        say("Подходящих датасетов не найдено.")
        return []
    say(f"Каталог: {len(catalog)} датасетов; после гибридного поиска и реранкинга — {len(ranked)}. Проверяю совместимость…")

    finalists: list[tuple[DatasetCandidate, object]] = []
    unmapped: list[tuple[DatasetCandidate, object]] = []  # колонки не распознаны — сопоставление предложит LLM
    pool_size = min(len(ranked), cfg.search.top_llm * 2)
    loop = asyncio.get_running_loop()
    # Проверка — в отдельных процессах: выборка датасета, лежащего одним большим JSON, скачивается и разбирается
    # целиком, и разбор держит GIL — в потоках сервиса на это время замирал весь сервер (чат, SSE, API).
    # Общий лимит времени: тяжёлые кандидаты отбрасываются, а не задерживают поиск.
    # Linux fork inherits the server's threads and large ML objects. Spawn a few clean workers instead;
    # eight parallel HF loaders exhausted Docker Desktop's memory in a real search.
    pool = ProcessPoolExecutor(max_workers=min(3, os.cpu_count() or 1),
                               mp_context=multiprocessing.get_context("spawn"))
    futures: list[asyncio.Future] = []
    checked: dict[int, tuple[DatasetCandidate | None, object | None]] = {}
    completed_count = 0
    try:
        futures = [loop.run_in_executor(pool, _check_one, catalog.records[i], s, spec, registry, cfg)
                   for i, s in ranked[:pool_size]]
        indexes = {future: index for index, future in enumerate(futures)}
        pending = set(futures)
        deadline = loop.time() + cfg.search.compat_timeout_s
        while pending and loop.time() < deadline:
            done, pending = await asyncio.wait(pending, timeout=min(10, deadline - loop.time()),
                                               return_when=asyncio.FIRST_COMPLETED)
            completed_count += len(done)
            for future in done:
                if future.cancelled():
                    continue
                if error := future.exception():
                    logger.warning("Проверка совместимости упала: %r", error)
                    continue
                checked[indexes[future]] = future.result()
            compatible = sum(bool(c and c.compat and c.compat.ok) for c, _ in checked.values())
            say(f"Проверено {completed_count}/{pool_size} датасетов; совместимых: {compatible}…")
            if compatible >= cfg.search.top_llm:
                break
        if pending:
            logger.info("Проверка совместимости: %d датасетов не уложились в %s с — пропущены",
                        len(pending), cfg.search.compat_timeout_s)
        for index in sorted(checked):  # результаты — в порядке ранжирования
            cand, sample = checked[index]
            if cand is not None and cand.compat and cand.compat.ok and len(finalists) < cfg.search.top_llm:
                finalists.append((cand, sample))
            elif cand is not None and cand.compat and cand.compat.needs_mapping:
                unmapped.append((cand, sample))
    finally:
        for future in futures:
            if not future.done():
                future.cancel()
        processes = list((pool._processes or {}).values())
        for p in processes:  # сначала останавливаем зависшие загрузчики, потом освобождаем пул
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.5)
                if p.is_alive():
                    p.kill()
        try:
            await asyncio.wait_for(asyncio.to_thread(pool.shutdown, wait=False, cancel_futures=True), timeout=5)
        except TimeoutError:
            logger.warning("Освобождение пула проверки датасетов заняло более 5 с")
    unmapped = unmapped[:max(0, cfg.search.top_llm - len(finalists))]
    if not finalists and not unmapped:
        say("Совместимых датасетов не найдено. Можно загрузить свой файл или уточнить запрос.")
        return []
    say(f"Совместимых финалистов: {len(finalists)}"
        + (f" (+{len(unmapped)} с нераспознанными колонками — их сопоставит LLM)" if unmapped else "")
        + ". Оцениваю их по критериям…")
    task = registry.task(spec.task_type)

    async def _assess(cand, sample):
        try:
            return await assess(spec, task, cand, sample, cfg.search.criteria_weights, cfg.search.llm_examples)
        except Exception as e:
            logger.warning("LLM-оценка %s не удалась: %s", cand.hf_id, e)
            return cand.model_copy(update={"why": "LLM-оценка недоступна; кандидат прошёл проверку совместимости"})

    evaluation = finalists + unmapped
    jobs = [asyncio.create_task(_assess(c, s)) for c, s in evaluation]
    assessed = []
    if jobs:
        done, pending = await asyncio.wait(jobs, timeout=cfg.search.assess_timeout_s)
        for job, (cand, _) in zip(jobs, evaluation):
            if job in done and not job.cancelled() and job.exception() is None:
                assessed.append(job.result())
            else:
                if job in pending:
                    job.cancel()
                assessed.append(cand.model_copy(update={"why": "LLM-оценка не уложилась в лимит времени"}))
        if pending:
            say(f"LLM-оценка {len(pending)} датасетов не уложилась в {cfg.search.assess_timeout_s:g} с.")
    return sorted((c for c in assessed if c.compat and c.compat.ok), key=lambda c: -c.score)


async def evaluate_user_dataset(path: Path, spec: TaskSpec, registry: Registry, cfg: AppConfig | None = None) -> DatasetCandidate:
    """Датасет пользователя (файл pandas-совместимого формата) проходит те же проверки и LLM-оценку."""
    cfg = cfg or get_config()
    task = registry.task(spec.task_type)
    df = await asyncio.to_thread(read_user_file, path)
    meta = {"revision": compat_mod.sha256_short(path), "license": None, "split_sizes": {"all": len(df)}}
    # выборка по всему файлу, а не первые строки: файл может быть отсортирован по метке
    sample = df.sample(n=min(2000, len(df)), random_state=0).reset_index(drop=True)
    user_spec = spec.model_copy(update={"constraints": spec.constraints.model_copy(update={"commercial_license": False})})
    cand = compat_mod.candidate_from_sample(
        f"user:{path.name}", user_spec, task, sample, meta, description="Датасет пользователя",
        source="user", local_path=str(path))
    if cand.compat and (cand.compat.ok or cand.compat.needs_mapping):
        try:
            cand = await assess(spec, task, cand, sample, cfg.search.criteria_weights, cfg.search.llm_examples)
        except Exception as e:
            logger.warning("LLM-оценка пользовательского датасета не удалась: %s", e)
    return cand
