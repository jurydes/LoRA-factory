"""A3 HParam: подбирает гиперпараметры по похожим прошлым запускам и заметкам; каждое решение
обосновано ссылкой на свидетельство (run_id или doc_chunk_id). Выход проходит guard."""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits

from factory.core import llm
from factory.core.registry import Registry
from factory.hparams import history, resources
from factory.hparams.guard import HParamContext, StrategyChoice, assemble
from factory.hparams.pilot import run_pilot as _run_pilot
from factory.runs import RunPaths


@dataclass
class HParamDeps:
    ctx: HParamContext
    paths: RunPaths
    baselines_val: dict
    seen_runs: set[str] = field(default_factory=set)
    seen_chunks: set[str] = field(default_factory=set)
    pilots: int = 0


INSTRUCTIONS = """Ты подбираешь гиперпараметры дообучения адаптера. Ты решаешь только: lr, число эпох (верхняя граница —
работает ранний останов), для LoRA — r, alpha, вариант (lora/rslora/dora) и набор модулей (attention/all_linear);
для P-tuning — число виртуальных токенов, размер и тип энкодера. Квантизацию, батч, длину последовательности и имена
модулей решает код. Сначала вызови find_similar_runs и lookup_hparam_docs. Каждый выбранный параметр обоснуй в rationales
с evidence: kind="run" и ref=<run_id> из find_similar_runs, или kind="doc" и ref=<doc_chunk_id> из lookup_hparam_docs.
Если похожих запусков нет — так и напиши (cold start) и опирайся на документацию. run_pilot — не больше одного раза."""

# Эвристики по типу задачи (сводка data/docs/hparams/tasks.md и general_sources.md: QLoRA, Unsloth, LoRA, практика)
TASK_HEURISTICS = {
    "text_classification": "короткий ответ-метка: r 8–16, attention обычно достаточно (all_linear — для трудных классов), "
                           "lr 1e-4…2e-4, 2–4 эпохи (большой датасет — 1–2); риск — схлопывание в мажоритарный класс",
    "sentiment_analysis": "задача проще для базовой модели: r 8, lr 1e-4, 1–3 эпохи (> 20 тыс. примеров — 1)",
    "summarization": "длинный вход и выход: all_linear, r 16–64, alpha 2r, lr 1e-4…2e-4, 1–3 эпохи",
    "text_generation": "перенос стиля: all_linear, r 16–32, lr 2e-4, 2–4 эпохи",
    "ner": "строгий формат и границы сущностей: all_linear, r 16–32, lr 1e-4…2e-4, 3–5 эпох (малые данные — до 8)",
    "qa": "экстрактивный ответ из длинного контекста: all_linear, r 16, lr 1e-4…2e-4, 1–3 эпохи (большие датасеты — 1–2)",
}
SIZE_HEURISTICS = [
    (1_000, "мало данных (< 1 тыс.): r 8, lr 2e-4, до 4 эпох с ранним остановом"),
    (10_000, "1–10 тыс. примеров: r 16, lr 2e-4, 3 эпохи"),
    (50_000, "10–50 тыс.: r 32, lr 2e-4, 2 эпохи"),
    (float("inf"), "больше 50 тыс.: r 32, lr 2e-4, 1 эпоха"),
]
GENERAL = ("общие правила: lr LoRA ≈ 10× lr полного дообучения и не снижается с ростом данных (крупные модели — 1e-4); "
           "все линейные слои, а не только attention; alpha = r или 2r; эффективный батч 8–16; rsLoRA при r ≥ 64; "
           "P-tuning — lr 1e-3…1e-2, 20–100 токенов, 5–15 эпох, на трудных задачах (NER, QA) малым моделям уступает LoRA")

agent = Agent(deps_type=HParamDeps, output_type=StrategyChoice, instructions=INSTRUCTIONS, retries={"output": 3, "tools": 3})


@agent.instructions
def _context(ctx: RunContext[HParamDeps]) -> str:
    c = ctx.deps.ctx
    return (f"Задача: {c.spec.task_type} ({c.task.family}), язык {c.spec.language}, меток: {len(c.spec.labels or [])}.\n"
            f"Модель: {c.model.id} ({c.model.params_b} млрд), адаптер: {c.adapter_kind}.\n"
            f"Train: {c.n_train} строк; p95 длины входа ≈ {c.profile.input_tokens_est.p95:.0f} ток., ответа ≈ "
            f"{c.profile.target_tokens_est.p95:.0f} ток. VRAM: {c.vram_gb:.1f} ГБ.\n"
            f"Baseline'ы на val ({c.task.primary_metric}): "
            f"{ {k: v.get(c.task.primary_metric) for k, v in ctx.deps.baselines_val.items()} }\n"
            f"Эвристики для этой задачи: {TASK_HEURISTICS.get(c.spec.task_type, '—')}.\n"
            f"Для этого объёма данных: {next(h for limit, h in SIZE_HEURISTICS if c.n_train < limit)}.\n"
            f"{GENERAL}. Эвристики — отправная точка: подкрепляй выбор фрагментами lookup_hparam_docs и похожими запусками; "
            f"если похожий запуск дал хороший результат на val — ориентируйся на его параметры.")


@agent.tool
def find_similar_runs(ctx: RunContext[HParamDeps], top: int = 5) -> dict:
    """Похожие завершённые запуски (тип задачи, язык, размер модели, число примеров и классов) с метриками на val."""
    c = ctx.deps.ctx
    runs = history.find_similar_runs(c.cfg.runs_dir, Registry.load(c.cfg), c.spec.task_type, c.spec.language, c.model.params_b,
                                     c.n_train, len(c.spec.labels or []), top=min(top, 10))
    ctx.deps.seen_runs |= {r["run_id"] for r in runs}
    return {"runs": runs, "cold_start": not runs}


@agent.tool
def lookup_hparam_docs(ctx: RunContext[HParamDeps], query: str) -> list[dict]:
    """Поиск по заметкам/документации по гиперпараметрам; у каждого фрагмента doc_chunk_id."""
    chunks = history.lookup_docs(ctx.deps.ctx.cfg.hparam_docs_dir, query)
    ctx.deps.seen_chunks |= {c["doc_chunk_id"] for c in chunks}
    return chunks


@agent.tool
def estimate_resources(ctx: RunContext[HParamDeps], micro_batch: int = 4, seq_len: int = 512,
                       quantization: str = "none", gradient_checkpointing: bool = False) -> dict:
    """Грубая оценка пиковой памяти (ГБ) и шагов на эпоху для заданного батча и длины."""
    c = ctx.deps.ctx
    est = resources.estimate_memory(c.model, seq_len, micro_batch, quantization, gradient_checkpointing)
    return {"memory_gb": est, "vram_gb": round(c.vram_gb, 2),
            "steps_per_epoch_at_effective_16": resources.steps_per_epoch(c.n_train, 16, 1)}


@agent.tool
async def run_pilot(ctx: RunContext[HParamDeps], choice: StrategyChoice) -> dict:
    """Короткий пробный прогон предлагаемой конфигурации: NaN/OOM, скорость, память, оценка времени."""
    if ctx.deps.pilots >= 1:
        return {"error": "пилот уже запускался; финальный пилот выполнит код после твоего ответа"}
    ctx.deps.pilots += 1
    cfg = assemble(choice, ctx.deps.ctx, "agent")
    report = await _run_pilot(ctx.deps.paths, cfg, ctx.deps.ctx.n_train, ctx.deps.ctx.cfg.hparams.pilot)
    return {"status": report.status, "fixes": report.fixes, "result": report.result}


@agent.output_validator
def _check_evidence(ctx: RunContext[HParamDeps], out: StrategyChoice) -> StrategyChoice:
    deps = ctx.deps
    kind = deps.ctx.adapter_kind
    if kind == "lora" and out.lora is None:
        raise ModelRetry("Для этой задачи нужен блок lora (адаптер LoRA)")
    if kind == "ptuning" and out.ptuning is None:
        raise ModelRetry("Для этой задачи нужен блок ptuning")
    covered = {r.param.split(".")[0] for r in out.rationales}
    missing = [p for p in ("lr", "epochs", kind) if p not in covered]
    if missing:
        raise ModelRetry(f"Нет обоснований для параметров: {missing}")
    for r in out.rationales:
        for e in r.evidence:
            if e.kind == "run" and e.ref not in deps.seen_runs:
                raise ModelRetry(f"run_id {e.ref} не встречался в find_similar_runs")
            if e.kind == "doc" and e.ref not in deps.seen_chunks:
                raise ModelRetry(f"doc_chunk_id {e.ref} не встречался в lookup_hparam_docs")
    return out


async def run(deps: HParamDeps) -> StrategyChoice:
    cfg = deps.ctx.cfg.hparams
    result = await llm.run_agent(agent, "hparam", "Подбери гиперпараметры.", deps=deps,
                                 usage_limits=UsageLimits(request_limit=cfg.request_limit, tool_calls_limit=cfg.tool_calls_limit))
    return result.output
