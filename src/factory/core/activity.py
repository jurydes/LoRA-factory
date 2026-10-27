"""Лента «что система делает сейчас»: шаги агентов (запрос к модели, вызов инструмента с аргументами, итог вызова)
и запуски рабочих процессов. Получатель задаётся на время фоновой операции (оркестратор пишет ленту в сессию);
без получателя события никуда не идут. Ошибка получателя не прерывает операцию."""
from __future__ import annotations

import contextvars
import inspect
import logging
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Iterator, Literal

logger = logging.getLogger(__name__)

Kind = Literal["step", "live", "result", "idle"]
# (вид, кто, текст, ключ): step — действие; live — временная строка «думает»; result — итог действия с ключом;
# idle — агент закончил шаг без действий (снять «думает»)
Sink = Callable[[Kind, str, str, str | None], Awaitable[None] | None]
_sink: contextvars.ContextVar[Sink | None] = contextvars.ContextVar("activity_sink", default=None)


@contextmanager
def report_to(sink: Sink | None) -> Iterator[None]:
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


async def emit(text: str, who: str = "", kind: Kind = "step", key: str | None = None) -> None:
    sink = _sink.get()
    if sink is None:
        return
    try:
        r = sink(kind, who, text, key)
        if inspect.isawaitable(r):
            await r
    except Exception as e:
        logger.debug("Лента активности: %s", e)


ROLES = {
    "intake": "A1 · диалог", "dataprep": "A2 · подготовка данных", "hparam": "A3 · гиперпараметры",
    "monitor": "A4 · мониторинг", "query_expansion": "LLM · расширение запроса", "dataset_eval": "LLM · оценка датасета",
    "augment": "LLM · синтез данных", "orchestrator": "LLM",
}

GOALS = {
    "no_empty_rows": "пустые строки", "no_exact_duplicates": "дубли", "near_duplicates": "почти-дубли",
    "junk_text": "мусор", "length_outliers": "выбросы длины", "language": "язык", "class_balance": "баланс классов",
    "labels_in_spec": "метки вне постановки", "label_noise": "шум меток",
}

_FIELDS = {"input_len": "длине входа", "target_len": "длине ответа", "language": "языку", "label": "метке",
           "is_dupe": "признаку дубля"}


def _op(a: dict, what: str) -> str:
    return ("применяет " if a.get("apply") else "примеряет ") + what


def _why(a: dict) -> str:
    return f" — {a['reason']}" if a.get("reason") else ""


def _choice(c: Any) -> str:
    if not isinstance(c, dict):
        return ""
    lora = c.get("lora") or {}
    parts = [f"lr {c['lr']:g}" if isinstance(c.get("lr"), (int, float)) else "",
             f"{c['epochs']} эп." if c.get("epochs") else "", f"r={lora['r']}" if lora.get("r") else ""]
    return f" ({', '.join(p for p in parts if p)})" if any(parts) else ""


TOOLS: dict[str, str | Callable[[dict], str]] = {
    # A1
    "list_task_types": "смотрит поддерживаемые задачи",
    "list_models": lambda a: "подбирает модели под железо" + (f" (до {a['max_params_b']}B)" if a.get("max_params_b") else ""),
    "search_datasets": lambda a: "ищет датасеты" + (f": «{a['extra_query']}»" if a.get("extra_query") else ""),
    # A2
    "check_quality": "проверяет цели качества",
    "profile": "смотрит профиль выборки",
    "sample": lambda a: "смотрит примеры" + (f" метки «{a['label']}»" if a.get("label") else ""),
    "inspect_label": lambda a: f"изучает метку «{a.get('label')}»",
    "find_examples": lambda a: f"ищет строки по шаблону «{a.get('pattern')}»",
    "junk_examples": "смотрит строки с мусором",
    "length_outliers": "смотрит выбросы по длине",
    "near_duplicates": lambda a: f"ищет почти-дубли (порог {a.get('threshold', 0.9)})",
    "language_report": "проверяет язык текстов",
    "label_issues": "ищет подозрительные метки",
    "diff": "сравнивает выборку до и после операций",
    "clean_text": lambda a: _op(a, "чистку текста"),
    "map_labels": lambda a: _op(a, "переименование меток"),
    "dedupe": lambda a: _op(a, "удаление почти-дублей" if a.get("mode") == "near" else "удаление дублей"),
    "filter": lambda a: _op(a, f"фильтр по {_FIELDS.get(a.get('field'), a.get('field'))} {a.get('operator', '')} {a.get('value', '')}".rstrip()),
    "drop_label_noise": lambda a: _op(a, "удаление шумных меток"),
    "rebalance": lambda a: _op(a, "балансировку классов"),
    "augment": lambda a: f"синтез {a.get('n_rows')} строк ({a.get('method')})" + _why(a),
    "undo": "отменяет последнюю операцию",
    "commit_recipe": "фиксирует рецепт",
    # A3
    "find_similar_runs": "ищет похожие прошлые запуски",
    "lookup_hparam_docs": lambda a: f"читает заметки: «{a.get('query')}»",
    "estimate_resources": lambda a: f"оценивает память (batch {a.get('micro_batch', 4)}, длина {a.get('seq_len', 512)})",
    "run_pilot": lambda a: "запускает пробный прогон" + _choice(a.get("choice")),
    # A4
    "get_recent_metrics": "смотрит последние метрики",
    "get_val_curve": "смотрит кривую на val",
    "get_training_config": "смотрит конфигурацию обучения",
    "get_event_history": "смотрит историю событий",
    "wait_for_new_metrics": lambda a: f"ждёт новых метрик (до {a.get('seconds', 60)} с)",
    "scale_lr": lambda a: f"меняет lr ×{a.get('factor')}" + _why(a),
    "set_grad_clip": lambda a: f"ограничивает норму градиента до {a.get('max_norm')}" + _why(a),
    "set_eval_every": lambda a: f"ставит оценку каждые {a.get('steps')} шагов" + _why(a),
    "set_log_every": lambda a: f"ставит лог каждые {a.get('steps')} шагов" + _why(a),
    "request_eval_now": lambda a: "запрашивает оценку на val" + _why(a),
    "save_checkpoint_now": lambda a: "сохраняет чекпоинт" + _why(a),
    "stop_training": lambda a: "останавливает обучение" + _why(a),
}


def describe(tool: str, args: dict) -> str:
    d = TOOLS.get(tool)
    if d is None:
        return f"вызывает {tool}"
    try:
        return d if isinstance(d, str) else d(args)
    except Exception:
        return f"вызывает {tool}"


def summarize(content: Any) -> str | None:
    """Короткий итог вызова инструмента по его ответу (если он о чём-то говорит)."""
    if isinstance(content, list):
        return f"{len(content)} шт." if content else "пусто"
    if not isinstance(content, dict):
        return None
    if "error" in content:
        return str(content["error"])
    if "command" in content:  # команда A4 процессу обучения
        return f"применено на шаге {content.get('at_step')}" if content.get("applied") else "процесс не подтвердил команду"
    if "failing" in content:
        bad = [GOALS.get(g, g) for g in content["failing"]]
        return "не выполнено: " + ", ".join(bad) if bad else "все цели выполнены"
    if content.get("applied") is False:
        return "ничего не меняет"
    if "rows_before" in content and "rows_after" in content:
        return f"строк {content['rows_before']} → {content['rows_after']}"
    if "would_remove" in content:
        return f"удалит {content['would_remove']} строк" if content["would_remove"] else "строки не удаляются"
    if "committed" in content:
        return "рецепт принят"
    for k, label in (("matches", "найдено"), ("count", "найдено"), ("n_suspicious", "подозрительных"),
                     ("generated", "сгенерировано"), ("new_points", "новых точек")):
        if k in content:
            v = content[k]
            return f"{label} {len(v) if isinstance(v, list) else v}"
    if "status" in content and "result" in content:  # пилот A3
        return "пилот прошёл" if content["status"] == "ok" else f"пилот: {content['status']}"
    return None


async def agent_node(role: str, node: Any, request_no: int) -> None:
    """Узел графа pydantic_ai → строки ленты. Вызывается до выполнения узла: запрос к модели — «думает»
    (с итогами инструментов прошлого шага), ответ модели — вызовы инструментов с аргументами."""
    from pydantic_ai import Agent
    from pydantic_ai.messages import (
        RetryPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )

    who = ROLES.get(role, role)
    if Agent.is_model_request_node(node):
        for part in node.request.parts:
            if isinstance(part, ToolReturnPart):
                note = summarize(part.content)
                if note:
                    await emit(note, who, "result", part.tool_call_id)
            elif isinstance(part, RetryPromptPart) and part.tool_name:
                text = part.content if isinstance(part.content, str) else "неверные аргументы"
                await emit("отклонено: " + text[:160], who, "result", part.tool_call_id)
        await emit("думает" + (f" (запрос {request_no})" if request_no > 1 else ""), who, "live")
    elif Agent.is_call_tools_node(node):
        parts = node.model_response.parts
        calls = [p for p in parts if isinstance(p, ToolCallPart) and not p.tool_name.startswith("final_result")]
        if not calls:
            await emit("", who, "idle")
            return
        for p in parts:  # пояснение модели рядом с вызовами — её ход мысли
            if isinstance(p, TextPart) and p.content.strip():
                text = " ".join(p.content.split())
                await emit("«" + (text[:200] + "…" if len(text) > 200 else text) + "»", who)
        for c in calls:
            try:
                args = c.args_as_dict()
            except Exception:
                args = {}
            await emit(describe(c.tool_name, args), who, key=c.tool_call_id)
