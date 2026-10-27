"""Общий LLM-клиент: единственная точка создания моделей pydantic_ai по ролям из config.yaml.

Каждый вызов пишется в llm_calls.jsonl текущего контекста (сессии или запуска): роль, хэш промпта,
токены, время. Сетевые ошибки повторяются с экспоненциальной паузой. Настройки сервера берутся из
кода курсовой (settings.load_global_config → dev.env).
"""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
import os
import re
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TypeVar

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

import httpx
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.vllm import VLLMProvider
from pydantic_ai.settings import ModelSettings

from factory.core import activity
from factory.core.config import get_config
from factory.core.storage import append_jsonl

logger = logging.getLogger(__name__)

T = TypeVar("T")

_log_path: contextvars.ContextVar[Path | None] = contextvars.ContextVar("llm_log_path", default=None)
_model_override: contextvars.ContextVar[Model | None] = contextvars.ContextVar("llm_model_override", default=None)


@contextmanager
def llm_log_to(path: Path | None) -> Iterator[None]:
    """Направляет журнал LLM-вызовов в указанный файл на время блока."""
    token = _log_path.set(path)
    try:
        yield
    finally:
        _log_path.reset(token)


@contextmanager
def override_model(model: Model) -> Iterator[None]:
    """Подменяет модель всех ролей (тесты: TestModel / FunctionModel)."""
    token = _model_override.set(model)
    try:
        yield
    finally:
        _model_override.reset(token)


def _server_settings():
    # импорт кода курсовой: dev.env с адресом OpenAI-совместимого сервера
    from settings import load_global_config
    return load_global_config().openai_server


def interface_address(name: str) -> str:
    """IPv4-адрес сетевого интерфейса по имени (или сам адрес, если передан IP)."""
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", name):
        return name
    import psutil
    addrs = psutil.net_if_addrs()
    ipv4 = [a.address for a in addrs.get(name, []) if a.family == socket.AF_INET]
    if not ipv4:
        raise ValueError(f"У сетевого интерфейса «{name}» нет IPv4-адреса; интерфейсы: {sorted(addrs)}")
    return ipv4[0]


_bound_clients: dict[tuple[str, int], Any] = {}


def _bound_http_client(local_address: str):
    """HTTP-клиент, чьи соединения исходят с адреса физического интерфейса: Windows отправляет их через этот
    интерфейс, и VPN в режиме TUN (маршрут по умолчанию на туннель) их не перехватывает. Один клиент на цикл
    событий: пул соединений привязан к циклу, в котором создан."""
    import httpx2
    key = (local_address, id(asyncio.get_running_loop()))
    client = _bound_clients.get(key)
    if client is None or client.is_closed:
        client = httpx2.AsyncClient(transport=httpx2.AsyncHTTPTransport(local_address=local_address),
                                    timeout=httpx2.Timeout(600, connect=5))
        _bound_clients[key] = client
    return client


def get_model(role: str) -> Model:
    override = _model_override.get()
    if override is not None:
        return override
    server = _server_settings()
    role_cfg = get_config().llm.roles.get(role) or get_config().llm.roles["orchestrator"]
    name = role_cfg.model or server.default_model
    http_client = _bound_http_client(interface_address(server.bind_interface)) if server.bind_interface else None
    provider = VLLMProvider(base_url=str(server.url), api_key=server.key, http_client=http_client)
    return OpenAIChatModel(name, provider=provider)


def model_settings(role: str) -> ModelSettings:
    cfg = get_config().llm
    role_cfg = cfg.roles.get(role) or cfg.roles["orchestrator"]
    settings: dict[str, Any] = {
        "temperature": role_cfg.temperature,
        "max_tokens": role_cfg.max_tokens,
        "timeout": cfg.timeout_s,
    }
    if cfg.disable_thinking:
        settings["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return settings


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ModelHTTPError):
        return exc.status_code >= 500 or exc.status_code == 429
    return isinstance(exc, (httpx.TransportError, ModelAPIError, asyncio.TimeoutError))


def _log_call(role: str, prompt: str, started: float, usage=None, error: str | None = None, extra: dict | None = None):
    path = _log_path.get()
    if path is None:
        return
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "role": role,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "elapsed_s": round(time.monotonic() - started, 3),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "requests": getattr(usage, "requests", None),
        "tool_calls": getattr(usage, "tool_calls", None),
        "ok": error is None,
        "error": error,
    }
    if extra:
        record.update(extra)
    append_jsonl(path, record)


async def run_agent(
    agent: Agent,
    role: str,
    prompt: str,
    *,
    deps: Any = None,
    conversation: Any = None,
    usage_limits: UsageLimits | None = None,
):
    """Запускает агента моделью роли; повторяет при сетевых ошибках; пишет вызов в журнал.
    Каждый шаг (запрос к модели, вызовы инструментов и их итоги) уходит в ленту активности.

    Повтор продолжает прогон с места сбоя по истории сообщений, а не начинает заново: инструменты с последствиями
    (команды A4 живому процессу, операции A2) не выполняются второй раз. С объектом диалога (A1) — перезапуск."""
    retries = get_config().llm.max_retries
    started = time.monotonic()
    history, usage, requests = None, None, 0
    for attempt in range(retries + 1):
        run = None
        try:
            async with agent.iter(None if history else prompt, message_history=history, usage=usage,
                                  deps=deps, conversation=conversation, usage_limits=usage_limits,
                                  model=get_model(role), model_settings=model_settings(role)) as run:
                async for node in run:
                    requests += Agent.is_model_request_node(node)
                    await activity.agent_node(role, node, requests)
            result = run.result
        except Exception as exc:
            if attempt < retries and _is_transient(exc):
                if run is not None and conversation is None:
                    history, usage = run.all_messages() or None, run.usage
                delay = 2 ** attempt
                logger.warning("LLM [%s]: сетевая ошибка (%s), повтор через %ss", role, exc, delay)
                await asyncio.sleep(delay)
                continue
            _log_call(role, prompt, started, error=f"{type(exc).__name__}: {exc}")
            raise
        _log_call(role, prompt, started, usage=result.usage)
        return result
    raise AssertionError("unreachable")


_stateless_agents: dict[tuple[str, Any], Agent] = {}


async def complete(role: str, prompt: str, output_type: type[T] = str, instructions: str | None = None) -> T:
    """Вызов без состояния (расширение запроса, оценка датасета, синтез): промпт → текст или pydantic-модель."""
    key = (instructions or "", output_type)
    agent = _stateless_agents.get(key)
    if agent is None:
        agent = Agent(output_type=output_type, instructions=instructions, retries={"output": 2})
        _stateless_agents[key] = agent
    result = await run_agent(agent, role, prompt)
    output = result.output
    if isinstance(output, str):
        from generation import (
            separate_thinking,  # постобработка ответа из кода курсовой
        )
        output = separate_thinking(output)
    return output
