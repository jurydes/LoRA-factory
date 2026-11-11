"""A1 Intake: уточняет постановку, пока она не станет однозначной, и представляет найденные датасеты.

Один ход пользователя = один вызов agent.run; история хранится между ходами (conversation.json).
Метрики в TaskSpec подставляет код из реестра задач; базовую модель LLM выбирает из отфильтрованного кодом списка."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits

from factory.core import llm
from factory.core.contracts import Constraints, DatasetCandidate, TaskSpec
from factory.core.registry import Registry


class AskUser(BaseModel):
    """Задать пользователю уточняющий вопрос."""
    question: str
    options: list[str] = Field(default_factory=list, description="Варианты ответа-кнопки (можно пусто)")


class SpecDraft(BaseModel):
    description: str = Field(description="Постановка своими словами, 1–3 предложения")
    task_type: str = Field(description="Ключ типа задачи из list_task_types")
    language: str = "ru"
    labels: list[str] | None = Field(default=None, description="Набор меток (обязателен для классификации)")
    base_model_id: str = Field(description="id модели из list_models")
    constraints: Constraints = Field(default_factory=Constraints)


class ProposeSpec(BaseModel):
    """Предложить постановку карточкой: пользователь правит и подтверждает."""
    spec: SpecDraft
    comment: str = ""


class PresentDatasets(BaseModel):
    """Показать найденные датасеты (только из результатов search_datasets)."""
    hf_ids: list[str]
    message: str


SearchFn = Callable[[TaskSpec, str | None], Awaitable[list[DatasetCandidate]]]


@dataclass
class IntakeDeps:
    registry: Registry
    spec: TaskSpec | None = None
    search: SearchFn | None = None
    candidates: list[DatasetCandidate] = field(default_factory=list)
    finalized_spec: TaskSpec | None = None
    hardware: Any = None  # factory.core.hardware.Hardware


INSTRUCTIONS = """Ты помогаешь пользователю дообучить языковую модель под NLP-задачу. Твоя работа на этом этапе —
уточнить постановку, пока она не станет однозначной: тип задачи (из list_task_types), язык данных, набор меток
(для классификации; для NER — типы сущностей), ограничения (бюджет GPU-часов, размер модели, лицензия, метод адаптации:
auto/lora/qlora/ptuning, целевая метрика, допустимая задержка ответа) и базовую модель — только из list_models: он учитывает
VRAM, RAM и скорость машины пользователя; по умолчанию бери рекомендованную и коротко объясни выбор. Метрики не обсуждай и не придумывай — их задаёт система.
Задавай по одному короткому вопросу за раз (AskUser), предлагай варианты-кнопки. Когда всё ясно — ProposeSpec.
Не спрашивай то, что можно разумно принять по умолчанию (язык ru, метод auto, самая подходящая модель из списка).
После подтверждения постановки ищи датасеты инструментом search_datasets и отвечай PresentDatasets; если ничего не
подошло — AskUser с предложением переформулировать задачу или ослабить ограничения. Пиши по-русски."""

# tools: 3 — поиск до подтверждения постановки отвечает ModelRetry; с одним повтором прогон падал целиком
agent = Agent(deps_type=IntakeDeps, output_type=[AskUser, ProposeSpec, PresentDatasets], instructions=INSTRUCTIONS,
              retries={"output": 2, "tools": 3})


@agent.instructions
def _state(ctx: RunContext[IntakeDeps]) -> str:
    if ctx.deps.spec and ctx.deps.spec.confirmed_by_user:
        return f"Подтверждённая постановка: {ctx.deps.spec.model_dump_json(exclude={'metrics'})}"
    return "Постановка ещё не подтверждена."


@agent.tool
def list_task_types(ctx: RunContext[IntakeDeps]) -> list[dict]:
    """Поддерживаемые типы задач (ключ, название, семейство)."""
    return [{"key": t.key, "title": t.title, "family": t.family} for t in ctx.deps.registry.supported_tasks()]


@agent.tool
def list_models(ctx: RunContext[IntakeDeps], language: str = "ru", max_params_b: float | None = None,
                adapter_method: str = "auto", commercial_license: bool = False, gpu_hours: float | None = None) -> dict:
    """Базовые модели, которые подходят под ограничения И помещаются на машину пользователя (фильтрует код).
    mode: lora — обучение в bf16, qlora — только в 4 битах; hours_typical — оценка на 10 тыс. примеров × 3 эпохи;
    quality — MMLU-Pro модели (общее качество; размер сам по себе не довод: новое поколение сильнее старого того же
    размера); recommended — лучшая LoRA-модель среди укладывающихся в память и время. Если LoRA не подходит,
    рекомендуется QLoRA. Предлагай рекомендованную по умолчанию.
    hardware.vram_others_gb — видеопамять, которую сейчас держат другие программы: она вычтена; если её много (>1 ГБ),
    скажи, что после закрытия этих программ станет доступна модель крупнее."""
    cons = Constraints(max_model_params_b=max_params_b, adapter_method=adapter_method,
                       commercial_license=commercial_license, gpu_hours=gpu_hours)
    hw = ctx.deps.hardware
    rows = ctx.deps.registry.available_models(cons, language, hw)
    for r in rows:
        r["gated"] = ctx.deps.registry.model(r["id"]).gated
    return {"hardware": hw.model_dump(include={"gpu_name", "vram_gb", "vram_others_gb", "ram_gb", "measured_tflops"}) if hw else None,
            "models": rows}


@agent.tool
async def search_datasets(ctx: RunContext[IntakeDeps], extra_query: str | None = None) -> list[dict]:
    """Поиск датасетов под подтверждённую постановку. extra_query — уточнение формулировки поиска."""
    deps = ctx.deps
    if not deps.spec or not deps.spec.confirmed_by_user or deps.search is None:
        raise ModelRetry("Поиск доступен только после подтверждения постановки: сначала ответь ProposeSpec "
                         "и дождись подтверждения пользователя, поиск не повторяй")
    found = await deps.search(deps.spec, extra_query)
    known = {c.hf_id for c in deps.candidates}
    deps.candidates += [c for c in found if c.hf_id not in known]
    return [{"hf_id": c.hf_id, "score": c.score, "why": c.why, "issues": c.compat.issues if c.compat else []}
            for c in found]


def finalize_spec(draft: SpecDraft, registry: Registry, hw=None) -> TaskSpec:
    """SpecDraft → TaskSpec: метрики из реестра, модель — только из отфильтрованного списка (ограничения + железо)."""
    allowed = {r["id"] for r in registry.available_models(draft.constraints, draft.language, hw)}
    if draft.base_model_id not in allowed:
        raise ValueError(f"Модель {draft.base_model_id} не подходит под ограничения или железо; доступны: {sorted(allowed)}")
    spec = TaskSpec(**draft.model_dump())
    return registry.apply_task_metrics(spec)


@agent.output_validator
def _validate(ctx: RunContext[IntakeDeps], out):
    if isinstance(out, ProposeSpec):
        try:
            ctx.deps.finalized_spec = finalize_spec(out.spec, ctx.deps.registry, ctx.deps.hardware)
        except (ValueError, KeyError) as e:
            raise ModelRetry(str(e))
    elif isinstance(out, PresentDatasets):
        known = {c.hf_id for c in ctx.deps.candidates}
        unknown = [i for i in out.hf_ids if i not in known]
        if unknown or not out.hf_ids:
            raise ModelRetry(f"Можно показывать только найденные search_datasets датасеты; неизвестны: {unknown}")
    return out


async def run_turn(deps: IntakeDeps, prompt: str, conversation=None):
    """Один ход диалога. Возвращает (выход, новая conversation)."""
    result = await llm.run_agent(agent, "intake", prompt, deps=deps, conversation=conversation,
                                 usage_limits=UsageLimits(request_limit=12, tool_calls_limit=10))
    return result.output, result.conversation
