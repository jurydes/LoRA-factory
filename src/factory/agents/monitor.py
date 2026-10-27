"""A4 Monitor: наблюдает за живым обучением и управляет им через инструменты, пишет диагноз раунда и итоговый.

Полномочия ограничены: число вмешательств лимитировано, патчи — из белого списка (только уменьшение lr,
клиппинг градиента), менять модель, датасет или test нельзя. Оценку «достаточно» агент может только ужесточить."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.tools import ToolDefinition

from factory.core import llm
from factory.core.contracts import MonitorDecision, RoundChoice
from factory.training.rounds import check_placeholders


class LiveRun(Protocol):
    """Живой процесс обучения: наблюдение и команды (реализует Supervisor)."""
    def recent_metrics(self, n: int) -> list[dict]: ...
    def val_curve(self) -> list[dict]: ...
    def config_summary(self) -> dict: ...
    def event_history(self, n: int) -> list[dict]: ...
    async def wait_new(self, seconds: float) -> list[dict]: ...
    async def send(self, cmd: str, patch: dict | None, decision: MonitorDecision) -> dict: ...


@dataclass
class MonitorDeps:
    task_type: str
    primary_metric: str
    allowed_actions: list[str] = field(default_factory=lambda: ["continue", "patch", "stop_early"])
    interventions_left: int = 3
    values: dict[str, float] = field(default_factory=dict)  # доступные плейсхолдеры
    code_sufficient: bool = False
    live: LiveRun | None = None
    actions: list[MonitorDecision] = field(default_factory=list)
    eval_steps: int = 400
    waited: bool = False
    observations: int = 0
    intervened: bool = False  # в этой проверке уже была поправка оптимизации
    read: set[str] = field(default_factory=set)  # инструменты чтения, уже вызванные в этой проверке


class MonitorVerdict(BaseModel):
    verdict: Literal["ok", "suspicious", "broken"] = Field(description="Как идёт обучение")
    explanation: str = Field(description="Коротко для пользователя: что видно и что сделано")


class RoundAssessment(BaseModel):
    diagnosis: str = Field(description="Диагноз раунда для пользователя; числа — только плейсхолдерами {…}")
    recommendation: RoundChoice
    extra_epochs: int | None = Field(default=None, ge=1, le=20, description="Для continue: сколько эпох добавить")


class FinalDiagnosis(BaseModel):
    diagnosis: str = Field(description="Итоговый диагноз; числа — только плейсхолдерами {…}")


INSTRUCTIONS = """Ты — агент мониторинга дообучения LLM-адаптера. Ты оцениваешь, корректно ли идёт обучение,
и принимаешь ограниченные решения. Нельзя менять модель, датасет или test. Решения о продолжении принимаются по val.
Будь сдержан: вмешивайся, только если свидетельства убедительны.
Два вида действий. Поправки оптимизации (scale_lr, set_grad_clip) ограничены: не больше одной за проверку и не больше
остатка вмешательств — после поправки сначала смотри её эффект. Остановка (stop_training) не поправка и под этот лимит
не подпадает: пока она в списке разрешённых, её можно применить всегда, независимо от поправок и остатка вмешательств.
Остановка безопасна: чекпоинт сохраняется, итоговым остаётся лучший адаптер по val, а не последний.
Переобучение: метрика на val достигла пика и после него две оценки подряд хуже пика, val loss растёт, а train loss
падает — продолжать нет смысла (лучший адаптер уже сохранён) → stop_training. Если метрики лишь колеблются в пределах
шума или после пика прошла одна оценка — подожди. Остановку выбирай, а не «отмечай подозрительный статус»: вердикт
suspicious без действия оставляет обучение жечь время зря.
Падение скорости (токены/с) — дело железа и других программ на GPU, а не оптимизации: lr и клиппинг его не лечат. Пиши по-русски, коротко и по делу.
Числовые значения в текстах для пользователя пиши плейсхолдерами в фигурных скобках из списка доступных, например
{val.macro_f1} или {baseline.majority.macro_f1}, — код подставит точные значения."""

# Один агент мониторинга — три задачи с разными типами выхода (в pydantic_ai тип выхода с валидатором задаётся
# при создании агента, поэтому это три объекта Agent с общими инструкциями и ролью «monitor»).
live_agent = Agent(deps_type=MonitorDeps, output_type=MonitorVerdict, retries={"output": 2, "tools": 4}, instructions=INSTRUCTIONS + """
Сейчас обучение ИДЁТ. Сначала посмотри данные инструментами (метрики, кривая val, конфигурация, история), затем, если
нужно, вмешайся инструментами-действиями — они сразу применяются к живому процессу, и ты увидишь, подтвердил ли он
команду. Действия: scale_lr (только уменьшение), set_grad_clip, set_eval_every / set_log_every (плотнее наблюдать),
request_eval_now, save_checkpoint_now, stop_training (досрочно, с сохранением). Если всё в порядке — ничего не меняй.
Каждый инструмент чтения вызывай не больше одного раза (повторный вызов вернёт то же самое и скрыт): прочитай нужное
за один заход, реши, затем сразу заверши ответом (вердикт и короткое объяснение). После действия можно один раз
вызвать wait_for_new_metrics и проверить эффект. В этом режиме числа в объяснении пиши обычными цифрами,
без плейсхолдеров в фигурных скобках.""")
round_agent = Agent(deps_type=MonitorDeps, output_type=RoundAssessment, instructions=INSTRUCTIONS, retries={"output": 2})
final_agent = Agent(deps_type=MonitorDeps, output_type=FinalDiagnosis, instructions=INSTRUCTIONS, retries={"output": 2})


def _live(ctx: RunContext[MonitorDeps], observe: str | None = None) -> LiveRun:
    if ctx.deps.live is None:
        raise ModelRetry("Нет живого процесса обучения")
    if observe:
        ctx.deps.observations += 1
        ctx.deps.read.add(observe)
    return ctx.deps.live


OBSERVE_LIMIT = 8  # чтений за проверку


async def _observing(ctx: RunContext[MonitorDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    """Каждое чтение — один раз за проверку: повторный вызов вернул бы то же самое (модель вызывала кривую val по 4–5
    раз подряд). После ожидания эффекта или лимита чтений инструменты наблюдения скрыты: модель действует или отвечает."""
    if ctx.deps.waited or ctx.deps.observations >= OBSERVE_LIMIT or tool_def.name in ctx.deps.read:
        return None
    return tool_def


@live_agent.tool(prepare=_observing)
def get_recent_metrics(ctx: RunContext[MonitorDeps], n: int = 20) -> list[dict]:
    """Последние точки метрик: step, epoch, train/val loss, lr, grad_norm, токены/с, память GPU, метрики на val."""
    return _live(ctx, observe="get_recent_metrics").recent_metrics(max(1, min(n, 100)))


@live_agent.tool(prepare=_observing)
def get_val_curve(ctx: RunContext[MonitorDeps]) -> list[dict]:
    """Все оценки на val за раунд: шаг, val loss и метрики задачи."""
    return _live(ctx, observe="get_val_curve").val_curve()


@live_agent.tool(prepare=_observing)
def get_training_config(ctx: RunContext[MonitorDeps]) -> dict:
    """Текущая конфигурация: lr, шедулер, батч, эпохи, частоты оценки, max_grad_norm, адаптер."""
    return _live(ctx, observe="get_training_config").config_summary()


@live_agent.tool(prepare=_observing)
async def wait_for_new_metrics(ctx: RunContext[MonitorDeps], seconds: int = 60) -> dict:
    """Подождать (до 120 с) и получить новые точки метрик — чтобы увидеть эффект поправки. Не больше одного раза."""
    if ctx.deps.waited:
        raise ModelRetry("Ожидание уже было — заверши ответом")
    ctx.deps.waited = True
    new = await _live(ctx).wait_new(max(5, min(seconds, 120)))
    return {"new_points": new, "note": "новых точек нет — процесс на долгом шаге или оценке" if not new else ""}


@live_agent.tool(prepare=_observing)
def get_event_history(ctx: RunContext[MonitorDeps], n: int = 15) -> list[dict]:
    """Недавние аномалии детекторов, твои прошлые решения и подтверждения команд процессом."""
    return _live(ctx, observe="get_event_history").event_history(max(1, min(n, 50)))


async def _act(ctx: RunContext[MonitorDeps], action: str, cmd: str, patch: dict | None, reason: str,
               intervention: bool, verdict: str = "suspicious") -> dict:
    d = ctx.deps
    gate = action if action in ("patch", "stop_early") else None
    if gate and gate not in d.allowed_actions:
        raise ModelRetry(f"Действие {gate} не разрешено настройками; доступны {d.allowed_actions}")
    if intervention and d.interventions_left <= 0:
        raise ModelRetry("Лимит вмешательств исчерпан — только наблюдение")
    decision = MonitorDecision(verdict=verdict, action=action, explanation=reason,
                               patch=patch if action == "patch" else None)
    result = await _live(ctx).send(cmd, patch, decision)
    if not result.get("sent", True):  # процесс команду не получил — это не действие и не вмешательство
        return result | {"interventions_left": d.interventions_left}
    if intervention:
        d.interventions_left -= 1
        d.intervened = True
    d.actions.append(decision)
    return result | {"interventions_left": d.interventions_left}


async def _intervening(ctx: RunContext[MonitorDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    """Вмешательства (lr, клиппинг, останов) видны агенту, только пока они возможны: одно за проверку — после него
    нужно увидеть эффект, а не добавлять ещё; при исчерпанном лимите — не видны вовсе (иначе агент раз за разом
    пробует и получает отказ)."""
    return None if ctx.deps.intervened or ctx.deps.interventions_left <= 0 else tool_def


@live_agent.tool(prepare=_intervening)
async def scale_lr(ctx: RunContext[MonitorDeps], factor: float, reason: str) -> dict:
    """Умножить текущий lr на factor (0.05 < factor < 1) — при скачках loss, плато или переобучении."""
    if not 0.05 < factor < 1.0:
        raise ModelRetry("factor должен быть в (0.05, 1): lr можно только уменьшать")
    return await _act(ctx, "patch", "patch", {"kind": "lr_scale", "factor": factor}, reason, intervention=True)


@live_agent.tool(prepare=_intervening)
async def set_grad_clip(ctx: RunContext[MonitorDeps], max_norm: float, reason: str) -> dict:
    """Ограничить норму градиента (0.05 < max_norm ≤ 1) — при взрыве градиента."""
    if not 0.05 < max_norm <= 1.0:
        raise ModelRetry("max_norm должен быть в (0.05, 1]")
    return await _act(ctx, "patch", "patch", {"kind": "grad_clip", "max_grad_norm": max_norm}, reason, intervention=True)


@live_agent.tool
async def set_eval_every(ctx: RunContext[MonitorDeps], steps: int, reason: str) -> dict:
    """Оценивать на val каждые steps шагов (чаще — точнее следить за плато и переобучением; ≥ 5)."""
    if not 5 <= steps <= 4 * max(ctx.deps.eval_steps, 5):
        raise ModelRetry(f"steps должен быть в [5, {4 * ctx.deps.eval_steps}]")
    return await _act(ctx, "patch", "patch", {"kind": "eval_every", "steps": steps}, reason, intervention=False, verdict="ok")


@live_agent.tool
async def set_log_every(ctx: RunContext[MonitorDeps], steps: int, reason: str) -> dict:
    """Писать train loss каждые steps шагов (1–1000) — плотнее наблюдать за нестабильностью."""
    if not 1 <= steps <= 1000:
        raise ModelRetry("steps должен быть в [1, 1000]")
    return await _act(ctx, "patch", "patch", {"kind": "log_every", "steps": steps}, reason, intervention=False, verdict="ok")


@live_agent.tool
async def request_eval_now(ctx: RunContext[MonitorDeps], reason: str) -> dict:
    """Немедленная оценка на val на ближайшем шаге."""
    return await _act(ctx, "eval_now", "eval_now", None, reason, intervention=False, verdict="ok")


@live_agent.tool
async def save_checkpoint_now(ctx: RunContext[MonitorDeps], reason: str) -> dict:
    """Сохранить чекпоинт на ближайшем шаге (например, перед рискованной поправкой)."""
    return await _act(ctx, "save_now", "save_now", None, reason, intervention=False, verdict="ok")


async def _stopping(ctx: RunContext[MonitorDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    """Остановка доступна, пока разрешена настройками и не применена: на неё не действуют ни лимит вмешательств,
    ни правило «одна поправка за проверку» (агент принимал это правило за запрет остановки)."""
    stopped = any(d.action == "stop_early" for d in ctx.deps.actions)
    return tool_def if "stop_early" in ctx.deps.allowed_actions and not stopped else None


@live_agent.tool(prepare=_stopping)
async def stop_training(ctx: RunContext[MonitorDeps], reason: str) -> dict:
    """Досрочно завершить раунд с сохранением чекпоинта и лучшего адаптера: обучение сломано, переобучается (val
    ухудшается после пика, train loss падает) или больше ничего не даст. Лимит вмешательств не расходует."""
    return await _act(ctx, "stop_early", "stop_and_save", None, reason, intervention=False, verdict="broken")


@round_agent.output_validator
@final_agent.output_validator
def _validate_diagnosis(ctx: RunContext[MonitorDeps], out: RoundAssessment | FinalDiagnosis):
    """Числа в диагнозе — только плейсхолдерами; «закончить» — только если проверка кода согласна."""
    if getattr(out, "recommendation", None) == "finish" and not ctx.deps.code_sufficient:
        raise ModelRetry("Проверка кода говорит, что качество недостаточно: finish недоступен (continue или restart)")
    problems = check_placeholders(out.diagnosis, ctx.deps.values)
    if problems:
        raise ModelRetry("; ".join(problems))
    return out


LIMITS = UsageLimits(request_limit=6)


async def review(deps: MonitorDeps, trigger: str, anomalies: list[dict]) -> MonitorVerdict:
    """Проверка живого обучения: по аномалиям детекторов или плановая. Действия агент выполняет инструментами."""
    stop = "доступна" if "stop_early" in deps.allowed_actions else "запрещена настройками"
    tune = "доступны" if "patch" in deps.allowed_actions and deps.interventions_left > 0 else "недоступны"
    prompt = (f"Задача: {deps.task_type}; основная метрика: {deps.primary_metric}. Повод: {trigger}.\n"
              f"Аномалии детекторов: {anomalies or 'нет'}.\n"
              f"Остановка (stop_training): {stop}; поправки lr и клиппинга: {tune} (осталось {deps.interventions_left}).")
    try:
        return (await llm.run_agent(live_agent, "monitor", prompt, deps=deps,
                                    usage_limits=UsageLimits(request_limit=14, tool_calls_limit=16))).output
    except (UsageLimitExceeded, UnexpectedModelBehavior):  # действия уже применены к процессу — не теряем их
        done = "; ".join(f"{d.action} — {d.explanation}" for d in deps.actions) or "без действий"
        return MonitorVerdict(verdict="suspicious", explanation=f"Агент не завершил проверку ответом. Сделано: {done}")


async def assess_round(deps: MonitorDeps, round_k: int, epochs_done: float, sufficiency_reason: str,
                       curve: list[dict], early_stopped: bool) -> RoundAssessment:
    prompt = (f"Раунд {round_k} завершён, эпох сделано: {epochs_done}; ранний останов: {early_stopped}.\n"
              f"Проверка кода: {'ДОСТАТОЧНО' if deps.code_sufficient else 'НЕДОСТАТОЧНО'} ({sufficiency_reason}).\n"
              f"Доступные значения (плейсхолдеры): {sorted(deps.values)}\nКривая val по раунду: {curve}\n"
              "Напиши диагноз и рекомендацию: finish (только если проверка кода — ДОСТАТОЧНО), "
              "continue (с числом дополнительных эпох) или restart (с другими параметрами).")
    return (await llm.run_agent(round_agent, "monitor", prompt, deps=deps, usage_limits=LIMITS)).output


async def final_diagnosis(deps: MonitorDeps, report_summary: dict) -> FinalDiagnosis:
    prompt = (f"Обучение окончательно остановлено, посчитаны метрики на test.\nСводка: {report_summary}\n"
              f"Доступные значения (плейсхолдеры): {sorted(deps.values)}\n"
              "Напиши итоговый диагноз (3–6 предложений): насколько адаптер лучше baseline'ов, слабые места "
              "(per-class, нераспарсенные ответы), что можно улучшить.")
    return (await llm.run_agent(final_agent, "monitor", prompt, deps=deps, usage_limits=LIMITS)).output


def fallback_round_assessment(code_sufficient: bool, reason: str, default_extra: int) -> RoundAssessment:
    """Без LLM: рекомендация по детерминированной проверке."""
    if code_sufficient:
        return RoundAssessment(diagnosis=f"Агент мониторинга недоступен. Проверка кода: {reason}.", recommendation="finish")
    return RoundAssessment(diagnosis=f"Агент мониторинга недоступен. Проверка кода: {reason}.",
                           recommendation="continue", extra_epochs=default_extra)

