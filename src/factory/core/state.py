"""Состояние сценария сессии (§5): шаги, разрешённые переходы, карточки, режимы работы."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from factory.core.contracts import (
    AutoConfirmation,
    DatasetCandidate,
    RunConfig,
    SessionSettings,
    TaskSpec,
    utcnow,
)


class Step(StrEnum):
    INTAKE = "INTAKE"
    SPEC_CONFIRM = "SPEC_CONFIRM"
    SEARCH = "SEARCH"
    DATASET_CHOICE = "DATASET_CHOICE"
    PREP = "PREP"
    PREP_CONFIRM = "PREP_CONFIRM"
    HPARAMS = "HPARAMS"
    CONFIG_CONFIRM = "CONFIG_CONFIRM"
    TRAINING = "TRAINING"
    ROUND_REVIEW = "ROUND_REVIEW"
    EVAL = "EVAL"
    DONE = "DONE"
    FAILED = "FAILED"


TRANSITIONS: dict[Step, set[Step]] = {
    Step.INTAKE: {Step.INTAKE, Step.SPEC_CONFIRM},
    Step.SPEC_CONFIRM: {Step.INTAKE, Step.SEARCH},
    Step.SEARCH: {Step.DATASET_CHOICE, Step.INTAKE},
    Step.DATASET_CHOICE: {Step.PREP, Step.INTAKE},
    Step.PREP: {Step.PREP_CONFIRM, Step.HPARAMS, Step.DATASET_CHOICE},
    Step.PREP_CONFIRM: {Step.HPARAMS, Step.PREP, Step.DATASET_CHOICE},
    Step.HPARAMS: {Step.CONFIG_CONFIRM},
    Step.CONFIG_CONFIRM: {Step.TRAINING, Step.HPARAMS, Step.DONE},  # DONE — взят готовый адаптер из реестра
    Step.TRAINING: {Step.ROUND_REVIEW},
    Step.ROUND_REVIEW: {Step.TRAINING, Step.HPARAMS, Step.EVAL},
    Step.EVAL: {Step.DONE},
    Step.DONE: {Step.INTAKE},
    Step.FAILED: {Step.INTAKE, Step.HPARAMS},  # HPARAMS — повтор с подготовленными данными
}


class TransitionError(RuntimeError):
    pass


def check_transition(src: Step, dst: Step) -> None:
    if dst == Step.FAILED:
        return
    if dst not in TRANSITIONS[src]:
        raise TransitionError(f"Переход {src} → {dst} не предусмотрен сценарием")


CardKind = Literal["spec", "dataset_choice", "prep", "augment", "config", "round", "result", "info"]


class Card(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    kind: CardKind
    payload: dict[str, Any] = Field(default_factory=dict)
    blocking: bool = True
    resolved: bool = False
    resolution: dict[str, Any] | None = None


class ChatItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    role: Literal["user", "assistant", "system"]
    text: str = ""
    card: Card | None = None
    options: list[str] = Field(default_factory=list)
    time: Any = Field(default_factory=utcnow)


class Activity(BaseModel):
    """Строка ленты «что происходит сейчас» (см. core/activity.py)."""
    time: datetime = Field(default_factory=utcnow)
    who: str = ""
    text: str
    result: str | None = None
    key: str | None = None
    live: bool = Field(default=False, description="Временная строка («думает»): заменяется следующим событием")


class SessionState(BaseModel):
    session_id: str
    step: Step = Step.INTAKE
    settings: SessionSettings = Field(default_factory=SessionSettings)
    spec: TaskSpec | None = None
    candidates: list[DatasetCandidate] = Field(default_factory=list)
    dataset: DatasetCandidate | None = None
    user_dataset_path: str | None = Field(default=None, description="Файл пользователя, загруженный до подтверждения постановки")
    recipe_path: str | None = None
    pending_config: RunConfig | None = None
    run_id: str | None = None
    run_ids: list[str] = Field(default_factory=list)
    restart_of_run_id: str | None = None
    items: list[ChatItem] = Field(default_factory=list)
    auto_confirmations: list[AutoConfirmation] = Field(default_factory=list)
    busy: str | None = Field(default=None, description="Текст фоновой операции, если она идёт")
    busy_since: datetime | None = None
    activity: list[Activity] = Field(default_factory=list, description="Последние шаги текущей операции")
    error: str | None = None
    version: int = 0

    def goto(self, dst: Step) -> None:
        check_transition(self.step, dst)
        self.step = dst

    def say(self, text: str, card: Card | None = None, options: list[str] | None = None,
            role: Literal["assistant", "system", "user"] = "assistant") -> ChatItem:
        item = ChatItem(role=role, text=text, card=card, options=options or [])
        self.items.append(item)
        return item

    def find_card(self, card_id: str) -> Card | None:
        for item in reversed(self.items):
            if item.card and item.card.id == card_id:
                return item.card
        return None

    def open_cards(self) -> list[Card]:
        return [i.card for i in self.items if i.card and i.card.blocking and not i.card.resolved]


def is_blocking(card: CardKind, mode: Literal["auto", "expert"], context: dict[str, Any] | None = None) -> bool:
    """Нужно ли ждать подтверждения пользователя.

    Эксперт — подтверждаются все карточки. Авто — только точки необратимости: массовое удаление данных,
    итоговая конфигурация перед обучением, синтез данных без уверенного решения агента, выбор датасета,
    когда уверенного кандидата нет.
    """
    context = context or {}
    if card in ("info", "result"):
        return False
    if mode == "expert":
        return True
    if card == "config":
        return True
    if card == "prep":
        return bool(context.get("mass_removal", False))
    if card == "augment":
        return not context.get("confident", False)
    if card == "dataset_choice":  # нет кандидатов или лучший слаб — выбор за пользователем
        return not (context.get("has_candidates", True) and context.get("confident", True))
    if card == "round":  # §3.4: достаточно — система завершает сама; иначе спрашивает пользователя
        return not context.get("sufficient", False)
    return False
