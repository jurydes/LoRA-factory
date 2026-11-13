"""Детерминированный каркас сценария: шаги, карточки подтверждения, фоновые операции, восстановление.

INTAKE → SPEC_CONFIRM → SEARCH → DATASET_CHOICE → PREP → (PREP_CONFIRM) → HPARAMS → CONFIG_CONFIRM → TRAINING →
ROUND_REVIEW → (TRAINING | HPARAMS | EVAL) → DONE. Пауза на ответ пользователя = конец запроса; следующий запрос
загружает session.json и conversation.json."""
from __future__ import annotations

import asyncio
import contextvars
import logging
import random
import shutil
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic_ai import ConversationTypeAdapter

from factory.core import activity, llm
from factory.core.config import AppConfig, get_config
from factory.core.contracts import (
    AutoConfirmation,
    DatasetCandidate,
    EvalReport,
    Manifest,
    Recipe,
    RoundReport,
    RunConfig,
    Seeds,
    SessionSettings,
    TaskSpec,
    utcnow,
)
from factory.core.registry import Registry
from factory.core.state import Activity, Card, SessionState, Step, is_blocking
from factory.core.storage import (
    atomic_write_bytes,
    read_json,
    read_jsonl,
    read_model,
    session_lock,
    write_json,
)
from factory.runs import (
    RunPaths,
    create_run,
    find_reusable,
    load_manifest,
    save_manifest,
)

logger = logging.getLogger(__name__)

# id сессии, чья фоновая операция выполняется в текущей задаче asyncio
_bg_session: contextvars.ContextVar[str | None] = contextvars.ContextVar("bg_session", default=None)
ACTIVITY_KEEP = 8  # строк ленты активности в состоянии сессии
RETRYABLE = (Step.INTAKE, Step.SEARCH, Step.PREP, Step.HPARAMS, Step.EVAL)  # шаги, которые умеет повторить _retry


class Orchestrator:
    def __init__(self, cfg: AppConfig | None = None, registry: Registry | None = None):
        self.cfg = cfg or get_config()
        self.registry = registry or Registry.load(self.cfg)
        self.tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------ хранение

    def session_dir(self, sid: str) -> Path:
        return self.cfg.sessions_dir / sid

    def load(self, sid: str) -> SessionState:
        st = read_model(self.session_dir(sid) / "session.json", SessionState)
        if st is None:
            raise KeyError(f"Сессия {sid} не найдена")
        return st

    def save(self, st: SessionState) -> None:
        st.version += 1
        write_json(self.session_dir(st.session_id) / "session.json", st)

    def _conversation(self, sid: str):
        path = self.session_dir(sid) / "conversation.json"
        return ConversationTypeAdapter.validate_json(path.read_bytes()) if path.exists() else None

    def _save_conversation(self, sid: str, conv) -> None:
        atomic_write_bytes(self.session_dir(sid) / "conversation.json", ConversationTypeAdapter.dump_json(conv))

    def hardware(self):
        """Машина + видеопамять, занятая сейчас другими программами: модель и батч подбираются под то, что
        реально доступно обучению."""
        from factory.core.hardware import detect, with_current_usage
        return with_current_usage(detect(self.cfg.data_dir))

    async def refresh_models(self, force: bool = False) -> int:
        """Каталог моделей HF под эту машину (search/models_hub): реестр дополняется без перезапуска сервиса.
        Свежий кэш (неделя, та же GPU) — без обращений к сети. Возвращает число моделей каталога."""
        if self.cfg.offline:
            return 0
        from factory.search import models_hub
        try:
            hw = await asyncio.to_thread(self.hardware)
            models = await models_hub.refresh(hw, self.cfg, force)
        except Exception:
            logger.exception("Каталог моделей HF не обновлён — подбор идёт по реестру")
            return 0
        self.registry.add_models(models)
        return len(models)

    def _llm_log(self, sid: str):
        return llm.llm_log_to(self.session_dir(sid) / "llm_calls.jsonl")

    def list_sessions(self) -> list[dict]:
        out = []
        for p in sorted(self.cfg.sessions_dir.glob("*/session.json"), key=lambda p: -p.stat().st_mtime):
            d = read_json(p, {})
            out.append({"session_id": d.get("session_id"), "step": d.get("step"),
                        "title": (d.get("spec") or {}).get("description", "")[:80]})
        return out

    def create_session(self, mode: str = "auto") -> SessionState:
        sid = uuid.uuid4().hex[:12]
        st = SessionState(session_id=sid, settings=SessionSettings(mode=mode))
        st.say("Привет! Опишите NLP-задачу, под которую нужно дообучить модель: что подаётся на вход, что нужно "
               "получить, на каком языке. Можно сразу указать ограничения (размер модели, время, лицензию, метод: "
               "LoRA / QLoRA / P-tuning).")
        self.session_dir(sid).mkdir(parents=True, exist_ok=True)
        self.save(st)
        return st

    async def _update(self, sid: str, fn: Callable[[SessionState], Any]) -> SessionState:
        async with session_lock(sid):
            st = self.load(sid)
            r = fn(st)
            if asyncio.iscoroutine(r):
                await r
            self.save(st)
            return st

    async def notify(self, sid: str, text: str, role: str = "system") -> None:
        await self._update(sid, lambda st: st.say(text, role=role))

    async def _background(self, sid: str, label: str, coro_fn: Callable[[], Awaitable[None]]) -> None:
        """Длинная операция вне запроса: прогресс и результат — через session.json (SSE).

        Если вызвана изнутри фоновой операции той же сессии (цепочка шагов в режиме Авто), выполняется
        в той же задаче: флаг занятости снимается один раз, когда закончится вся цепочка."""
        if _bg_session.get() == sid:
            await coro_fn()
            return

        async def runner():
            _bg_session.set(sid)
            try:
                with activity.report_to(lambda kind, who, text, key: self._activity(sid, kind, who, text, key)):
                    await coro_fn()
            except Exception as e:
                logger.exception("Фоновая операция «%s» сессии %s упала", label, sid)
                err = e
                await self._update(sid, lambda st: self._fail_soft(st, label, err))
            finally:
                await self._update(sid, lambda st: setattr(st, "busy", None))
                self.tasks.pop(sid, None)
        self.tasks[sid] = asyncio.create_task(runner())

    def _fail_soft(self, st: SessionState, label: str, e: Exception) -> None:
        st.error = f"{type(e).__name__}: {e}"
        if st.step == Step.TRAINING:  # наблюдение за раундом сломалось: без перехода сессия осталась бы без кнопок
            self._abandon_run(st.run_id, "failed", abort=True)
            st.goto(Step.FAILED)
        retry = self._retryable(st)
        st.say(f"Операция «{label}» завершилась ошибкой: {st.error}." + (" Можно повторить (кнопка «Повторить»)." if retry else ""),
               card=Card(kind="info", payload={"retry": retry, "step": st.step}, blocking=False), role="system")

    @staticmethod
    def _retryable(st: SessionState) -> bool:
        """Есть ли у шага повтор (см. _retry) — кнопку «Повторить» показываем только тогда."""
        return st.step in RETRYABLE or (st.step == Step.FAILED and bool(st.recipe_path))

    def _abandon_run(self, run_id: str | None, status: str, abort: bool = False) -> None:
        """Запуск больше не нужен сессии: статус в реестре, процессу обучения — команда прерваться."""
        if not run_id or not (self.cfg.runs_dir / run_id / "manifest.json").exists():
            return
        paths = RunPaths(self.cfg.runs_dir / run_id)
        if abort:
            from factory.training.supervisor import send_command
            send_command(paths, "abort")
        manifest = load_manifest(paths)
        manifest.status = status
        save_manifest(paths, manifest)

    @staticmethod
    def _close_cards(st: SessionState, action: str) -> None:
        """Пользователь ответил в чате, а не кнопками: карточки, ждавшие решения, больше не действуют."""
        for item in st.items:
            c = item.card
            if c and not c.resolved and c.kind not in ("info", "result"):
                c.resolved, c.resolution = True, {"action": action}

    def _set_busy(self, st: SessionState, text: str) -> None:
        if st.busy and _bg_session.get() != st.session_id:
            raise RuntimeError(f"Уже идёт операция: {st.busy}")
        st.busy, st.busy_since, st.activity = text, utcnow(), []
        st.error = None

    async def _activity(self, sid: str, kind: str, who: str, text: str, key: str | None) -> None:
        def apply(st: SessionState):
            items = st.activity
            if items and items[-1].live:  # «думает» сменяется следующим событием
                items.pop()
            if kind == "idle":
                return
            if kind == "result":
                hit = next((a for a in reversed(items) if key and a.key == key), None)
                if hit:
                    hit.result = text
                return
            items.append(Activity(who=who, text=text, key=key, live=kind == "live"))
            del items[:-ACTIVITY_KEEP]
        await self._update(sid, apply)

    def _auto(self, st: SessionState, card: str, reason: str, resolution: dict | None = None) -> None:
        """Подтверждение системой: карточка закрывается (кнопки на ней больше не нужны) и попадает в журнал."""
        st.auto_confirmations.append(AutoConfirmation(card=card, reason=reason))
        for item in reversed(st.items):
            if item.card and item.card.kind == card and not item.card.resolved:
                item.card.resolved, item.card.resolution = True, {"action": "auto", **(resolution or {})}
                break
        st.say(f"Автоподтверждено ({card}): {reason}", role="system")

    # ------------------------------------------------------------ вход: сообщения

    async def on_message(self, sid: str, text: str) -> SessionState:
        text = text.strip()
        prompt = None

        def handle(st: SessionState):
            nonlocal prompt
            st.say(text, role="user")
            if st.busy:
                st.say(f"Сейчас идёт: {st.busy}. Дождитесь окончания.", role="system")
                return
            if st.step in (Step.DONE, Step.FAILED):
                self._close_cards(st, "new_task")
                st.goto(Step.INTAKE)
                st.spec, st.candidates, st.dataset, st.run_id, st.pending_config = None, [], None, None, None
                st.user_dataset_path, st.restart_of_run_id, st.recipe_path = None, None, None
                st.auto_confirmations = []  # журнал автоподтверждений относится к прошлой задаче
                (self.session_dir(sid) / "conversation.json").unlink(missing_ok=True)
                prompt = f"Новая задача: {text}"
            elif st.step == Step.INTAKE:
                prompt = text
            elif st.step == Step.SPEC_CONFIRM:
                self._close_cards(st, "edit_in_chat")
                st.goto(Step.INTAKE)
                prompt = f"Пользователь хочет поправить постановку: {text}"
            elif st.step == Step.DATASET_CHOICE:
                self._close_cards(st, "none_fit")
                st.goto(Step.INTAKE)
                if st.spec:
                    st.spec = st.spec.model_copy(update={"confirmed_by_user": False})
                prompt = (f"Предложенные датасеты не подошли. Пользователь пишет: {text}. "
                          "Предложи переформулировать задачу или ослабить ограничения.")
            else:
                st.say("На этом шаге используйте кнопки карточки.", role="system")
                return
            self._set_busy(st, "думаю над ответом")

        st = await self._update(sid, handle)
        if prompt is not None:
            await self._background(sid, "диалог", lambda: self._intake_turn(sid, prompt))
        return st

    async def _search_fn(self, sid: str):
        from factory.search import funnel
        done: dict[tuple[str, str | None], list[DatasetCandidate]] = {}  # воронка идёт минуты — повтор не нужен

        async def search(spec: TaskSpec, extra_query: str | None):
            key = (spec.model_dump_json(), extra_query or None)
            if key in done:
                return done[key]
            if extra_query:
                spec = spec.model_copy(update={"description": f"{spec.description} {extra_query}"})
            done[key] = await funnel.search(spec, self.registry, self.cfg,
                                            progress=lambda m: asyncio.ensure_future(self.notify(sid, m)))
            return done[key]
        return search

    async def _intake_turn(self, sid: str, prompt: str) -> None:
        from factory.agents import intake

        st = self.load(sid)
        deps = intake.IntakeDeps(registry=self.registry, spec=st.spec, search=await self._search_fn(sid),
                                 candidates=list(st.candidates), hardware=self.hardware())
        with self._llm_log(sid):
            try:
                output, conv = await intake.run_turn(deps, prompt, self._conversation(sid))
            except Exception as e:
                if st.step == Step.SEARCH:
                    await self.notify(sid, f"Агент диалога недоступен ({type(e).__name__}) — ищу датасеты напрямую.")
                    await self._direct_search(sid)
                    return
                await self.notify(sid, f"LLM недоступна: {type(e).__name__}: {e}")
                return
        self._save_conversation(sid, conv)

        def apply(st: SessionState):
            if isinstance(output, intake.AskUser):
                st.say(output.question, options=output.options)
                if st.step == Step.SEARCH:  # ничего не подошло — назад к диалогу
                    st.goto(Step.INTAKE)
                    if st.spec:
                        st.spec = st.spec.model_copy(update={"confirmed_by_user": False})
            elif isinstance(output, intake.ProposeSpec):
                st.spec = deps.finalized_spec
                if st.step != Step.INTAKE:
                    st.goto(Step.INTAKE)
                st.goto(Step.SPEC_CONFIRM)
                blocking = is_blocking("spec", st.settings.mode)
                st.say(output.comment, card=Card(
                    kind="spec", blocking=blocking,
                    payload={"spec": st.spec.model_dump(mode="json"),
                             "task_title": self.registry.task(st.spec.task_type).title,
                             "models": self.registry.available_models(st.spec.constraints, st.spec.language, self.hardware())}))
            elif isinstance(output, intake.PresentDatasets):
                self._present(st, deps.candidates, output.hf_ids, output.message)

        st = await self._update(sid, apply)
        if st.step == Step.SPEC_CONFIRM and not st.open_cards():
            await self._confirm_spec(sid, None, auto=True)
        elif st.step == Step.DATASET_CHOICE and not st.open_cards():
            await self._auto_choose_dataset(sid)

    def _present(self, st: SessionState, candidates: list[DatasetCandidate], shown: list[str] | None, message: str):
        by_id = {c.hf_id: c for c in candidates}
        order = [by_id[i] for i in (shown or []) if i in by_id]
        order += sorted((c for c in candidates if c.hf_id not in set(shown or [])), key=lambda c: -c.score)
        st.candidates = order
        if st.step != Step.DATASET_CHOICE:
            if st.step != Step.SEARCH:
                st.goto(Step.SEARCH)
            st.goto(Step.DATASET_CHOICE)
        # свой файл пользователя берём и без высокой оценки; найденный — только уверенный кандидат
        confident = bool(order) and (order[0].source == "user" or order[0].score >= self.cfg.search.auto_choose_min_score)
        blocking = is_blocking("dataset_choice", st.settings.mode, {"has_candidates": bool(order), "confident": confident})
        st.say(message or ("Нашёл подходящие датасеты:" if order else "Подходящих датасетов не нашлось."),
               card=Card(kind="dataset_choice", blocking=blocking,
                         payload={"candidates": [c.model_dump(mode="json") for c in order]}))
        if order and not confident and st.settings.mode == "auto":
            st.say(f"Лучший кандидат набрал {order[0].score:g} из 100 (порог автовыбора — "
                   f"{self.cfg.search.auto_choose_min_score:g}): выберите датасет сами, загрузите свой файл или опишите "
                   "задачу иначе.", role="system")

    async def _direct_search(self, sid: str) -> None:
        st = self.load(sid)
        found = await (await self._search_fn(sid))(st.spec, None)
        await self._update(sid, lambda st: self._present(st, found, None, ""))
        st = self.load(sid)
        if not st.open_cards() and st.step == Step.DATASET_CHOICE:
            await self._auto_choose_dataset(sid)

    # ------------------------------------------------------------ вход: действия с карточками

    async def on_action(self, sid: str, card_id: str | None, action: str, payload: dict | None = None) -> SessionState:
        payload = payload or {}
        handlers = {
            "confirm_spec": lambda: self._confirm_spec(sid, payload.get("spec")),
            "choose_dataset": lambda: self._choose_dataset(sid, payload["hf_id"]),
            "none_fit": lambda: self.on_message(sid, payload.get("text") or "Ни один датасет не подходит"),
            "confirm_prep": lambda: self._confirm_prep(sid),
            "edit_recipe": lambda: self._edit_recipe(sid, payload["ops"]),
            "other_dataset": lambda: self._other_dataset(sid),
            "approve_augment": lambda: self._augment_decision(sid, True),
            "decline_augment": lambda: self._augment_decision(sid, False),
            "confirm_config": lambda: self._confirm_config(sid, payload.get("config")),
            "recompute_config": lambda: self._recompute_config(sid),
            "reuse_adapter": lambda: self._reuse_adapter(sid, payload["run_id"]),
            "round_finish": lambda: self._round_decision(sid, "finish", None, "user"),
            "round_continue": lambda: self._round_decision(sid, "continue", int(payload.get("epochs") or self.cfg.training.default_extra_epochs), "user"),
            "round_restart": lambda: self._round_decision(sid, "restart", None, "user"),
            "stop_training": lambda: self._control(sid, "stop_and_save"),
            "abort_training": lambda: self._control(sid, "abort"),
            "retry": lambda: self._retry(sid),
        }
        if action not in handlers:
            raise ValueError(f"Неизвестное действие {action}")
        if card_id:
            def resolve(st: SessionState):
                card = st.find_card(card_id)
                if card is None:
                    raise ValueError("Карточка не найдена")
                if card.resolved:
                    raise ValueError("Карточка уже обработана")
                card.resolved, card.resolution = True, {"action": action, **payload}
            await self._update(sid, resolve)
        try:
            await handlers[action]()
        except Exception:
            if card_id:  # действие не прошло — карточка снова ждёт ответа
                def reopen(st: SessionState):
                    card = st.find_card(card_id)
                    if card is not None:
                        card.resolved, card.resolution = False, None
                await self._update(sid, reopen)
            raise
        return self.load(sid)

    async def update_settings(self, sid: str, settings: dict) -> SessionState:
        def apply(st: SessionState):
            st.settings = SessionSettings.model_validate(st.settings.model_dump() | settings)
            if "mode" in settings:
                st.say(f"Режим работы: {'Эксперт' if st.settings.mode == 'expert' else 'Авто'}.", role="system")
            if "limits" in settings:
                l = st.settings.limits
                st.say(f"Лимиты мониторинга: вмешательств до {l.max_monitor_interventions}, раундов до {l.max_rounds}, "
                       f"разрешено: {', '.join(l.allowed_monitor_actions)}.", role="system")
        return await self._update(sid, apply)

    async def _confirm_spec(self, sid: str, edited: dict | None, auto: bool = False) -> None:
        def apply(st: SessionState):
            if st.step != Step.SPEC_CONFIRM:
                raise ValueError("Сейчас нечего подтверждать")
            spec = TaskSpec.model_validate(st.spec.model_dump() | (edited or {})) if edited else st.spec
            allowed = {r["id"] for r in self.registry.available_models(spec.constraints, spec.language, self.hardware())}
            if spec.base_model_id not in allowed:
                raise ValueError(f"Модель {spec.base_model_id} не подходит под ограничения; доступны: {sorted(allowed)}")
            if self.cfg.offline and "bert_score_f1" in self.registry.task(spec.task_type).metrics:
                model_path = self.cfg.training.bert_score_model
                if not model_path or not Path(model_path).is_dir():
                    raise ValueError("Для этой задачи в закрытом контуре укажите локальный training.bert_score_model в config.yaml")
            st.spec = self.registry.apply_task_metrics(spec).model_copy(update={"confirmed_by_user": True})
            if auto:
                self._auto(st, "spec", "режим Авто: постановка однозначна")
            st.goto(Step.SEARCH)
            if st.user_dataset_path:
                self._set_busy(st, "проверяю загруженный датасет")
                return
            self._set_busy(st, "ищу датасеты")
            st.say("Ищу датасеты под постановку…", role="system")
        st = await self._update(sid, apply)
        if st.user_dataset_path:  # свой датасет вместо поиска по каталогу
            await self._background(sid, "проверка датасета", lambda: self._evaluate_upload(sid, Path(st.user_dataset_path)))
            return
        if self.cfg.offline:
            await self._background(sid, "ожидание датасета", lambda: self._direct_search(sid))
            return
        await self._background(sid, "поиск датасетов", lambda: self._intake_turn(
            sid, "Постановка подтверждена. Найди подходящие датасеты (search_datasets) и покажи их (PresentDatasets)."))

    async def on_upload(self, sid: str, filename: str, data: bytes) -> SessionState:
        """Свой датасет (pandas-совместимый файл): до подтверждения постановки — запоминается и используется
        вместо поиска; при подтверждении или выборе датасета — проверяется сразу."""
        safe = Path(filename).name
        path = self.session_dir(sid) / "uploads" / safe
        evaluate_now = False

        def check(st: SessionState):
            nonlocal evaluate_now
            if st.busy:
                raise RuntimeError(f"Уже идёт операция: {st.busy}")
            if st.step not in (Step.INTAKE, Step.SPEC_CONFIRM, Step.DATASET_CHOICE):
                raise ValueError("Загрузить свой датасет можно до выбора датасета")
            if st.step == Step.INTAKE:
                atomic_write_bytes(path, data)
                st.user_dataset_path = str(path)
                st.say(f"Датасет {safe} получен: он будет использован вместо поиска после уточнения постановки.", role="system")
                return
            if st.step == Step.SPEC_CONFIRM:
                st.spec = self.registry.apply_task_metrics(st.spec).model_copy(update={"confirmed_by_user": True})
                for c in st.open_cards():
                    c.resolved, c.resolution = True, {"action": "upload"}
                st.goto(Step.SEARCH)
            st.user_dataset_path = str(path)
            self._set_busy(st, "проверяю загруженный датасет")
            atomic_write_bytes(path, data)
            evaluate_now = True
        await self._update(sid, check)
        if evaluate_now:
            await self._background(sid, "проверка датасета", lambda: self._evaluate_upload(sid, path))
        return self.load(sid)

    async def _evaluate_upload(self, sid: str, path: Path) -> None:
        from factory.search import funnel

        st = self.load(sid)
        with self._llm_log(sid):
            cand = await funnel.evaluate_user_dataset(path, st.spec, self.registry, self.cfg)
        ok = bool(cand.compat and cand.compat.ok)

        def apply(st: SessionState):
            if not ok:
                st.say(f"Датасет {path.name} не подходит: {'; '.join(cand.compat.issues if cand.compat else [])}", role="system")
            others = [c for c in st.candidates if c.hf_id != cand.hf_id]
            for c in st.open_cards():
                c.resolved, c.resolution = True, {"action": "superseded"}
            self._present(st, [cand] + others if ok else others, [cand.hf_id] if ok else None, f"Ваш датасет {path.name} проверен.")
        st = await self._update(sid, apply)
        if st.step == Step.DATASET_CHOICE and not st.open_cards():
            await self._auto_choose_dataset(sid)

    async def _auto_choose_dataset(self, sid: str) -> None:
        st = self.load(sid)
        if st.candidates:
            top = st.candidates[0]
            await self._update(sid, lambda st: self._auto(st, "dataset_choice", f"выбран лучший кандидат {top.hf_id} (скор {top.score})",
                                                          {"hf_id": top.hf_id}))
            await self._choose_dataset(sid, top.hf_id)

    async def _choose_dataset(self, sid: str, hf_id: str) -> None:
        def apply(st: SessionState):
            cand = next((c for c in st.candidates if c.hf_id == hf_id), None)
            if cand is None:
                raise ValueError(f"Датасета {hf_id} нет среди кандидатов")
            st.dataset = cand
            st.goto(Step.PREP)
            self._set_busy(st, "готовлю данные")
        await self._update(sid, apply)
        await self._background(sid, "подготовка данных", lambda: self._run_prep(sid))

    async def _other_dataset(self, sid: str) -> None:
        def apply(st: SessionState):
            st.goto(Step.DATASET_CHOICE)
            self._present_again(st)
        await self._update(sid, apply)

    def _present_again(self, st: SessionState) -> None:
        st.say("Выберите другой датасет:", card=Card(
            kind="dataset_choice", payload={"candidates": [c.model_dump(mode="json") for c in st.candidates]}))

    # ------------------------------------------------------------ блок 2

    async def _run_prep(self, sid: str) -> None:
        from factory.prep.pipeline import prepare

        st = self.load(sid)
        task = self.registry.task(st.spec.task_type)
        split_seed = random.Random(st.dataset.hf_id + str(st.dataset.revision)).randint(0, 2 ** 31 - 1)  # один на датасет
        with self._llm_log(sid):
            try:
                result = await prepare(st.dataset, st.spec, task, self.session_dir(sid), self.cfg, st.settings.mode,
                                       split_seed, progress=lambda m: asyncio.ensure_future(self.notify(sid, m)))
            except Exception as e:
                logger.exception("Подготовка данных")
                err = f"{type(e).__name__}: {e}"

                def back(st: SessionState):
                    st.say(f"Не удалось подготовить датасет {st.dataset.hf_id}: {err}", role="system")
                    # упавший датасет не предлагаем снова: повторный выбор упал бы так же
                    st.candidates = [c for c in st.candidates if c.hf_id != st.dataset.hf_id]
                    st.dataset = None
                    st.goto(Step.DATASET_CHOICE)
                    self._present_again(st)
                await self._update(sid, back)
                return

        def apply(st: SessionState):
            st.recipe_path = str(self.session_dir(sid) / "recipe.json")
            if result.labels and set(result.labels) != set(st.spec.labels or []):
                st.say(f"Метки в данных: {', '.join(result.labels)} (в постановке были: {', '.join(st.spec.labels or [])}). "
                       "Модель учится и оценивается на метках данных; сопоставление — в рецепте (map_schema).", role="system")
                st.spec = st.spec.model_copy(update={"labels": result.labels})
            if result.agent_error:
                st.say(f"Агент подготовки данных недоступен ({result.agent_error}); рецепт построен по правилам.", role="system")
            if result.augment_proposal:
                st.goto(Step.PREP_CONFIRM)
                p = result.augment_proposal
                st.say(f"Агент предлагает синтез данных ({p['method']}, ~{p['n_rows']} строк, уверенность {p['confidence']:.2f}): {p['reason']}",
                       card=Card(kind="augment", payload=p, blocking=True))
                return
            self._prep_card(st, result.recipe, result.mass_removal)
        st = await self._update(sid, apply)
        if st.step == Step.HPARAMS:
            await self._start_hparams(sid)

    def _prep_card(self, st: SessionState, recipe: Recipe, mass_removal: bool) -> None:
        blocking = is_blocking("prep", st.settings.mode, {"mass_removal": mass_removal})
        text = (f"Подготовка данных: {recipe.rationale}" +
                (f"\nВнимание: рецепт удаляет {recipe.removed_fraction:.0%} train — нужно подтверждение." if mass_removal else ""))
        card = Card(kind="prep", blocking=blocking, payload={"recipe": recipe.model_dump(mode="json"), "mass_removal": mass_removal})
        if st.step != Step.PREP_CONFIRM:
            st.goto(Step.PREP_CONFIRM)
        st.say(text, card=card)
        if not blocking:
            self._auto(st, "prep", f"удалено {recipe.removed_fraction:.1%} train — ниже порога подтверждения")
            st.goto(Step.HPARAMS)

    async def _augment_decision(self, sid: str, approve: bool) -> None:
        from factory.prep.augment import synthesize
        from factory.prep.pipeline import apply_and_save, load_recipe

        st = self.load(sid)
        work = self.session_dir(sid)
        recipe = load_recipe(work)
        task = self.registry.task(st.spec.task_type)
        card = next((i.card for i in reversed(st.items) if i.card and i.card.kind == "augment"), None)
        if approve and card:
            await self._update(sid, lambda st: self._set_busy(st, "генерирую синтетические данные"))
            p = card.payload
            import pandas as pd

            async def gen():
                train = pd.read_parquet(work / "data" / "train.parquet")
                try:
                    with self._llm_log(sid):
                        op = await synthesize(p["method"], train, task.family, int(p["n_rows"]), st.spec.description,
                                              st.spec.language, work, self.cfg.dataprep.augment_workers, recipe.split_seed)
                    if op.n_rows == 0:
                        raise ValueError("все сгенерированные строки отсеяны проверками (копии, язык, длина)")
                except Exception as e:
                    # карточка синтеза уже закрыта — без карточки подготовки сессии некуда было бы идти дальше
                    logger.warning("Синтез данных не удался: %s", e)
                    note = f"Синтез не удался ({type(e).__name__}: {e}) — продолжаю без синтетических данных."
                    await self._update(sid, lambda st: (st.say(note, role="system"), self._prep_card(
                        st, recipe, recipe.removed_fraction > self.cfg.dataprep.confirm_removed_fraction)))
                    if self.load(sid).step == Step.HPARAMS:
                        await self._start_hparams(sid)
                    return
                new = recipe.model_copy(update={"ops": list(recipe.ops) + [op]})
                new = await asyncio.to_thread(apply_and_save, new, work / "data", work, task)
                await self._update(sid, lambda st: (st.say(f"Сгенерировано {op.n_rows} синтетических строк (только train).", role="system"),
                                                     self._prep_card(st, new, new.removed_fraction > self.cfg.dataprep.confirm_removed_fraction)))
                if self.load(sid).step == Step.HPARAMS:
                    await self._start_hparams(sid)
            await self._background(sid, "синтез данных", gen)
            return
        st = await self._update(sid, lambda st: self._prep_card(st, recipe, recipe.removed_fraction > self.cfg.dataprep.confirm_removed_fraction))
        if st.step == Step.HPARAMS:
            await self._start_hparams(sid)

    async def _edit_recipe(self, sid: str, ops: list[dict]) -> None:
        """Режим эксперта: правка операций рецепта (map_schema остаётся первой) и повторное применение к train."""
        from factory.prep.pipeline import apply_and_save, load_recipe

        st = self.load(sid)
        if st.settings.mode != "expert" or st.step != Step.PREP_CONFIRM:
            raise ValueError("Править рецепт можно в режиме эксперта на шаге подтверждения подготовки данных")
        work = self.session_dir(sid)
        recipe = load_recipe(work)
        schema = [o for o in recipe.ops if o.op == "map_schema"]
        edited = Recipe.model_validate(recipe.model_dump(mode="json") | {"ops": [o.model_dump(mode="json") for o in schema]
                                                                          + [o for o in ops if o.get("op") != "map_schema"]})
        task = self.registry.task(st.spec.task_type)
        new = await asyncio.to_thread(apply_and_save, edited.model_copy(update={"built_by": recipe.built_by,
                                                                                "rationale": recipe.rationale + " (рецепт отредактирован пользователем)"}),
                                      work / "data", work, task)
        await self._update(sid, lambda st: self._prep_card(st, new, new.removed_fraction > self.cfg.dataprep.confirm_removed_fraction))

    async def _confirm_prep(self, sid: str) -> None:
        await self._update(sid, lambda st: st.goto(Step.HPARAMS))
        await self._start_hparams(sid)

    # ------------------------------------------------------------ блок 3

    async def _start_hparams(self, sid: str) -> None:
        await self._update(sid, lambda st: self._set_busy(st, "подбираю гиперпараметры и запускаю пилот"))
        await self._background(sid, "подбор гиперпараметров", lambda: self._run_hparams(sid))

    def _hparam_context(self, st: SessionState, recipe: Recipe, train_seed: int):
        from factory.hparams.guard import HParamContext
        from factory.hparams.retuning import load_previous_attempts
        from factory.training.formatting import instruction
        return HParamContext(
            spec=st.spec, task=self.registry.task(st.spec.task_type), model=self.registry.model(st.spec.base_model_id),
            profile=recipe.profile_after, n_train=recipe.profile_after.n_rows, vram_gb=self.hardware().usable_vram_gb, cfg=self.cfg,
            seeds=Seeds(split_seed=recipe.split_seed, train_seed=train_seed),
            previous_attempts=load_previous_attempts(self.cfg.runs_dir, st.restart_of_run_id),
            # запас под промпт: реальная инструкция задачи (~3 символа на токен для русского) + обвязка шаблона
            prompt_overhead_tokens=len(instruction(st.spec, self.registry.task(st.spec.task_type))) // 3 + 32)

    async def _run_hparams(self, sid: str) -> None:
        from factory.hparams import heuristic, strategy
        from factory.hparams.guard import assemble
        from factory.prep.pipeline import load_recipe
        from factory.runs import reuse_key
        from factory.training import launcher

        st = self.load(sid)
        recipe = load_recipe(self.session_dir(sid))
        train_seed = random.randint(0, 2 ** 31 - 1)
        ctx = self._hparam_context(st, recipe, train_seed)

        # запуск создаётся с предварительной конфигурацией: она нужна рабочему процессу для baseline'ов
        if st.run_id and (self.cfg.runs_dir / st.run_id).exists() and load_manifest(RunPaths(self.cfg.runs_dir / st.run_id)).status in ("created", "pilot"):
            paths = RunPaths(self.cfg.runs_dir / st.run_id)
        else:
            provisional = assemble(heuristic.suggest(ctx, None), ctx, "heuristic")
            paths, _ = create_run(self.cfg.runs_dir, st.spec, st.dataset, recipe, provisional, self.session_dir(sid),
                                  sid, restart_of=st.restart_of_run_id)
            await self._update(sid, lambda st: (setattr(st, "run_id", paths.run_id), st.run_ids.append(paths.run_id)))
        ctx.notes.clear()

        if not paths.baselines_val.exists():
            await self.notify(sid, "Считаю baseline'ы на val (мажоритарный/простой и исходная модель с промптом)…")
            res = await launcher.run(paths, "baseline", paths.root / "baseline_status.json")
            if res.get("status") != "ok":
                raise RuntimeError(f"baseline не посчитан: {res.get('message')}")
        baselines = read_json(paths.baselines_val, {})

        await self.notify(sid, f"Подбираю гиперпараметры (стратегия: {self.cfg.hparams.strategy}) и запускаю пилот…")
        with llm.llm_log_to(paths.llm_calls):
            proposal = await strategy.propose(ctx, paths, baselines, self.cfg.hparams.strategy)
        config = proposal.config
        write_json(paths.config, config)
        manifest = load_manifest(paths)
        manifest.config, manifest.seeds, manifest.status = config, config.seeds, "pilot"
        manifest.reuse_key = reuse_key(st.spec, st.dataset, recipe, config)
        save_manifest(paths, manifest)
        reusable = find_reusable(self.cfg.runs_dir, manifest.reuse_key)

        def apply(st: SessionState):
            st.pending_config = config
            st.goto(Step.CONFIG_CONFIRM)
            pilot = proposal.pilot
            text = f"Конфигурация обучения (стратегия: {'эвристика' if proposal.strategy == 'heuristic' else 'агент A3'})."
            if proposal.fallback_reason:
                text += f" Агент недоступен ({proposal.fallback_reason}) — использована эвристика."
            if not pilot.ok:
                text += f"\nПилот не прошёл: {pilot.status} — {pilot.result.get('message', '')[:300]}"
            st.say(text, card=Card(kind="config", blocking=is_blocking("config", st.settings.mode), payload={
                "config": config.model_dump(mode="json"), "pilot": {"status": pilot.status, "fixes": pilot.fixes, **pilot.result},
                "notes": proposal.notes, "baselines_val": baselines, "run_id": paths.run_id,
                "reusable_run": reusable.run_id if reusable else None,
                "editable": st.settings.mode == "expert"}))
        await self._update(sid, apply)

    async def _recompute_config(self, sid: str) -> None:
        await self._update(sid, lambda st: st.goto(Step.HPARAMS))
        await self._start_hparams(sid)

    async def _confirm_config(self, sid: str, edited: dict | None) -> None:
        from factory.hparams.guard import validate_user_config
        from factory.prep.pipeline import load_recipe
        from factory.runs import reuse_key

        st = self.load(sid)
        if st.step != Step.CONFIG_CONFIRM:
            raise ValueError("Сейчас нечего подтверждать")
        config = st.pending_config
        recipe = load_recipe(self.session_dir(sid))
        if edited:
            config = RunConfig.model_validate(config.model_dump(mode="json") | edited)
            ctx = self._hparam_context(st, recipe, config.seeds.train_seed)
            problems = validate_user_config(config, ctx)
            if problems:
                raise ValueError("Конфигурация не прошла проверку: " + "; ".join(problems))
        paths = RunPaths(self.cfg.runs_dir / st.run_id)
        write_json(paths.config, config)
        manifest = load_manifest(paths)
        manifest.config, manifest.status = config, "running"
        # ключ повторного использования — по конфигурации, с которой реально учим (с правками эксперта)
        manifest.reuse_key = reuse_key(st.spec, st.dataset, recipe, config)
        manifest.auto_confirmations = list(st.auto_confirmations)
        save_manifest(paths, manifest)

        def apply(st: SessionState):
            st.pending_config = config
            st.goto(Step.TRAINING)
            self._set_busy(st, "обучение")
            st.say("Обучение запущено. Прогресс и метрики — на панели обучения.", role="system")
        await self._update(sid, apply)
        from factory.training.supervisor import RoundPlan
        await self._background(sid, "обучение", lambda: self._run_training(sid, RoundPlan(round_k=1, epochs=config.schedule.epochs), config))

    async def _reuse_adapter(self, sid: str, run_id: str) -> None:
        prev = load_manifest(RunPaths(self.cfg.runs_dir / run_id))
        current = self.load(sid).run_id
        if current != run_id:  # подготовленный запуск (baseline'ы, пилот) не понадобился
            self._abandon_run(current, "interrupted")

        def apply(st: SessionState):
            st.goto(Step.DONE)
            # решение пользователя (кнопка), а не автоподтверждение — в журнал auto_confirmations не идёт
            st.say(f"Взят готовый адаптер запуска {run_id}: совпали постановка, датасет, рецепт и конфигурация.", role="system")
            report = read_json(RunPaths(self.cfg.runs_dir / run_id).eval_report, {})
            st.say("Готовый адаптер:", card=Card(kind="result", blocking=False, payload={
                "run_id": run_id, "adapter_path": prev.adapter_path, "report": report, "reused": True}))
        await self._update(sid, apply)

    # ------------------------------------------------------------ блок 4

    async def _control(self, sid: str, cmd: str) -> None:
        from factory.training.supervisor import send_command
        st = self.load(sid)
        if st.step != Step.TRAINING or not st.run_id:
            raise ValueError("Обучение не идёт")
        send_command(RunPaths(self.cfg.runs_dir / st.run_id), cmd)
        await self.notify(sid, "Команда отправлена процессу обучения: " +
                          ("остановить с сохранением" if cmd == "stop_and_save" else "прервать"))

    async def _run_training(self, sid: str, plan, config: RunConfig, attach_pid: int | None = None) -> None:
        from factory.training.supervisor import Supervisor

        st = self.load(sid)
        paths = RunPaths(self.cfg.runs_dir / st.run_id)
        pilots = sorted(paths.pilot.glob("status_*.json"))
        pilot = (read_json(pilots[-1], {}) or {}) if pilots else {}
        epoch_s = (pilot["estimated_time_s"] * pilot["steps_per_epoch"] / pilot["total_steps"]
                   if pilot.get("estimated_time_s") and pilot.get("total_steps") else None)
        sup = Supervisor(paths, self.cfg, self.registry, st.settings,
                         notify=lambda text: self.notify(sid, text),
                         pilot_tps=pilot.get("tokens_per_sec"), vram_gb=self.hardware().vram_gb or None, epoch_s=epoch_s)
        with llm.llm_log_to(paths.llm_calls):
            outcome = await sup.run_round(plan, config, attach_pid=attach_pid)
        if outcome.status != "ok":
            manifest = load_manifest(paths)
            manifest.status = "failed"
            save_manifest(paths, manifest)
            tail = ""
            if paths.train_log.exists():
                tail = paths.train_log.read_text(encoding="utf-8", errors="replace")[-1500:]

            def fail(st: SessionState):
                st.error = outcome.result.get("message") or outcome.status
                st.goto(Step.FAILED)
                # «Повторить» на FAILED — новый запуск на тех же данных (см. _retry)
                st.say(f"Обучение не завершилось: {outcome.status}. {st.error}", role="system",
                       card=Card(kind="info", blocking=False, payload={"log_tail": tail, "retry": self._retryable(st)}))
            await self._update(sid, fail)
            return
        await self._round_review(sid, paths, plan, outcome)

    async def _round_review(self, sid: str, paths: RunPaths, plan, outcome) -> None:
        from factory.agents import monitor
        from factory.training.rounds import flatten_values, is_sufficient, render

        st = self.load(sid)
        task = self.registry.task(st.spec.task_type)
        manifest = load_manifest(paths)
        r = outcome.result
        primary = task.primary_metric
        val = r.get("val_metrics", {})
        baselines = r.get("baselines_val", {})
        if manifest.rounds:
            prev_value = manifest.rounds[-1].val_metrics.get(primary)
        else:
            # раунд 1: прирост — между двумя последними оценками кривой (одна и та же подвыборка val); сравнивать
            # итог на полном val с оценкой на подвыборке нельзя — разница выборок выглядела бы как «нет прироста»
            evals = [p for p in read_jsonl(paths.metrics)[0] if primary in (p.get("val_metrics") or {})]
            prev_value = None
            if len(evals) >= 2 and primary in val:
                prev_value = val[primary] - (evals[-1]["val_metrics"][primary] - evals[-2]["val_metrics"][primary])
        sufficient, reason = is_sufficient(primary, val, baselines, st.spec.constraints.target_metric_value,
                                           prev_value, self.cfg.training.min_round_gain)
        values = flatten_values(val=val, baseline=baselines)
        deps = monitor.MonitorDeps(task_type=task.key, primary_metric=primary, values=values, code_sufficient=sufficient)
        curve = [{"step": p["step"], **(p.get("val_metrics") or {}), "val_loss": p.get("val_loss")}
                 for p in read_jsonl(paths.metrics)[0] if p.get("val_metrics")][-10:]
        with llm.llm_log_to(paths.llm_calls):
            try:
                assessment = await monitor.assess_round(deps, plan.round_k, r.get("epochs_done", 0), reason, curve,
                                                        bool(r.get("early_stopped")))
            except Exception as e:
                logger.warning("A4 недоступен для отчёта раунда: %s", e)
                assessment = monitor.fallback_round_assessment(sufficient, reason, self.cfg.training.default_extra_epochs)
        # агент может только ужесточить: «достаточно» остаётся, только если и код, и агент согласны
        sufficient = sufficient and assessment.recommendation == "finish"
        report = RoundReport(
            round=plan.round_k, epochs_done=r.get("epochs_done", 0), val_metrics=val, baselines_val=baselines,
            checkpoint_path=r.get("checkpoint"), adapter_best_path=r.get("adapter_best"), sufficient=sufficient,
            diagnosis=render(assessment.diagnosis, values), recommendation=assessment.recommendation,
            extra_epochs=assessment.extra_epochs)
        write_json(paths.round_dir(plan.round_k) / "report.json", report)
        manifest.rounds.append(report)
        manifest.status = "round_review"
        save_manifest(paths, manifest)

        limits = st.settings.limits
        max_reached = plan.round_k >= limits.max_rounds

        def apply(st: SessionState):
            st.goto(Step.ROUND_REVIEW)
            blocking = is_blocking("round", st.settings.mode, {"sufficient": sufficient})
            st.say(f"Раунд {plan.round_k} завершён.", card=Card(kind="round", blocking=blocking, payload={
                "report": report.model_dump(mode="json"), "primary_metric": primary, "sufficiency_reason": reason,
                "lr_end": r.get("lr_end"), "max_rounds_reached": max_reached,
                "default_extra_epochs": assessment.extra_epochs or self.cfg.training.default_extra_epochs}))
        st = await self._update(sid, apply)
        if not st.open_cards():
            choice = "finish" if max_reached else assessment.recommendation
            await self._update(sid, lambda st: self._auto(st, "round", f"рекомендация: {choice}" +
                                                           (" (достигнут лимит раундов)" if max_reached else ""), {"choice": choice}))
            await self._round_decision(sid, choice, assessment.extra_epochs, "system")

    async def _round_decision(self, sid: str, choice: str, extra_epochs: int | None, who: str) -> None:
        from factory.training.supervisor import RoundPlan

        st = self.load(sid)
        if st.step != Step.ROUND_REVIEW:
            raise ValueError("Сейчас нет раунда на рассмотрении")
        paths = RunPaths(self.cfg.runs_dir / st.run_id)
        manifest = load_manifest(paths)
        last = manifest.rounds[-1]
        if choice == "continue" and len(manifest.rounds) >= st.settings.limits.max_rounds:
            raise ValueError(f"Достигнут лимит раундов ({st.settings.limits.max_rounds})")
        last.choice, last.chosen_by = choice, who
        if choice == "continue":
            last.extra_epochs = extra_epochs or self.cfg.training.default_extra_epochs
        save_manifest(paths, manifest)
        write_json(paths.round_dir(last.round) / "report.json", last)

        if choice == "finish":
            def go(st: SessionState):
                st.goto(Step.EVAL)
                self._set_busy(st, "финальная оценка на test")
            await self._update(sid, go)
            await self._background(sid, "финальная оценка", lambda: self._run_eval(sid))
        elif choice == "continue":
            status = read_json(paths.round_dir(last.round) / "status.json", {}) or {}
            if not last.checkpoint_path:
                raise ValueError("Нет чекпоинта для продолжения")
            plan = RoundPlan(round_k=last.round + 1, epochs=last.extra_epochs, resume_from=Path(last.checkpoint_path),
                             lr=status.get("lr_end"), step_offset=int(status.get("global_step", 0)),
                             epoch_offset=float(last.epochs_done))
            config = read_model(paths.round_dir(last.round) / "config.json", RunConfig) or manifest.config

            def go(st: SessionState):
                st.goto(Step.TRAINING)
                self._set_busy(st, "обучение")
                st.say(f"Продолжаю обучение с текущего места ещё на {last.extra_epochs} эп. (раунд {plan.round_k}).", role="system")
            await self._update(sid, go)
            await self._background(sid, "обучение", lambda: self._run_training(sid, plan, config))
        else:
            manifest.status = "finished" if manifest.adapter_path else "interrupted"
            save_manifest(paths, manifest)

            def go(st: SessionState):
                st.restart_of_run_id = st.run_id
                st.run_id = None
                st.goto(Step.HPARAMS)
                st.say("Перезапуск с другими параметрами: подбираю новую конфигурацию.", role="system")
            await self._update(sid, go)
            await self._start_hparams(sid)

    async def _run_eval(self, sid: str) -> None:
        from factory.agents import monitor
        from factory.training import launcher
        from factory.training.rounds import flatten_values, render

        st = self.load(sid)
        task = self.registry.task(st.spec.task_type)
        paths = RunPaths(self.cfg.runs_dir / st.run_id)
        manifest = load_manifest(paths)
        primary = task.primary_metric
        best = max(manifest.rounds, key=lambda r: r.val_metrics.get(primary, float("-inf")))
        res = await launcher.run(paths, "final_eval", paths.root / "final_eval_status.json", "--adapter", best.adapter_best_path,
                                 "--round", str(best.round))
        if res.get("status") != "ok":
            raise RuntimeError(f"финальная оценка не удалась: {res.get('message')}")
        raw = read_json(paths.eval_report)
        values = flatten_values(test=raw["metrics"], baseline=raw["baselines"], val=best.val_metrics,
                                unparsed={"fraction": raw.get("unparsed_fraction", 0.0)})
        deps = monitor.MonitorDeps(task_type=task.key, primary_metric=primary, values=values)
        summary = {"metrics_test": raw["metrics"], "baselines_test": raw["baselines"], "per_class": raw.get("per_class"),
                   "unparsed_fraction": raw.get("unparsed_fraction"), "rounds": len(manifest.rounds)}
        with llm.llm_log_to(paths.llm_calls):
            try:
                diag = render((await monitor.final_diagnosis(deps, summary)).diagnosis, values)
            except Exception as e:
                logger.warning("A4 недоступен для итогового диагноза: %s", e)
                diag = "Итоговый диагноз агента недоступен; см. метрики."
        limit = st.spec.constraints.max_latency_ms
        if limit and raw.get("latency_ms") and raw["latency_ms"] > limit:
            diag += (f" Задержка ответа {raw['latency_ms']:.0f} мс превышает ограничение {limit} мс — "
                     "возьмите модель меньше, квантизацию или сократите max_new_tokens.")
        report = EvalReport.model_validate(raw | {"diagnosis": diag})
        write_json(paths.eval_report, report)
        if paths.adapter.exists():
            shutil.rmtree(paths.adapter)
        shutil.copytree(best.adapter_best_path, paths.adapter)
        manifest.adapter_path = str(paths.adapter)
        manifest.status = "finished"
        manifest.final_val_metrics = best.val_metrics
        manifest.test_metrics = report.metrics
        save_manifest(paths, manifest)

        def apply(st: SessionState):
            st.goto(Step.DONE)
            st.say("Готово! Итоговые метрики на test:", card=Card(kind="result", blocking=False, payload={
                "run_id": paths.run_id, "adapter_path": str(paths.adapter), "report": report.model_dump(mode="json"),
                "primary_metric": primary, "best_round": best.round}))
        await self._update(sid, apply)

    # ------------------------------------------------------------ повтор и восстановление

    async def _retry(self, sid: str) -> None:
        st = self.load(sid)
        if st.busy:
            raise ValueError(f"Уже идёт: {st.busy}")
        if st.step == Step.HPARAMS or (st.step == Step.FAILED and st.recipe_path):
            if st.step == Step.FAILED:  # повтор после сбоя обучения — новый запуск на тех же данных
                await self._update(sid, lambda st: (st.goto(Step.HPARAMS), setattr(st, "run_id", None)))
            await self._start_hparams(sid)
            return
        last_user = next((i.text for i in reversed(st.items) if i.role == "user"), None)
        retries = {
            Step.SEARCH: ("ищу датасеты", lambda: self._direct_search(sid)),
            Step.PREP: ("готовлю данные", lambda: self._run_prep(sid)),
            Step.EVAL: ("финальная оценка на test", lambda: self._run_eval(sid)),
            Step.INTAKE: ("думаю над ответом", lambda: self._intake_turn(sid, last_user)) if last_user else None,
        }
        if not retries.get(st.step):
            raise ValueError("На этом шаге повторять нечего")
        busy, fn = retries[st.step]
        await self._update(sid, lambda st: self._set_busy(st, busy))
        await self._background(sid, busy, fn)

    async def _reattach(self, m: Manifest, paths: RunPaths) -> None:
        from factory.training.supervisor import RoundPlan

        sid = m.session_id
        if not sid or not (self.session_dir(sid) / "session.json").exists():
            return
        round_k = len(m.rounds) + 1
        config = read_model(paths.round_dir(round_k) / "config.json", RunConfig) or m.config
        started = next((e for e in reversed(read_jsonl(paths.events)[0]) if e.get("type") == "round_started"), {})
        plan = RoundPlan(round_k, float(started.get("epochs") or config.schedule.epochs),
                         Path(started["resume_from"]) if started.get("resume_from") else None, None,
                         int(started.get("step_offset") or 0), float(started.get("epoch_offset") or 0.0))

        def mark(st: SessionState):
            st.busy, st.busy_since, st.activity = "обучение", utcnow(), []
            st.say(f"Сервис перезапускался; обучение (раунд {round_k}) продолжает идти, наблюдение возобновлено.", role="system")
        await self._update(sid, mark)
        await self._background(sid, "обучение", lambda: self._run_training(sid, plan, config, attach_pid=m.pid))

    async def recover(self) -> None:
        """При старте: незавершённые фоновые операции сессий снимаются; запуски «идёт» без живого процесса
        помечаются прерванными и продолжаются с последнего чекпоинта (один раз), а если раунд успел закончиться —
        переходят к его итогам."""
        from factory.training.checkpoints import last_checkpoint

        from factory.hparams.history import iter_manifests
        from factory.prep.pipeline import load_recipe
        from factory.training.launcher import pid_alive
        from factory.training.supervisor import RoundOutcome, RoundPlan, checkpoint_offsets, checkpoint_state

        def interrupted(st: SessionState):
            text = f"Операция «{st.busy}» прервана перезапуском сервиса."
            st.busy = None
            if st.step == Step.PREP_CONFIRM:  # синтез данных: карточка синтеза закрыта — продолжаем без него
                recipe = load_recipe(self.session_dir(st.session_id))
                st.say(text + " Синтетические данные не добавлены.", role="system")
                self._prep_card(st, recipe, recipe.removed_fraction > self.cfg.dataprep.confirm_removed_fraction)
                return
            retry = self._retryable(st)
            st.say(text, role="system", card=Card(kind="info", blocking=False, payload={"retry": retry}))

        for p in self.cfg.sessions_dir.glob("*/session.json"):
            st = read_model(p, SessionState)
            if st and st.busy and st.step != Step.TRAINING:
                was = st.step
                st = await self._update(st.session_id, interrupted)
                if was == Step.PREP_CONFIRM and st.step == Step.HPARAMS:  # Авто: подготовка подтвердилась сама
                    await self._start_hparams(st.session_id)

        watched: set[str] = set()  # сессии, чьё обучение снова под наблюдением
        for m in list(iter_manifests(self.cfg.runs_dir)):
            if m.status != "running":
                continue
            paths = RunPaths(self.cfg.runs_dir / m.run_id)
            if pid_alive(m.pid):  # процесс пережил перезапуск сервиса — снова наблюдаем за ним
                await self._reattach(m, paths)
                watched.add(m.session_id)
                continue
            m.status = "interrupted"
            save_manifest(paths, m)
            sid = m.session_id
            if not sid or not (self.session_dir(sid) / "session.json").exists():
                continue
            st = self.load(sid)
            round_k = len(m.rounds) + 1
            rdir = paths.round_dir(round_k)
            config = read_model(rdir / "config.json", RunConfig) or m.config
            done = read_json(rdir / "status.json", {}) or {}
            if st.run_id == m.run_id and done.get("status") == "ok":
                # раунд дошёл до конца, перезапуск пришёлся на подведение итогов — итоги, а не доучивание
                def mark_review(st: SessionState):
                    st.busy, st.busy_since, st.activity = "итоги раунда", utcnow(), []
                    st.say(f"Сервис перезапускался после конца раунда {round_k} — подвожу его итоги.", role="system")
                await self._update(sid, mark_review)
                outcome = RoundOutcome("ok", done, config)
                await self._background(sid, "итоги раунда", lambda sid=sid, paths=paths, round_k=round_k, outcome=outcome:
                                       self._round_review(sid, paths, RoundPlan(round_k, 0), outcome))
                watched.add(sid)
                continue
            if m.resumed_after_interrupt or st.run_id != m.run_id:
                await self._update(sid, lambda st: (st.say(f"Запуск {m.run_id} прерван и уже продолжался после сбоя — нужен ручной перезапуск.",
                                                            role="system"), st.goto(Step.FAILED), setattr(st, "busy", None)))
                continue
            ckpt = last_checkpoint(rdir)
            prev = m.rounds[-1] if m.rounds else None
            base_step = int((read_json(paths.round_dir(prev.round) / "status.json", {}) or {}).get("global_step", 0)) if prev else 0
            base_epoch = prev.epochs_done if prev else 0.0
            target_epochs = (prev.extra_epochs if prev and prev.extra_epochs else config.schedule.epochs)
            if ckpt:
                step, epoch, lr = checkpoint_state(Path(ckpt))
                step_off, epoch_off = checkpoint_offsets(Path(ckpt), read_jsonl(paths.events)[0], base_step, base_epoch)
                done = epoch_off - base_epoch + epoch  # эпох этого раунда пройдено
                plan = RoundPlan(round_k, max(target_epochs - done, 0.5), Path(ckpt), lr, step_off + step, epoch_off + epoch)
            elif prev and prev.checkpoint_path:
                lr = (read_json(paths.round_dir(prev.round) / "status.json", {}) or {}).get("lr_end")
                plan = RoundPlan(round_k, target_epochs, Path(prev.checkpoint_path), lr, base_step, base_epoch)
            else:
                plan = RoundPlan(round_k, config.schedule.epochs)
            m.resumed_after_interrupt = True
            m.status = "running"
            save_manifest(paths, m)

            def mark(st: SessionState):
                st.busy, st.busy_since, st.activity = "обучение", utcnow(), []
                st.say(f"Сервис перезапускался: обучение прервано и продолжается с последнего чекпоинта (раунд {round_k}).", role="system")
            await self._update(sid, mark)
            await self._background(sid, "обучение", lambda sid=sid, plan=plan, config=config: self._run_training(sid, plan, config))
            watched.add(sid)

        # сессия «обучается», но её запуска нет среди идущих (сбой между шагами) — иначе она висела бы без кнопок
        for p in self.cfg.sessions_dir.glob("*/session.json"):
            st = read_model(p, SessionState)
            if st and st.busy and st.step == Step.TRAINING and st.session_id not in watched:
                await self._update(st.session_id, lambda st: (
                    st.say("Обучение прервано перезапуском сервиса, продолжить его нельзя. Можно повторить на тех же данных.",
                           role="system", card=Card(kind="info", blocking=False, payload={"retry": bool(st.recipe_path)})),
                    st.goto(Step.FAILED), setattr(st, "busy", None)))
